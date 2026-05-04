import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import Blip2Processor, TrainingArguments, Trainer
from transformers.trainer_utils import get_last_checkpoint

from model import load_model, apply_lora
from dataset import CLEVRDataset


class AnswerWeightedTrainer(Trainer):
    def __init__(self, *args, answer_weight=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.answer_weight = answer_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.size())

        # Upweight the final answer token and 2 preceding tokens
        weights = torch.ones_like(loss)
        for b in range(shift_labels.size(0)):
            valid = (shift_labels[b] != -100).nonzero(as_tuple=True)[0]
            if len(valid) > 0:
                last = valid[-1].item()
                for offset in range(3):
                    idx = last - offset
                    if idx >= 0:
                        weights[b, idx] = self.answer_weight

        loss = (loss * weights).sum() / (weights * (shift_labels != -100)).sum()
        return (loss, outputs) if return_outputs else loss


def train(args):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=False)
    model = load_model()
    model = apply_lora(model)

    train_ds = CLEVRDataset(args.data, args.image_dir, processor, mode=args.mode)

    total_steps  = (len(train_ds) // (args.batch_size * args.grad_accum)) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        dataloader_num_workers=2,
    )

    answer_weight = 5.0 if args.mode == "cot" else 1.0
    trainer = AnswerWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        answer_weight=answer_weight,
    )

    last_ckpt = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None
    trainer.train(resume_from_checkpoint=last_ckpt)
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True)
    parser.add_argument("--image_dir",   required=True)
    parser.add_argument("--mode",        default="answer_only", choices=["answer_only", "cot"])
    parser.add_argument("--output_dir",  required=True)
    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--lr",          type=float, default=5e-5)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--grad_accum",  type=int,   default=2)
    parser.add_argument("--warmup_ratio",type=float, default=0.06)
    args = parser.parse_args()
    train(args)

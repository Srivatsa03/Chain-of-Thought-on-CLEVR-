import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image


class CLEVRDataset(Dataset):
    def __init__(self, json_path, image_dir, processor, mode="answer_only",
                 max_question_length=64, max_answer_length=None):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_dir = image_dir
        self.processor = processor
        self.mode = mode
        self.max_question_length = max_question_length
        # CoT outputs are much longer than plain answers
        if max_answer_length is None:
            self.max_answer_length = 32 if mode == "answer_only" else 200
        else:
            self.max_answer_length = max_answer_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(os.path.join(self.image_dir, item["image_filename"])).convert("RGB")

        question_text = f"Question: {item['question']} Answer:"
        target_text = item["answer"] if self.mode == "answer_only" else item["cot_answer"]

        # Encode image
        pixel_values = self.processor.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].squeeze()

        # Tokenize question (fixed length)
        q_enc = self.processor.tokenizer(
            question_text,
            max_length=self.max_question_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        q_ids  = q_enc["input_ids"].squeeze()       # [max_question_length]
        q_mask = q_enc["attention_mask"].squeeze()   # [max_question_length]

        # Tokenize answer WITH special tokens so OPT gets proper BOS context.
        # But mask BOS in labels so model never learns to predict BOS as first token.
        # (OPT BOS id == EOS id == 2; predicting BOS first causes eos_token_id to fire immediately.)
        a_enc = self.processor.tokenizer(
            target_text,
            max_length=self.max_answer_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        a_ids  = a_enc["input_ids"].squeeze()        # [max_answer_length]
        a_mask = a_enc["attention_mask"].squeeze()   # [max_answer_length]

        # Replace first PAD with explicit EOS so model learns to stop generating.
        pad_id = self.processor.tokenizer.pad_token_id
        eos_id = self.processor.tokenizer.eos_token_id
        pad_positions = (a_ids == pad_id).nonzero(as_tuple=True)[0]
        first_pad = pad_positions[0].item() if len(pad_positions) > 0 else None
        if first_pad is not None:
            a_ids[first_pad] = eos_id   # explicit EOS stop token
            a_mask[first_pad] = 1       # attend to EOS position

        # input_ids = [question | answer]  →  BLIP-2 LM sees [32 Q-Former | question | answer]
        input_ids      = torch.cat([q_ids, a_ids])    # [Q_len + A_len]
        attention_mask = torch.cat([q_mask, a_mask])  # [Q_len + A_len]

        # Labels: mask BOS at position 0 (never train to predict BOS/EOS as first token),
        # keep answer tokens + EOS as targets, mask padding after EOS.
        a_labels = a_ids.clone()
        a_labels[0] = -100  # mask BOS — prevents early EOS firing at inference
        if first_pad is not None:
            a_labels[first_pad + 1:] = -100  # mask padding after EOS

        labels = torch.cat([
            torch.full((self.max_question_length,), -100, dtype=torch.long),
            a_labels,
        ])

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


def build_collator(processor):
    def collate_fn(batch):
        return {
            "pixel_values":   torch.stack([b["pixel_values"]   for b in batch]),
            "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels":         torch.stack([b["labels"]         for b in batch]),
        }
    return collate_fn

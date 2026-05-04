import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

MODEL_NAME = "Salesforce/blip2-opt-2.7b"


def load_model(model_name=MODEL_NAME):
    processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    # Freeze vision encoder and Q-Former; only train the language model via LoRA
    for name, param in model.named_parameters():
        if "language_model" not in name:
            param.requires_grad = False
    return model, processor


def apply_lora(model, r=16, lora_alpha=32, lora_dropout=0.1):
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        # Target OPT attention projections inside the language model
        target_modules=["q_proj", "v_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def load_for_inference(checkpoint_path, model_name=MODEL_NAME):
    processor = Blip2Processor.from_pretrained(model_name, use_fast=False)
    base_model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, processor

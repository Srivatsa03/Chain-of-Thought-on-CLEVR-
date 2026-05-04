import argparse
import json
import os
import re
import sys

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration

sys.path.insert(0, os.path.dirname(__file__))

ORDERED_ANSWERS = [
    'yes', 'no',
    'red', 'blue', 'green', 'purple', 'gray', 'cyan', 'brown', 'yellow',
    'cube', 'sphere', 'cylinder',
    'small', 'large',
    'rubber', 'metal',
    '10', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
]
MAX_QUESTION_LENGTH = 64
MODEL_NAME = "Salesforce/blip2-opt-2.7b"


def extract_answer(text):
    text = text.strip().lower()
    for tok in ORDERED_ANSWERS:
        if text.startswith(tok):
            return tok
    for tok in ORDERED_ANSWERS:
        if re.search(r'\b' + tok + r'\b', text):
            return tok
    return text.split()[0] if text.split() else ""


def evaluate_zeroshot(data_path, image_dir, output_path):
    print(f"Loading base BLIP-2 model (no LoRA, no fine-tuning)...")
    processor = Blip2Processor.from_pretrained(MODEL_NAME, use_fast=False)
    model = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Model loaded on {device}")

    with open(data_path) as f:
        data = json.load(f)

    eos_token_id = processor.tokenizer.eos_token_id
    correct = 0
    results = []

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    live_path = output_path.replace(".json", "_live.jsonl")
    live_f = open(live_path, "w")

    for item in tqdm(data, desc="Evaluating (zero-shot)"):
        image = Image.open(os.path.join(image_dir, item["image_filename"])).convert("RGB")
        prompt = f"Question: {item['question']} Answer:"

        pixel_values = processor.image_processor(
            images=image, return_tensors="pt"
        )["pixel_values"].to(device)

        q_enc = processor.tokenizer(
            prompt, max_length=MAX_QUESTION_LENGTH,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = q_enc["input_ids"].to(device)
        attention_mask = q_enc["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=5,
                num_beams=1,
                eos_token_id=eos_token_id,
            )

        new_tokens     = generated_ids[0][MAX_QUESTION_LENGTH:]
        generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        pred     = extract_answer(generated_text)
        gt       = item["answer"].lower()
        is_correct = pred == gt
        correct += is_correct

        result = {
            "question":      item["question"],
            "question_type": item.get("question_type", "unknown"),
            "program_depth": item.get("program_depth", 0),
            "ground_truth":  gt,
            "prediction":    pred,
            "generated_text":generated_text,
            "correct":       is_correct,
        }
        results.append(result)
        live_f.write(json.dumps(result) + "\n")
        live_f.flush()

    live_f.close()

    total    = len(data)
    accuracy = correct / total

    from collections import defaultdict
    type_correct  = defaultdict(int)
    type_total    = defaultdict(int)
    depth_correct = defaultdict(int)
    depth_total   = defaultdict(int)

    for r in results:
        type_total[r["question_type"]]  += 1
        type_correct[r["question_type"]] += r["correct"]
        bucket = "short (<=4)" if r["program_depth"] <= 4 else "long (>=5)"
        depth_total[bucket]  += 1
        depth_correct[bucket] += r["correct"]

    print(f"\n{'='*40}")
    print(f"Mode:               zero-shot (no fine-tuning)")
    print(f"Overall Accuracy:   {accuracy:.4f} ({correct}/{total})")
    print("\nPer question-type accuracy:")
    type_acc = {}
    for qtype in sorted(type_total):
        acc = type_correct[qtype] / type_total[qtype]
        type_acc[qtype] = acc
        print(f"  {qtype:<20} {acc:.4f} ({type_correct[qtype]}/{type_total[qtype]})")
    print("\nPer reasoning depth accuracy:")
    depth_acc = {}
    for bucket in sorted(depth_total):
        acc = depth_correct[bucket] / depth_total[bucket]
        depth_acc[bucket] = acc
        print(f"  {bucket:<20} {acc:.4f} ({depth_correct[bucket]}/{depth_total[bucket]})")
    print(f"{'='*40}\n")

    with open(output_path, "w") as f:
        json.dump({
            "mode":         "zero_shot",
            "accuracy":     accuracy,
            "correct":      correct,
            "total":        total,
            "type_accuracy": type_acc,
            "depth_accuracy": depth_acc,
            "results":      results,
        }, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output",    default="results/zeroshot_results.json")
    args = parser.parse_args()
    evaluate_zeroshot(args.data, args.image_dir, args.output)

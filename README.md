# Does Chain-of-Thought Supervision Improve Compositional Visual Question Answering?

A controlled study comparing answer-only vs. chain-of-thought fine-tuning on the CLEVR benchmark using BLIP-2 + LoRA.

---

## Overview

This project investigates whether training a vision-language model with explicit chain-of-thought (CoT) reasoning traces improves accuracy on compositional visual question answering. We fine-tune BLIP-2 (OPT-2.7B) with LoRA on 50,000 CLEVR training samples under three conditions and evaluate on 2,000 validation samples.

---

## Results

| Model | Supervision | Accuracy |
|---|---|---|
| BLIP-2 (base) | Zero-shot | 8.75% |
| BLIP-2 + LoRA | Answer-Only | **45.95%** |
| BLIP-2 + LoRA | Chain-of-Thought | 28.90% |

**Key finding:** CoT outperforms answer-only on short reasoning chains (depth ≤ 4: 51.43% vs 34.29%) but underperforms on long chains (depth ≥ 5: 28.50% vs 46.16%).

---

## Project Structure

```
├── src/
│   ├── model.py              # BLIP-2 + LoRA model setup
│   ├── dataset.py            # Dataset loader for answer-only and CoT modes
│   ├── train.py              # Training with AnswerWeightedTrainer
│   ├── evaluate.py           # Evaluation for fine-tuned models
│   ├── evaluate_zeroshot.py  # Zero-shot baseline evaluation
│   ├── program_to_cot.py     # CLEVR program → CoT trace conversion
│   ├── build_dataset.py      # Dataset preprocessing pipeline
│   ├── generate_plots.py     # Result visualization (6 plots)
│   └── analyze.py            # Result analysis utilities
├── results/
│   ├── answer_only_2k_results.json
│   ├── cot_2k_final_results.json
│   ├── zeroshot_2k_results.json
│   └── plots/                # All figures used in the report
├── report.tex                # Full ACL-format project report
├── references.bib            # Bibliography
├── requirements.txt          # Python dependencies
├── setup_ec2.sh              # AWS EC2 environment setup
└── run_all.sh                # End-to-end pipeline script
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download CLEVR dataset
```bash
wget https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.tar.gz
tar -xf CLEVR_v1.0.tar.gz -C data/
```

### 3. Build datasets
```bash
python src/build_dataset.py --split train --max_samples 50000 --output data/train_50k.json
python src/build_dataset.py --split val   --max_samples 2000   --output data/val_2k.json
```

### 4. Train models
```bash
# Answer-only
python src/train.py --data data/train_50k.json --image_dir data/images/val \
    --mode answer_only --epochs 10 --lr 5e-5 --output_dir checkpoints/answer_only

# Chain-of-Thought
python src/train.py --data data/train_50k.json --image_dir data/images/val \
    --mode cot --epochs 3 --lr 1e-5 --output_dir checkpoints/cot
```

### 5. Evaluate
```bash
python src/evaluate.py --data data/val_2k.json --image_dir data/images/val \
    --checkpoint checkpoints/answer_only --output results/answer_only_2k_results.json

python src/evaluate.py --data data/val_2k.json --image_dir data/images/val \
    --checkpoint checkpoints/cot --mode cot --output results/cot_2k_final_results.json

python src/evaluate_zeroshot.py --data data/val_2k.json --image_dir data/images/val \
    --output results/zeroshot_2k_results.json
```

### 6. Generate plots
```bash
python src/generate_plots.py
```

Or run everything at once:
```bash
bash run_all.sh
```

---

## Model Details

- **Backbone:** `Salesforce/blip2-opt-2.7b`
- **Vision Encoder:** ViT-L/14 (frozen)
- **Q-Former:** 32 learned query tokens (frozen)
- **Language Model:** OPT-2.7B (fine-tuned via LoRA)
- **LoRA:** rank=16, alpha=32, target modules: q_proj, v_proj
- **Trainable parameters:** ~3.7M out of 3.7B
- **Training hardware:** AWS EC2 g6xe.large (NVIDIA L40S 48GB)

---

## Dataset

CLEVR v1.0 — [https://cs.stanford.edu/people/jcjohns/clevr/](https://cs.stanford.edu/people/jcjohns/clevr/)

---

## Authors

**Srivatsa Kamballa** and **Niket Pathak**  
Department of Computer Science, University of Illinois Chicago  
CS533 — Deep Learning for NLP, Spring 2026

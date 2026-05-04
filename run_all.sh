#!/bin/bash
set -e

DATA_DIR="data"
IMAGE_DIR="$DATA_DIR/images/val"
TRAIN_JSON="$DATA_DIR/train_50k.json"
VAL_JSON="$DATA_DIR/val_2k.json"

echo "=== Step 1: Build datasets ==="
python src/build_dataset.py --split train --max_samples 50000 --output $TRAIN_JSON
python src/build_dataset.py --split val   --max_samples 2000   --output $VAL_JSON

echo "=== Step 2: Train Answer-Only model ==="
python src/train.py \
    --data $TRAIN_JSON \
    --image_dir $IMAGE_DIR \
    --mode answer_only \
    --epochs 10 \
    --lr 5e-5 \
    --output_dir checkpoints/answer_only

echo "=== Step 3: Train Chain-of-Thought model ==="
python src/train.py \
    --data $TRAIN_JSON \
    --image_dir $IMAGE_DIR \
    --mode cot \
    --epochs 3 \
    --lr 1e-5 \
    --output_dir checkpoints/cot

echo "=== Step 4: Evaluate Answer-Only model ==="
python src/evaluate.py \
    --data $VAL_JSON \
    --image_dir $IMAGE_DIR \
    --checkpoint checkpoints/answer_only \
    --output results/answer_only_2k_results.json

echo "=== Step 5: Evaluate CoT model ==="
python src/evaluate.py \
    --data $VAL_JSON \
    --image_dir $IMAGE_DIR \
    --checkpoint checkpoints/cot \
    --mode cot \
    --output results/cot_2k_final_results.json

echo "=== Step 6: Evaluate Zero-shot baseline ==="
python src/evaluate_zeroshot.py \
    --data $VAL_JSON \
    --image_dir $IMAGE_DIR \
    --output results/zeroshot_2k_results.json

echo "=== Step 7: Generate plots ==="
python src/generate_plots.py

echo "=== All done. Results in results/, plots in results/plots/ ==="

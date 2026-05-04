#!/bin/bash
# Run this once after SSHing into your g6xe.large EC2 instance.
# Assumes Deep Learning AMI (PyTorch 2.x, CUDA 12.1) on Ubuntu.

set -e

echo "===== Setting up CS533 Project on EC2 ====="

# 1. Create and activate conda environment
conda create -y -n clevr_cot python=3.10
source activate clevr_cot

# 2. Install dependencies
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121
pip install \
    transformers==4.37.2 \
    datasets==2.17.0 \
    accelerate==0.26.1 \
    peft==0.8.2 \
    Pillow==10.2.0 \
    numpy==1.26.3 \
    pandas==2.2.0 \
    tqdm==4.66.1 \
    scikit-learn==1.4.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.2 \
    huggingface_hub==0.20.3 \
    scipy==1.12.0 \
    jupyter==1.0.0 \
    ipykernel==6.29.0

echo "===== Dependencies installed ====="

# 3. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"

# 4. Create directory structure
mkdir -p data/clevr
mkdir -p data/processed
mkdir -p checkpoints/answer_only
mkdir -p checkpoints/chain_of_thought
mkdir -p results

# 5. Download CLEVR dataset (~18 GB — takes ~5 min on EC2)
echo "===== Downloading CLEVR dataset ====="
cd data/clevr
wget -q --show-progress https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
echo "Unzipping CLEVR..."
unzip -q CLEVR_v1.0.zip
rm CLEVR_v1.0.zip
cd ../..

echo "===== CLEVR downloaded and extracted ====="
echo "Directory: data/clevr/CLEVR_v1.0/"

# 6. Preprocess dataset (generates data/processed/train.json and val.json)
echo "===== Preprocessing dataset ====="
python src/build_dataset.py \
    --train_samples 50000 \
    --val_samples 10000

echo ""
echo "===== Setup complete! ====="
echo "Next: run ./run_all.sh inside a tmux session"
echo "      tmux new -s train"
echo "      ./run_all.sh"

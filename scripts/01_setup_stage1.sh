#!/bin/bash
# Stage 1 Setup: Installs LLaMA-Factory and dependencies on the cloud GPU instance.
# Run this ONCE on your A100/4090 instance before training.

set -e

echo "=== Stage 1 Setup: LLaMA-Factory for Qwen 3 Omni Thinker ==="

# --- System dependencies ---
apt-get update -q && apt-get install -y -q git wget ffmpeg libsndfile1

# --- Python environment ---
if command -v conda &>/dev/null; then
    conda create -n llama_factory python=3.11 -y
    conda activate llama_factory
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

# --- Install PyTorch 12.8 EXPLICITLY before LLaMA-Factory ---
# This prevents pip from accidentally pulling the 13.0 default
echo "Installing PyTorch for CUDA 12.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# --- Clone LLaMA-Factory ---
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

# --- Install remaining dependencies ---
# We omit 'torch' from the brackets here so it doesn't overwrite our 12.8 installation
pip install -e ".[metrics,bitsandbytes]" --upgrade

# --- Flash Attention 2 ---
# ACTUAL pre-built wheel fetching to avoid the CUDA source compilation
echo "Fetching pre-compiled Flash Attention..."
pip install flash-attn-3 --index-url https://download.pytorch.org/whl/cu128

# --- Verify GPU is visible ---
python -c "import torch; print(f'\nCUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}'); print(f'PyTorch Version: {torch.__version__}')"

echo ""
echo "=== Setup complete. Activate your env before running training: ==="
echo "  source .venv/bin/activate  (or: conda activate llama_factory)"
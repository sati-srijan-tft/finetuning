#!/bin/bash
# Stage 1 Setup: Installs LLaMA-Factory and dependencies on the cloud GPU instance.
# Run this ONCE on your A100/4090 instance before training.

set -e

echo "=== Stage 1 Setup: LLaMA-Factory for Qwen 3 Omni Thinker ==="

# --- System dependencies ---
apt-get update -q && apt-get install -y -q git wget ffmpeg libsndfile1

# --- Python environment (use conda if available, else venv) ---
if command -v conda &>/dev/null; then
    conda create -n llama_factory python=3.11 -y
    conda activate llama_factory
else
    python3 -m venv .venv
    source .venv/bin/activate
fi

# --- Clone LLaMA-Factory ---
if [ ! -d "LLaMA-Factory" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory

# --- Install with torch and bitsandbytes (4-bit quantization) ---
pip install -e ".[torch,metrics,bitsandbytes]" --upgrade

# --- Flash Attention 2 (required for flash_attn: fa2 in config) ---
# A100 supports FA2 natively; pre-built wheel avoids a ~30-min CUDA compile
pip install flash-attn --no-build-isolation

# --- Verify GPU is visible ---
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\"}')"

echo ""
echo "=== Setup complete. Activate your env before running training: ==="
echo "  source .venv/bin/activate  (or: conda activate llama_factory)"

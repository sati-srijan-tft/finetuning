#!/bin/bash
# Stage 1 Training: Fine-tunes the Qwen 3 Omni Thinker using LLaMA-Factory.
# Run from inside the LLaMA-Factory directory after 01_setup_stage1.sh.
#
# Usage:
#   cd LLaMA-Factory
#   bash ../scripts/03_run_stage1_training.sh
#
# For multi-GPU (e.g., 2x RTX 4090), set CUDA_VISIBLE_DEVICES and use torchrun:
#   CUDA_VISIBLE_DEVICES=0,1 bash ../scripts/03_run_stage1_training.sh --multi_gpu

set -e

MULTI_GPU=false
for arg in "$@"; do
    [[ "$arg" == "--multi_gpu" ]] && MULTI_GPU=true
done

CONFIG_PATH="../configs/stage1_lora_config.yaml"

echo "=== Stage 1: Thinker Fine-Tuning ==="
echo "Config: $CONFIG_PATH"
echo "Multi-GPU: $MULTI_GPU"

# Verify config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found at $CONFIG_PATH"
    exit 1
fi

# Verify dataset was placed correctly
if [ ! -f "data/data.jsonl" ]; then
    echo "ERROR: data/data.jsonl not found. Run 02_prepare_stage1_data.py first."
    exit 1
fi

if [ "$MULTI_GPU" = true ]; then
    # Multi-GPU: use torchrun (adjust --nproc_per_node to your GPU count)
    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Launching on $N_GPU GPUs ..."
    torchrun \
        --nproc_per_node="$N_GPU" \
        --master_port=29500 \
        src/train.py "$CONFIG_PATH"
else
    # Single GPU
    llamafactory-cli train "$CONFIG_PATH"
fi

echo ""
echo "=== Stage 1 training complete. ==="
echo "LoRA adapters saved to: outputs/stage1_lora"
echo "Next: run 04_merge_lora_adapters.sh to create the merged model."

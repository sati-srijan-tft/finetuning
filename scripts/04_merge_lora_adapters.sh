#!/bin/bash
# Stage 1 Post-Training: Merges LoRA adapters back into the base model weights.
# The merged model is a standard HuggingFace model ready for Stage 2 conversion.
#
# Run from inside the LLaMA-Factory directory.
#   cd LLaMA-Factory
#   bash ../scripts/04_merge_lora_adapters.sh

set -e

MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Instruct"          # base model (HF Hub or local path)
ADAPTER_PATH="./outputs/stage1_lora"      # LoRA adapter output from Stage 1
MERGED_PATH="./outputs/stage1_merged"     # destination for merged weights

echo "=== Merging LoRA Adapters into Base Model ==="
echo "Base model:  $MODEL_PATH"
echo "Adapters:    $ADAPTER_PATH"
echo "Output:      $MERGED_PATH"

if [ ! -d "$ADAPTER_PATH" ]; then
    echo "ERROR: Adapter directory not found: $ADAPTER_PATH"
    echo "       Complete Stage 1 training first."
    exit 1
fi

llamafactory-cli export \
    --model_name_or_path "$MODEL_PATH" \
    --adapter_name_or_path "$ADAPTER_PATH" \
    --template qwen \
    --finetuning_type lora \
    --export_dir "$MERGED_PATH" \
    --export_size 4 \
    --export_device cpu \
    --export_legacy_format false \
    --trust_remote_code true

echo ""
echo "=== Merge complete. ==="
echo "Merged model saved to: $MERGED_PATH"
echo "Next: run 05_convert_to_nemo.sh to prepare for Stage 2."

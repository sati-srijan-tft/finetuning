#!/bin/bash
# Stage 2 Setup: Converts the merged HuggingFace model to NeMo format.
# Requires NVIDIA NeMo to be installed on the cloud instance.
#
# Install NeMo first if not present:
#   pip install nemo_toolkit[all]
#   (or follow https://github.com/NVIDIA/NeMo for Docker-based setup)
#
# Run from your project root:
#   bash scripts/05_convert_to_nemo.sh

set -e

HF_MODEL_PATH="./LLaMA-Factory/outputs/stage1_merged"  # from Stage 1
NEMO_OUTPUT_PATH="./outputs/qwen_omni_thinker_tuned.nemo"

CONVERTER_SCRIPT="/opt/NeMo/scripts/checkpoint_converters/convert_qwen_hf_to_nemo.py"

echo "=== Converting HuggingFace Model to NeMo Format ==="
echo "Input (HF):    $HF_MODEL_PATH"
echo "Output (.nemo): $NEMO_OUTPUT_PATH"

# Check if merged model exists
if [ ! -d "$HF_MODEL_PATH" ]; then
    echo "ERROR: Merged HF model not found at $HF_MODEL_PATH"
    echo "       Complete Stage 1 (training + merge) first."
    exit 1
fi

# Check if NeMo converter exists
if [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo "ERROR: NeMo converter not found at $CONVERTER_SCRIPT"
    echo "       Adjust CONVERTER_SCRIPT to match your NeMo installation path."
    echo "       Common paths:"
    echo "         /opt/NeMo/scripts/checkpoint_converters/convert_qwen_hf_to_nemo.py"
    echo "         $(python -c 'import nemo; import os; print(os.path.dirname(nemo.__file__))')/scripts/..."
    exit 1
fi

mkdir -p "$(dirname "$NEMO_OUTPUT_PATH")"

python "$CONVERTER_SCRIPT" \
    --in-file "$HF_MODEL_PATH" \
    --out-file "$NEMO_OUTPUT_PATH" \
    --precision bf16

echo ""
echo "=== Conversion complete. ==="
echo "NeMo model saved to: $NEMO_OUTPUT_PATH"
echo "Update configs/talker_finetune.yaml → model.nemo_path to this path."
echo "Next: run 06_prepare_stage2_manifest.py then 07_run_stage2_training.sh"

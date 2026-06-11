#!/bin/bash
# Stage 2 Setup: Convert the merged HuggingFace model checkpoint into a NeMo .nemo file.
#
# Steps:
# 1. Install NeMo if not already installed:
#      pip install nemo-toolkit[asr,tts]
#    If you use a custom environment, ensure the installed NeMo package is the one active
#    in the same Python interpreter used by this script.
#
# 2. Confirm your Stage 1 merged HF model exists at:
#      ./LLaMA-Factory/outputs/stage1_merged
#    This is the input model directory for conversion.
#
# 3. Locate the NeMo converter script from the installed NeMo package.
#    The current NeMo release may not place it at /opt/NeMo/scripts/...,
#    so this script searches the installed package directory for a matching converter file.
#
# 4. Run the converter with the input HF checkpoint and output .nemo file.
#    The default output path is:
#      ./outputs/qwen_omni_thinker_tuned.nemo
#
# 5. After conversion, update your Stage 2 config (e.g. configs/talker_finetune.yaml):
#      model.nemo_path: ./outputs/qwen_omni_thinker_tuned.nemo
#
# 6. Continue with Stage 2:
#      python scripts/06_prepare_stage2_manifest.py
#      bash scripts/07_run_stage2_training.sh

set -e

HF_MODEL_PATH="./LLaMA-Factory/outputs/stage1_merged"
NEMO_OUTPUT_PATH="./outputs/qwen_omni_thinker_tuned.nemo"

echo "=== Fixed HF → NeMo conversion helper ==="
echo "Input HF checkpoint: $HF_MODEL_PATH"
echo "Output NeMo file:    $NEMO_OUTPUT_PATH"

test -d "$HF_MODEL_PATH" || {
    echo "ERROR: merged HF model directory not found at $HF_MODEL_PATH"
    echo "       Run Stage 1 training and merge first."
    exit 1
}

CONVERTER_SCRIPT="$(python - <<'PY'
import os
import glob
import sys
try:
    import nemo
except Exception:
    sys.exit(1)
root = os.path.dirname(nemo.__file__)
pattern1 = os.path.join(root, '**', 'convert*qwen*to*nemo*.py')
pattern2 = os.path.join(root, '**', '*convert*to*nemo*.py')
paths = glob.glob(pattern1, recursive=True)
if not paths:
    paths = glob.glob(pattern2, recursive=True)
if paths:
    print(paths[0])
PY
)"

if [ -z "$CONVERTER_SCRIPT" ] || [ ! -f "$CONVERTER_SCRIPT" ]; then
    echo "ERROR: Could not locate a NeMo converter script in the current Python environment."
    echo "       Make sure NeMo is installed and available to the same Python interpreter used here."
    echo "       Example install command: pip install nemo-toolkit[asr,tts]"
    echo "       If NeMo is installed in a custom path, set CONVERTER_SCRIPT to the correct file."
    echo "       Search example:"
    echo "         python -c \"import nemo, os, glob; root=os.path.dirname(nemo.__file__); print(glob.glob(root+'/**/convert*to*nemo*.py', recursive=True))\""
    exit 1
fi

mkdir -p "$(dirname "$NEMO_OUTPUT_PATH")"

echo "Using converter script: $CONVERTER_SCRIPT"

time python "$CONVERTER_SCRIPT" \
    --in-file "$HF_MODEL_PATH" \
    --out-file "$NEMO_OUTPUT_PATH" \
    --precision bf16

echo ""
echo "=== Conversion complete ==="
echo "Saved NeMo model to: $NEMO_OUTPUT_PATH"
echo "Update configs/talker_finetune.yaml → model.nemo_path to this path."
echo "Next: run python scripts/06_prepare_stage2_manifest.py then bash scripts/07_run_stage2_training.sh"

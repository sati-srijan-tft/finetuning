#!/bin/bash
# Stage 2 Training: Fine-tunes the Qwen 3 Omni Talker using NVIDIA NeMo.
# Freezes the Thinker (LLM + audio encoder) and trains only the audio decoder.
#
# Prerequisites:
#   - NeMo installed: pip install nemo_toolkit[all]
#   - .nemo checkpoint ready (from 05_convert_to_nemo.sh)
#   - Manifests ready (from 06_prepare_stage2_manifest.py)
#   - configs/talker_finetune.yaml updated with correct paths
#
# Run from your project root:
#   bash scripts/07_run_stage2_training.sh

set -e

# --- Flag parsing ---
# --small_gpu          : use talker_finetune_small_gpu.yaml (FP16, tensor-parallel)
# --quantized_thinker  : use talker_finetune_quantized_thinker.yaml
#                        (NeMo-native ModelOpt quantization on the frozen Thinker;
#                         requires: pip install nvidia-modelopt[torch])
# Flags are mutually exclusive — last one wins if both are passed.
SMALL_GPU=false
QUANTIZED_THINKER=false
for arg in "$@"; do
    [[ "$arg" == "--small_gpu" ]]         && SMALL_GPU=true
    [[ "$arg" == "--quantized_thinker" ]] && QUANTIZED_THINKER=true
done

CONFIG_DIR="./configs"
if [ "$QUANTIZED_THINKER" = true ]; then
    CONFIG_NAME="talker_finetune_quantized_thinker.yaml"
    echo ">>> Quantized Thinker mode: using $CONFIG_NAME"
    echo "    Ensure nvidia-modelopt is installed: pip install nvidia-modelopt[torch]"
elif [ "$SMALL_GPU" = true ]; then
    CONFIG_NAME="talker_finetune_small_gpu.yaml"
    echo ">>> Small GPU mode: using $CONFIG_NAME"
else
    CONFIG_NAME="talker_finetune.yaml"
fi
NEMO_EXAMPLES="/opt/NeMo/examples/multimodal"

echo "=== Stage 2: Talker Fine-Tuning (Voice Synthesis) ==="
echo "Config: $CONFIG_DIR/$CONFIG_NAME"

# Validate config exists
if [ ! -f "$CONFIG_DIR/$CONFIG_NAME" ]; then
    echo "ERROR: Config not found at $CONFIG_DIR/$CONFIG_NAME"
    exit 1
fi

# Validate NeMo training script exists
TRAIN_SCRIPT="$NEMO_EXAMPLES/train.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: NeMo train.py not found at $TRAIN_SCRIPT"
    echo "       Check your NeMo installation. Common alternatives:"
    echo "         $(python -c 'import nemo; import os; print(os.path.dirname(os.path.dirname(nemo.__file__)))')/examples/multimodal/train.py"
    exit 1
fi

# Extract .nemo path from config and validate
NEMO_PATH=$(python3 -c "
import yaml
with open('$CONFIG_DIR/$CONFIG_NAME') as f:
    cfg = yaml.safe_load(f)
print(cfg['model']['nemo_path'])
")

if [ ! -f "$NEMO_PATH" ]; then
    echo "ERROR: .nemo model not found at: $NEMO_PATH"
    echo "       Run 05_convert_to_nemo.sh and update the config."
    exit 1
fi

echo "NeMo model: $NEMO_PATH"

# Multi-GPU support via torchrun (auto-detects GPU count)
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $N_GPU"

if [ "$N_GPU" -gt 1 ]; then
    echo "Launching distributed training on $N_GPU GPUs ..."
    # torchrun replaces the deprecated torch.distributed.launch (removed in PyTorch 2.x)
    torchrun \
        --nproc_per_node="$N_GPU" \
        --master_port=29500 \
        "$TRAIN_SCRIPT" \
        --config-path="$(realpath $CONFIG_DIR)" \
        --config-name="$CONFIG_NAME" \
        trainer.devices="$N_GPU" \
        trainer.strategy=ddp
else
    python "$TRAIN_SCRIPT" \
        --config-path="$(realpath $CONFIG_DIR)" \
        --config-name="$CONFIG_NAME"
fi

echo ""
echo "=== Stage 2 training complete. ==="
echo "Check outputs/stage2_talker/ for checkpoints and the final .nemo model."

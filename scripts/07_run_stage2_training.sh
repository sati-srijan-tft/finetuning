#!/bin/bash
# Stage 2 Training: Fine-tunes the Qwen 3 Omni Talker (audio decoder).
# Freezes the Thinker and trains only Talker modules via LoRA.
#
# IMPORTANT: NeMo 2.0 has no HF→NeMo conversion scripts for Qwen models.
# The HuggingFace path (--hf / --hf_h100) is the recommended route.
# The legacy NeMo flags are kept for reference but are effectively unusable
# with Qwen until upstream NeMo adds conversion support.
#
# --- Flags ---
#   --hf           HF path, 24 GB GPU  (configs/talker_finetune_bnb.yaml)
#   --hf_h100      HF path, H100 80 GB (configs/260611_Stage2-Talker-H100_ENG.yaml)
#   --config <f>   Explicit config filename inside configs/; auto-detects HF vs NeMo
#   --small_gpu    [NeMo] talker_finetune_small_gpu.yaml
#   --quantized_thinker  [NeMo] talker_finetune_quantized_thinker.yaml
#
# Run from your project root:
#   bash scripts/07_run_stage2_training.sh --hf_h100

set -e

# --- Flag parsing ---
HF=false
HF_H100=false
SMALL_GPU=false
QUANTIZED_THINKER=false
EXPLICIT_CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hf)               HF=true ;;
        --hf_h100)          HF_H100=true ;;
        --small_gpu)        SMALL_GPU=true ;;
        --quantized_thinker) QUANTIZED_THINKER=true ;;
        --config)           EXPLICIT_CONFIG="$2"; shift ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
    shift
done

CONFIG_DIR="./configs"

# --- Resolve config name and training backend ---
if [ -n "$EXPLICIT_CONFIG" ]; then
    CONFIG_NAME="$EXPLICIT_CONFIG"
    # Auto-detect backend: HF configs have a top-level 'model_path' key
    if python3 -c "
import yaml, sys
with open('$CONFIG_DIR/$CONFIG_NAME') as f:
    c = yaml.safe_load(f)
sys.exit(0 if 'model_path' in c else 1)
" 2>/dev/null; then
        USE_HF=true
    else
        USE_HF=false
    fi
elif [ "$HF_H100" = true ]; then
    CONFIG_NAME="260611_Stage2-Talker-H100_ENG.yaml"
    USE_HF=true
    echo ">>> H100 HF mode: using $CONFIG_NAME"
elif [ "$HF" = true ]; then
    CONFIG_NAME="talker_finetune_bnb.yaml"
    USE_HF=true
    echo ">>> HF BnB mode: using $CONFIG_NAME"
elif [ "$QUANTIZED_THINKER" = true ]; then
    CONFIG_NAME="talker_finetune_quantized_thinker.yaml"
    USE_HF=false
    echo ">>> [NeMo] Quantized Thinker mode: using $CONFIG_NAME"
    echo "    Requires: pip install nvidia-modelopt[torch]"
elif [ "$SMALL_GPU" = true ]; then
    CONFIG_NAME="talker_finetune_small_gpu.yaml"
    USE_HF=false
    echo ">>> [NeMo] Small GPU mode: using $CONFIG_NAME"
else
    CONFIG_NAME="talker_finetune.yaml"
    USE_HF=false
    echo ">>> [NeMo] Baseline mode: using $CONFIG_NAME"
    echo "    NOTE: NeMo 2.0 lacks Qwen conversion scripts. Use --hf or --hf_h100 instead."
fi

echo "=== Stage 2: Talker Fine-Tuning (Voice Synthesis) ==="
echo "Config  : $CONFIG_DIR/$CONFIG_NAME"
echo "Backend : $([ "$USE_HF" = true ] && echo HuggingFace || echo NeMo)"

# Validate config exists
if [ ! -f "$CONFIG_DIR/$CONFIG_NAME" ]; then
    echo "ERROR: Config not found at $CONFIG_DIR/$CONFIG_NAME"
    exit 1
fi

# ---------------------------------------------------------------------------
# HuggingFace path
# ---------------------------------------------------------------------------
if [ "$USE_HF" = true ]; then
    HF_SCRIPT="scripts/10_run_stage2_hf.py"
    if [ ! -f "$HF_SCRIPT" ]; then
        echo "ERROR: HF trainer not found at $HF_SCRIPT"
        exit 1
    fi

    N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "GPUs detected: $N_GPU"

    if [ "$N_GPU" -gt 1 ]; then
        echo "Launching distributed HF training on $N_GPU GPUs ..."
        torchrun \
            --nproc_per_node="$N_GPU" \
            --master_port=29500 \
            "$HF_SCRIPT" \
            --config "$CONFIG_DIR/$CONFIG_NAME"
    else
        python "$HF_SCRIPT" --config "$CONFIG_DIR/$CONFIG_NAME"
    fi

    echo ""
    echo "=== Stage 2 HF training complete. ==="
    CONFIG_OUTPUT=$(python3 -c "
import yaml
with open('$CONFIG_DIR/$CONFIG_NAME') as f:
    c = yaml.safe_load(f)
print(c.get('training', {}).get('output_dir', 'outputs/stage2_talker'))
")
    echo "LoRA adapter saved to: $CONFIG_OUTPUT"
    exit 0
fi

# ---------------------------------------------------------------------------
# NeMo path (legacy — requires a working .nemo checkpoint for Qwen)
# ---------------------------------------------------------------------------
NEMO_EXAMPLES="/opt/NeMo/examples/multimodal"
TRAIN_SCRIPT="$NEMO_EXAMPLES/train.py"

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: NeMo train.py not found at $TRAIN_SCRIPT"
    echo "       Check your NeMo installation or use --hf / --hf_h100 instead."
    exit 1
fi

NEMO_PATH=$(python3 -c "
import yaml
with open('$CONFIG_DIR/$CONFIG_NAME') as f:
    cfg = yaml.safe_load(f)
print(cfg['model']['nemo_path'])
")

if [ ! -f "$NEMO_PATH" ]; then
    echo "ERROR: .nemo model not found at: $NEMO_PATH"
    echo "       Run 05_convert_to_nemo.sh and update the config,"
    echo "       or switch to the HF path with --hf or --hf_h100."
    exit 1
fi

echo "NeMo model: $NEMO_PATH"

N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs detected: $N_GPU"

if [ "$N_GPU" -gt 1 ]; then
    echo "Launching distributed NeMo training on $N_GPU GPUs ..."
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
echo "=== Stage 2 NeMo training complete. ==="
echo "Check outputs/stage2_talker/ for checkpoints."

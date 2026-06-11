# Qwen 3 Omni 30B — Hindi/Indic Fine-Tuning Run Guide

## Directory Structure

```
finetuning/
├── data/
│   ├── dataset_info.json       # LLaMA-Factory dataset registration
│   ├── sample_data.jsonl       # Starter Hindi training samples
│   └── audio_data/             # (Optional) Audio clips for ASR training
│       └── hindi/
│           ├── clip_001.wav
│           └── clip_001.txt    # Transcript for each .wav
├── configs/
│   ├── stage1_lora_config.yaml              # Stage 1: LoRA training config
│   ├── talker_finetune.yaml                 # Stage 2: NeMo TTS config (A100 80GB baseline)
│   ├── talker_finetune_small_gpu.yaml       # Stage 2: 2× 24GB or A100 40GB variant
│   ├── talker_finetune_bnb.yaml             # Stage 2: BitsAndBytes HF path (no NeMo)
│   └── talker_finetune_quantized_thinker.yaml # Stage 2: NeMo + quantized frozen Thinker
├── scripts/
│   ├── 00_prepare_indicTTS_hindi.py # Download SPRINGLab/IndicTTS-Hindi + build NeMo manifests
│   ├── 01_setup_stage1.sh           # Install LLaMA-Factory on GPU instance
│   ├── 02_prepare_stage1_data.py    # Validate & copy data to LLaMA-Factory
│   ├── 03_run_stage1_training.sh    # Run Stage 1 LoRA training
│   ├── 04_merge_lora_adapters.sh    # Merge LoRA into base model
│   ├── fix_data_jsonl.py            # One-shot fix: normalize array content → string format
│   ├── 05_convert_to_nemo.sh        # Convert HF → .nemo for Stage 2
│   ├── 06_prepare_stage2_manifest.py# Resample audio + build NeMo manifests (generic)
│   ├── 07_run_stage2_training.sh    # Run Stage 2 NeMo TTS training
│   └── 08_test_inference.py         # Verify Stage 1 model responds in Hindi
└── outputs/                    # Created automatically during training
    ├── stage1_lora/            # LoRA adapter checkpoints
    ├── stage1_merged/          # Merged HF model
    └── stage2_talker/          # Final NeMo TTS checkpoints
```

---

## Stage 1: Fine-Tune the Thinker (Text + Comprehension)

### Step 1 — Add your training data

**Text instruction data** (optional but recommended to mix with ASR):
Place `.jsonl` files in `data/`. Each line must follow the ChatML/ShareGPT format:
```json
{"messages": [
  {"role": "system",    "content": "आप एक सहायक AI हैं।"},
  {"role": "user",      "content": "आपका प्रश्न..."},
  {"role": "assistant", "content": "उत्तर..."}
]}
```

**Audio/ASR samples** use a string `content` with an `<audio>` tag plus a top-level `audios` list.
All `content` fields must be strings — never arrays — to avoid PyArrow schema errors:
```json
{"messages": [
  {"role": "system",    "content": "आप हिंदी ऑडियो को text में बदलने में सक्षम हैं।"},
  {"role": "user",      "content": "<audio>इस ऑडियो को हिंदी में लिखें।"},
  {"role": "assistant", "content": "ट्रांसक्रिप्ट यहाँ..."}
], "audios": ["audio_data/indicvoices/filename.wav"]}
```
Audio paths in `audios` are relative to `LLaMA-Factory/data/`.

`data/sample_data.jsonl` contains starter Hindi examples (text + one audio sample) you can build on.

> **If you have an existing `data.jsonl` with old array-format content**, fix it before training:
> ```bash
> python scripts/fix_data_jsonl.py LLaMA-Factory/data/data.jsonl
> ```

Recommended text data sources:
- **IndicInstruct** (AI4Bharat) — Hindi instruction pairs
- **Samanantar** — parallel Hindi text
- **Sangraha** — cleaned Indic web text

**ASR data — IndicVoices** is handled automatically by the prep script. You do not need to download it manually; the script streams it from HuggingFace.

> **HuggingFace access:** IndicVoices requires accepting the dataset terms at  
> `https://huggingface.co/datasets/ai4bharat/IndicVoices`  
> Then authenticate on the cloud instance: `huggingface-cli login`

### Step 2 — Upload to your cloud GPU instance

```bash
# Example: rsync to an A100 instance
rsync -avz ./finetuning/ user@gpu-instance:/workspace/finetuning/
```

### Step 3 — On the GPU instance: setup

```bash
cd /workspace/finetuning
bash scripts/01_setup_stage1.sh
pip install datasets soundfile librosa   # extra deps for IndicVoices

pip install torchcodec --index-url https://download.pytorch.org/whl/cu128  #the newer version uses this instead of soundfile and librosa (for cuda 12.8)
```

### Step 4 — Prepare data (with IndicVoices ASR)

```bash
source .venv/bin/activate   # or: conda activate llama_factory
cd /workspace/finetuning

# Mix text instruction data + 10,000 IndicVoices ASR samples
python scripts/02_prepare_stage1_data.py \
    --input_dir data/ \
    --output_dir LLaMA-Factory/data/ \
    --use_indicvoices \
    --max_asr_samples 10000 \
    --split_eval

# Scale up for more ASR coverage (first run streams ~26GB from HF cache):
# --max_asr_samples 50000
```

This saves audio files to `LLaMA-Factory/data/audio_data/indicvoices/` and creates
`data.jsonl` with interleaved text instruction + ASR entries.
It also writes `LLaMA-Factory/data/dataset_info.json` with the required column and tag mappings.
If you copy data manually, ensure `dataset_info.json` contains both of these — omitting either causes silent data loss:
- `"audios": "audios"` in the `columns` block — without it, `<audio>` tags are treated as literal text
- `"system_tag": "system"` in the `tags` block — without it, every sample with a system message is rejected with "Invalid role tag"

### Step 5 — Start training

```bash
cd LLaMA-Factory
bash ../scripts/03_run_stage1_training.sh

# Multi-GPU (e.g. 2x RTX 4090):
bash ../scripts/03_run_stage1_training.sh --multi_gpu
```

Monitor training loss at `outputs/stage1_lora/trainer_log.jsonl`.

### Step 6 — Merge LoRA adapters

```bash
cd LLaMA-Factory
bash ../scripts/04_merge_lora_adapters.sh
```

### Step 7 — Test the model

```bash
python scripts/08_test_inference.py \
    --model_path LLaMA-Factory/outputs/stage1_merged \
    --load_in_4bit \
    --prompt "भारत के बारे में बताओ।"
```

---

## Stage 2: Fine-Tune the Talker (Voice Synthesis)

### Step 1 — Prepare audio data

**Recommended: SPRINGLab/IndicTTS-Hindi (HuggingFace)**

This single command downloads the dataset, resamples to 24 kHz, saves `.wav` files, and writes NeMo manifests:
```bash

python scripts/00_prepare_indicTTS_hindi.py \
    --output_dir /data/indictts_hindi \
    --manifest_dir ./manifests \
    --split_eval \
    --eval_ratio 0.05
```

If the dataset is gated, authenticate first:
```bash
huggingface-cli login
# or pass: --hf_token hf_XXXXXXXXXXXX
```

**Alternative: bring your own `.wav` + `.txt` pairs**

Layout your audio + transcripts like this:
```
/data/indic_tts/
    hindi/
        clip_001.wav   +   clip_001.txt
        clip_002.wav   +   clip_002.txt
```

Then generate the NeMo manifests with the generic script:
```bash
python scripts/06_prepare_stage2_manifest.py \
    --audio_dir /data/indic_tts \
    --output_dir ./manifests \
    --target_sr 24000 \
    --split_eval
```

### Step 2 — Convert Stage 1 model to NeMo format

```bash
bash scripts/05_convert_to_nemo.sh
```

### Step 3 — Choose a Stage 2 config and update it

Pick the config that matches your hardware, then fill in the three placeholder paths:

| Config file | Target hardware | Thinker memory | Notes |
|---|---|---|---|
| `talker_finetune.yaml` | 1× A100 80 GB | ~60 GB BF16 | Baseline; highest quality |
| `talker_finetune_small_gpu.yaml` | 2× 24 GB or 1× A100 40 GB | ~30 GB/card (tensor-parallel) | See DeepSpeed CPU-offload comment for single 24 GB |
| `talker_finetune_bnb.yaml` | 1× 24 GB | ~15 GB (4-bit NF4) | HuggingFace path — no NeMo conversion needed |
| `talker_finetune_quantized_thinker.yaml` | 1× A100 80 GB | ~30 GB INT8 or ~15 GB W4A16 | NeMo-native quant; requires `nvidia-modelopt` |

In your chosen config, set:
- `model.nemo_path` (or `model_path` for the BnB config) → path to your checkpoint
- `model.data.train_ds.manifest_filepath` → `./manifests/train_manifest.json`
- `model.data.validation_ds.manifest_filepath` → `./manifests/val_manifest.json`
- `exp_manager.exp_dir` → where you want checkpoints saved

#### Small-GPU config extra steps

```bash
# 2× GPU tensor-parallel run (already the default in talker_finetune_small_gpu.yaml)
bash scripts/07_run_stage2_training.sh --config talker_finetune_small_gpu.yaml

# Single 24 GB card: uncomment the DeepSpeed block in the config, then:
bash scripts/07_run_stage2_training.sh --config talker_finetune_small_gpu.yaml
```

#### BitsAndBytes (HF) path — no NeMo needed

Skip steps 1–2 above (no NeMo conversion required). Use the Stage 1 merged model directly:

```bash
# Install extra dep
pip install bitsandbytes

# Edit configs/talker_finetune_bnb.yaml:
#   model_path → ./LLaMA-Factory/outputs/stage1_merged
#   data.train_manifest / data.val_manifest → your manifest paths

bash scripts/07_run_stage2_training.sh --config talker_finetune_bnb.yaml
```

Output lands in `outputs/stage2_talker_bnb/`.

#### Quantized Thinker (NeMo-native)

Requires NeMo conversion (step 2). Extra dependency:

```bash
pip install nvidia-modelopt[torch]
```

Edit `configs/talker_finetune_quantized_thinker.yaml`:
- `quantization.algorithm`: `int8_sq` (30 GB, best quality) or `w4a16` (15 GB, most savings)
- `quantization.num_calib_steps`: 512 default; raise to 1024 for better INT8 accuracy
- `quantization.exclude_from_quantization`: verify module names match your `.nemo` checkpoint  
  (run `python -c "import nemo; m=<load model>; print([n for n,_ in m.named_modules()])"`)

```bash
bash scripts/07_run_stage2_training.sh --config talker_finetune_quantized_thinker.yaml
```

Output lands in `outputs/stage2_talker_quant/`.

### Step 4 — Run Stage 2 training (baseline / A100 80 GB)

```bash
bash scripts/07_run_stage2_training.sh
```

---

## Key Config Tuning Tips

| Parameter | Location | Guidance |
|---|---|---|
| `lora_rank` | stage1_lora_config.yaml | 8–64; higher = more capacity but more VRAM |
| `per_device_train_batch_size` | stage1_lora_config.yaml | 1–4 on A100 80GB with 4-bit |
| `gradient_accumulation_steps` | stage1_lora_config.yaml | Increase to simulate larger batch |
| `num_train_epochs` | stage1_lora_config.yaml | 3–5 for instruction tuning |
| `learning_rate` | stage1_lora_config.yaml | 1e-4 to 3e-4 for LoRA |
| `target_sr` | 06_prepare_stage2_manifest.py | Must match Qwen Omni native rate (24 kHz) |
| `batch_size` (NeMo configs) | talker_finetune*.yaml | 8 on A100 80GB; 2 on 24GB cards; 1 for validation |
| `accumulate_grad_batches` | talker_finetune_small_gpu.yaml | 16 to keep effective batch = 32; raise if OOM |
| `tensor_model_parallel_size` | talker_finetune_small_gpu.yaml | Must equal `trainer.devices` |
| `quantization.algorithm` | talker_finetune_quantized_thinker.yaml | `int8_sq` (quality) vs `w4a16` (VRAM) |
| `quantization.sq_alpha` | talker_finetune_quantized_thinker.yaml | 0.0–1.0; 0.5 is standard SmoothQuant default |
| `bnb_4bit_compute_dtype` | talker_finetune_bnb.yaml | `bfloat16` on A100; `float16` on consumer GPUs |
| `talker_module_regex` | talker_finetune_bnb.yaml | Adjust prefix if model inspection shows different name |

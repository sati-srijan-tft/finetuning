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
│   ├── stage1_lora_config.yaml                    # Stage 1: LoRA training config (A100)
│   ├── talker_finetune_bnb.yaml                   # Stage 2: HF path, 24 GB GPU (4-bit Thinker)
│   ├── 260611_Stage2-Talker-H100_ENG.yaml         # Stage 2: HF path, H100 80 GB (8-bit Thinker) ← recommended
│   ├── talker_finetune.yaml                       # [NeMo — unusable for Qwen] A100 80GB baseline
│   ├── talker_finetune_small_gpu.yaml             # [NeMo — unusable for Qwen] 2× 24GB variant
│   └── talker_finetune_quantized_thinker.yaml     # [NeMo — unusable for Qwen] ModelOpt quant
├── scripts/
│   ├── 00_prepare_indicTTS_hindi.py  # Download SPRINGLab/IndicTTS-Hindi + build manifests
│   ├── 01_setup_stage1.sh            # Install LLaMA-Factory on GPU instance
│   ├── 02_prepare_stage1_data.py     # Validate & copy data to LLaMA-Factory
│   ├── 03_run_stage1_training.sh     # Run Stage 1 LoRA training
│   ├── 04_merge_lora_adapters.sh     # Merge LoRA into base model
│   ├── fix_data_jsonl.py             # One-shot fix: normalize array content → string format
│   ├── 05_convert_to_nemo.sh         # [NeMo — unusable for Qwen] HF → .nemo conversion
│   ├── 06_prepare_stage2_manifest.py # Resample audio + build JSON manifests
│   ├── 07_run_stage2_training.sh     # Run Stage 2 training (HF or NeMo)
│   ├── 08_test_inference.py          # Verify Stage 1 model responds in Hindi
│   ├── 10_run_stage2_hf.py           # HF Trainer for Stage 2 (called by 07_run_stage2_training.sh)
│   └── 11_merge_stage2_lora.py      # Merge Stage 2 LoRA → final standalone model
└── outputs/                     # Created automatically during training
    ├── stage1_lora/             # LoRA adapter checkpoints
    ├── stage1_merged/           # Merged HF model (base for Stage 2)
    ├── stage2_talker_bnb/       # Stage 2 HF adapter (24 GB path)
    ├── stage2_talker_h100/      # Stage 2 HF adapter (H100 path)
    └── final_model/             # Final merged model — Thinker + Talker, ready for inference
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
python model_merge_into_base_model.py
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

> **NeMo is not an option for Qwen models.** NeMo 2.0 does not include HF → NeMo conversion
> scripts for Qwen Omni. The NeMo configs (`talker_finetune*.yaml`) are kept for reference but
> cannot be used until upstream NeMo adds support. **Use the HuggingFace path below.**

### Step 1 — Prepare audio data

**Recommended: SPRINGLab/IndicTTS-Hindi (HuggingFace)**

This single command downloads the dataset, resamples to 24 kHz, saves `.wav` files, and writes NeMo manifests:
```bash

python scripts/00_prepare_indicTTS_hindi.py \
    --output_dir ./data/indictts_hindi \
    --manifest_dir ./manifests \
    --split_eval \
    --eval_ratio 0.05

# Dry-run / smoke-test with a small subset (100 examples per split):
python scripts/00_prepare_indicTTS_hindi.py \
    --output_dir ./data/indictts_hindi \
    --manifest_dir ./manifests \
    --max_samples 100 \
    --split_eval
    
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

### Step 2 — Install HF dependencies

```bash
pip install transformers peft bitsandbytes accelerate soundfile
# Only needed if your audio is not already at 24 kHz:
pip install librosa
```

### Step 3 — Choose a config and set the paths

| Config file | Target hardware | Thinker quant | Flag |
|---|---|---|---|
| `260611_Stage2-Talker-H100_ENG.yaml` | H100 80 GB | 8-bit (~30 GB) | `--hf_h100` |
| `talker_finetune_bnb.yaml` | Any 24 GB GPU | 4-bit NF4 (~15 GB) | `--hf` |

In your chosen config, update the three placeholder paths:
- `model_path` → `./LLaMA-Factory/outputs/stage1_merged`
- `data.train_manifest` → `./manifests/train_manifest.json`
- `data.val_manifest` → `./manifests/val_manifest.json`

### Step 4 — Run Stage 2 training

```bash
# H100 80 GB (recommended)
bash scripts/07_run_stage2_training.sh --hf_h100

# 24 GB GPU
bash scripts/07_run_stage2_training.sh --hf

# Explicit config override (auto-detects HF vs NeMo from the file)
bash scripts/07_run_stage2_training.sh --config 260611_Stage2-Talker-H100_ENG.yaml
```

Multi-GPU is handled automatically — `torchrun` is used when `nvidia-smi` detects more than one GPU.

Output lands in the `output_dir` set in the config (`outputs/stage2_talker_h100/` or `outputs/stage2_talker_bnb/`).

### Step 5 — Merge Stage 2 LoRA into the final model

Stage 2 training saves only the LoRA adapter. This step bakes it into `stage1_merged` to produce
a single standalone HuggingFace checkpoint ready for inference or deployment.

```bash
# Auto-detects stage2_talker_h100 or stage2_talker_bnb automatically
python scripts/11_merge_stage2_lora.py

# Explicit paths
python scripts/11_merge_stage2_lora.py \
    --base_model  ./LLaMA-Factory/outputs/stage1_merged \
    --adapter     ./outputs/stage2_talker_h100 \
    --output_dir  ./outputs/final_model
```

The merge runs on CPU in bfloat16 — no GPU needed, but requires enough RAM to hold the model
(~60 GB for the 30B model). Output is sharded into 4 GB safetensor files in `outputs/final_model/`.

### Step 6 — Test the final model

```bash
python scripts/08_test_inference.py \
    --model_path ./outputs/final_model \
    --load_in_4bit \
    --prompt "भारत के बारे में बताओ।"
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
| `lora.rank` | Stage 2 HF configs | 8–32; H100 config uses 16, BnB uses 8 |
| `training.per_device_train_batch_size` | Stage 2 HF configs | 8 on H100; 2 on 24 GB — reduce if OOM |
| `training.gradient_accumulation_steps` | Stage 2 HF configs | Adjust to keep effective batch = 32 |
| `quantization.load_in_8bit` | 260611_Stage2-Talker-H100_ENG.yaml | Switch to `load_in_4bit` if OOM on H100 |
| `quantization.bnb_4bit_compute_dtype` | talker_finetune_bnb.yaml | `bfloat16` on A100/H100; `float16` on consumer GPUs |
| `lora.talker_module_regex` | Stage 2 HF configs | If no modules match at startup, the script prints an inspect command; update this regex |

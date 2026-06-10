# Comprehensive Guide: Stage-Wise Fine-Tuning Qwen 3 Omni 30B for Indic Languages

This document outlines the end-to-end, memory-efficient pipeline for fine-tuning the **Qwen 3 Omni 30B** model on Indian languages. To optimize VRAM usage and protect core capabilities, the process is split into two distinct operational stages: teaching the model to understand the language (The Thinker), and teaching the model to speak the language (The Talker).

---

## Architecture Overview

Qwen 3 Omni utilizes a native **"Thinker-Talker" Mixture-of-Experts (MoE)** pipeline. Stage-wise training allows you to isolate parameter groups:

| Stage | Component | Focus | Primary Framework | Data Format |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | The Thinker | Text Reasoning & Audio Comprehension | LLaMA-Factory | ChatML (JSONL) |
| **Stage 2** | The Talker | Voice Synthesis (Text-to-Speech) | NVIDIA NeMo | NeMo Audio Manifest |

---

## Compute & Hardware Prerequisites

*   **Development/Scripting Environment:** Local machine (e.g., Ryzen 5 4600H, 16GB RAM, GTX 1650) for data preprocessing, script preparation, and sanity checks.
*   **Training Environment:** Cloud GPU Instance (e.g., 1x NVIDIA A100 80GB or 2x RTX 4090s) to accommodate the 30B model under **4-bit QLoRA** quantization.

---

## Stage 1: Fine-Tuning the Thinker (Text & Comprehension)

This stage trains the primary LLM backbone to read, reason, and listen to Indian languages while leaving the audio generation layers frozen.

### 1. Data Preparation (ChatML Format)
Create a file named `dataset_info.json` and your core training data in a `data.jsonl` file.

#### Text-Only Instruction Tuning Example
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful and polite AI assistant."
    },
    {
      "role": "user",
      "content": "भारत की राजधानी क्या है और इसके बारे में कुछ रोचक तथ्य बताएं।"
    },
    {
      "role": "assistant",
      "content": "भारत की राजधानी नई दिल्ली है। एक रोचक तथ्य यह है कि नई दिल्ली को ब्रिटिश वास्तुकार एडविन लुटियंस द्वारा डिजाइन किया गया था..."
    }
  ]
}
```

#### Audio-to-Text Comprehension (Speech Recognition) Example
Audio samples must use string `content` with an `<audio>` tag, and list audio files in a top-level `audios` key. All `content` fields across text and audio samples must be strings — never arrays — to avoid PyArrow schema errors when loading with HuggingFace datasets.
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant capable of understanding audio."
    },
    {
      "role": "user",
      "content": "<audio>Please transcribe what is said in this audio."
    },
    {
      "role": "assistant",
      "content": "ऑडियो में वक्ता कह रहा है कि आज मौसम बहुत सुहावना है।"
    }
  ],
  "audios": ["audio_data/hindi_clip_001.wav"]
}
```
The `<audio>` tag marks where the audio is injected in the prompt. Audio paths in `audios` are relative to `LLaMA-Factory/data/`.

### 2. Environment Setup
Run these commands on your cloud GPU instance:
```bash
git clone --depth 1 [https://github.com/hiyouga/LLaMA-Factory.git](https://github.com/hiyouga/LLaMA-Factory.git)
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```

### 3. Registering the Dataset
Move your data files into `LLaMA-Factory/data/` and append your dataset schema to `LLaMA-Factory/data/dataset_info.json`:
```json
"indic_thinker_data": {
  "file_name": "data.jsonl",
  "formatting": "sharegpt",
  "columns": {
    "messages": "messages",
    "audios": "audios"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant",
    "system_tag": "system"
  }
}
```

### 4. Training Execution
Launch training via the CLI using the provided config and scripts (see `RUN_GUIDE.md` for the full step-by-step):
```bash
cd LLaMA-Factory
bash ../scripts/03_run_stage1_training.sh

# Multi-GPU (e.g. 2× RTX 4090):
bash ../scripts/03_run_stage1_training.sh --multi_gpu
```

Key parameters configured in `configs/stage1_lora_config.yaml`:
*   **Model Name:** `Qwen/Qwen3-Omni-30B-A3B-Instruct`
*   **Fine-tuning Method:** `LoRA`
*   **Quantization:** `4-bit` (crucial for VRAM optimization)
*   **Dataset:** `indic_thinker_data`
*   **LoRA Target Modules:** `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj` (do **not** target `all` — avoids training audio encoder weights)

Once training concludes, merge the LoRA adapters into the base model:
```bash
bash ../scripts/04_merge_lora_adapters.sh
```

---

## Stage 2: Fine-Tuning the Talker (Voice & Synthesis)

This stage locks down the text processing layers you trained in Stage 1 and focuses entirely on teaching the audio-decoder modules how to speak with natural Indian accents and intonations.

### 1. Checkpoint Conversion to NeMo format
Because NVIDIA NeMo manages speech generation via Megatron-LM structures, convert your merged Hugging Face model from Stage 1 into a `.nemo` format:
```bash
python /opt/NeMo/scripts/checkpoint_converters/convert_qwen_hf_to_nemo.py \
    --in-file /path/to/your_merged_hf_model \
    --out-file /path/to/qwen_omni_thinker_tuned.nemo
```

### 2. Data Preparation (NeMo Manifest)
Convert your studio-quality voice datasets (e.g., IndicTTS) into a standard text-audio mapping JSONL format. 

**`train_manifest.json`**
```json
{"audio_filepath": "/data/indic_tts/hindi/clip_001.wav", "duration": 4.2, "text": "नमस्ते, मैं आपकी कैसे सहायता कर सकता हूँ?"}
{"audio_filepath": "/data/indic_tts/hindi/clip_002.wav", "duration": 2.8, "text": "लेनदेन को सफलतापूर्वक सत्यापित किया गया है।"}
```
> **Note:** Stage 2 (Talker / TTS) audio must be resampled to **24 kHz** — the native rate of Qwen Omni’s audio decoder. Stage 1 (Thinker / ASR) audio uses **16 kHz** (WhisperFeatureExtractor). Mixing rates between stages is intentional and correct.

### 3. Creating the Training Configuration
Create a configuration file named `talker_finetune.yaml` to enforce component isolation:
```yaml
name: "Qwen_Omni_Talker_Indic"

model:
  nemo_path: "/path/to/qwen_omni_thinker_tuned.nemo"
  
  # Completely freeze the language processing components
  freeze_llm: true
  freeze_audio_encoder: true
  
  # Target the acoustic projection/generation layers
  peft:
    peft_scheme: "lora"
    target_modules: ["audio_decoder", "code_predictor"] 

  data:
    train_ds:
      manifest_filepath: "/path/to/train_manifest.json"
      batch_size: 8
```

### 4. Training Execution
Execute the training routine using NeMo's dedicated multimodal processing wrappers:
```bash
python /opt/NeMo/examples/multimodal/train.py \
    --config-path=/path/to/configs \
    --config-name=talker_finetune.yaml
```

Once Stage 2 finishes, the output weights are combined with your Stage 1 framework to output a completed, fully interactive voice and text assistant tailored to Indic processing tasks.
```
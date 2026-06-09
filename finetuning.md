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
```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful AI assistant capable of understanding audio."
    },
    {
      "role": "user",
      "content": [
        {"audio": "audio_data/hindi_clip_001.wav"},
        {"text": "Please transcribe what is said in this audio."}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"text": "ऑडियो में वक्ता कह रहा है कि आज मौसम बहुत सुहावना है।"}
      ]
    }
  ]
}
```

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
    "messages": "messages"
  },
  "tags": {
    "role_tag": "role",
    "content_tag": "content",
    "user_tag": "user",
    "assistant_tag": "assistant"
  }
}
```

### 4. Training Execution
Launch the web interface to configure your training parameters:
```bash
llamafactory-cli webui
```

#### Essential WebUI Configurations
*   **Model Name:** `Qwen/Qwen3-Omni-30B`
*   **Fine-tuning Method:** `LoRA`
*   **Quantization:** `4-bit` (Crucial for VRAM optimization)
*   **Dataset:** `indic_thinker_data`
*   **LoRA Target Modules:** Manually target the core LLM projection layers: `q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj` (Do **not** select `all`).

Once training concludes, navigate to the **Export** tab in the UI to permanently merge your newly trained LoRA adapters back into the base Qwen 3 Omni model weights.

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
> **Note:** Ensure all target audio is cleanly resampled to the model’s native expected frequency (typically 16kHz or 24kHz) before generating the manifest file.

### 3. Creating the Training Configuration
Create a configuration file named `talker_finetune.yaml` to enforce component isolation:
```yaml
name: "Qwen_Omni_Talker_Indic"

model:
  nemo_path: "/path/to/qwen_omni_thinker_tuned.nemo"
  
  # Completely freeze the language processing components
  freeze_llm: True
  freeze_audio_encoder: True 
  
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
#!/usr/bin/env python3
"""
Stage 2 HuggingFace Trainer for Qwen Omni Talker fine-tuning.

Loads the Stage 1 merged model, freezes the Thinker (quantized via BitsAndBytes),
applies LoRA to Talker modules via PEFT, and fine-tunes on TTS manifests.

Usage:
    python scripts/10_run_stage2_hf.py --config configs/260611_Stage2-Talker-H100_ENG.yaml
    python scripts/10_run_stage2_hf.py --config configs/talker_finetune_bnb.yaml

Prerequisites:
    pip install transformers peft bitsandbytes accelerate soundfile
    # If audio in manifests is not pre-resampled to 24 kHz:
    pip install librosa
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List
import soundfile as sf
import librosa

import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithPast
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    Qwen3OmniMoeProcessor,
)

# Qwen3OmniMoeThinker is not registered with AutoModelForCausalLM in transformers;
# import it directly from the submodule.
try:
    from transformers import Qwen3OmniMoeForConditionalGeneration as _ModelCls
    
except ImportError:
    # Fallback: older or patched transformers may expose it via AutoModel
    from transformers import AutoModel as _ModelCls


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _patched_talker_forward(
    self, 
    input_ids=None, 
    attention_mask=None, 
    audio_codes=None, # The RVQ target codes from your updated collator
    **kwargs
):
    # 1. Run the Thinker (Text LLM)
    # Strip text labels — they belong to the text LM loss, not the Talker codec loss.
    # Passing them through causes the frozen thinker to compute unnecessary text loss.
    kwargs.pop("labels", None)
    thinker_outputs = self.thinker(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        return_dict=True,
        **kwargs
    )
    
    # Extract the final hidden state from the Thinker
    thinker_hidden_states = thinker_outputs.hidden_states[-1]
    
    # 2. Run the Talker
    # The safetensors confirm 'hidden_projection' is inside 'talker'.
    # We pass the thinker's hidden states directly into the talker.
    talker_outputs = self.talker(
        hidden_states=thinker_hidden_states,
        audio_codes=audio_codes 
    )
    
    # Extract logits. 
    # Qwen-Omni returns logits for all 15 codebooks. 
    # Shape is typically [batch_size, seq_len, 15, vocab_size] OR a tuple of 15 tensors.
    logits = talker_outputs.logits if hasattr(talker_outputs, "logits") else talker_outputs
    
    # 3. Compute Audio Codec Loss across all 15 heads
    loss = None
    if audio_codes is not None:
        loss_fct = CrossEntropyLoss()
        loss = torch.zeros(1, device=audio_codes.device, dtype=torch.float32)
        num_codebooks = 15  # Confirmed by talker.code_predictor.lm_head.0 through .14
        
        # We iterate through the 15 codebook heads
        for i in range(num_codebooks):
            # Check if logits are returned as a stacked tensor or a tuple/list
            if isinstance(logits, (tuple, list)):
                cb_logits = logits[i][:, :-1, :].contiguous()
            else:
                # Shape: [batch, seq_len, num_codebooks, vocab_size]
                cb_logits = logits[:, :-1, i, :].contiguous()
                
            # Targets shape: [batch, num_codebooks, seq_len]
            cb_labels = audio_codes[:, i, 1:].contiguous()
            
            # Calculate Cross-Entropy loss for this specific codebook
            cb_loss = loss_fct(
                cb_logits.view(-1, cb_logits.size(-1)), 
                cb_labels.view(-1)
            )
            loss += cb_loss
            
        # Average the loss across all 15 codebook predictors
        loss = loss / num_codebooks

    # 4. Return standard HF output so the Trainer can run backward() and log progress
    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits
    )

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_bnb_config(quant: dict) -> BitsAndBytesConfig:
    if quant.get("load_in_8bit"):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=quant.get("llm_int8_threshold", 6.0),
            llm_int8_has_fp16_weight=quant.get("llm_int8_has_fp16_weight", False),
        )
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
    )


# ---------------------------------------------------------------------------
# LoRA target discovery
# ---------------------------------------------------------------------------

def find_talker_targets(model, regex: str) -> List[str]:
    """
    Match full dotted module names against the regex, then return the unique
    bare names (last component) that PEFT's target_modules expects.

    If nothing matches, print a helper command so you can inspect the real names.
    """
    matched_full = [name for name, _ in model.named_modules() if re.fullmatch(regex, name)]
    if not matched_full:
        print(
            f"\nWARNING: talker_module_regex '{regex}' matched no modules.\n"
            "Inspect the model to find the right prefix:\n\n"
            f"  python -c \"\n"
            f"  from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeThinkerForConditionalGeneration\n"
            f"  m = Qwen3OmniMoeThinkerForConditionalGeneration.from_pretrained('{model.config._name_or_path}')\n"
            f"  print([n for n,_ in m.named_modules()])\n"
            f"  \"\n\n"
            "Then update talker_module_regex in your config and re-run.\n"
        )
        return []
    unique_bare = list({n.split(".")[-1] for n in matched_full})
    print(f"  LoRA targets: {len(matched_full)} modules → bare names: {unique_bare}")
    return unique_bare


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_manifest(path: str, min_dur: float, max_dur: float) -> List[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            dur = entry.get("duration", max_dur)
            if min_dur <= dur <= max_dur:
                entries.append(entry)
    return entries


class TalkerDataCollator:
    """
    Loads audio from manifest entries and feeds them to the model processor.

    Manifest format (produced by 06_prepare_stage2_manifest.py):
        {"audio_filepath": "/path/clip.wav", "text": "transcript", "duration": 2.5}

    The processor encodes (text, audio) pairs into input_ids that include both
    text tokens and Talker audio-codec tokens; labels mirror input_ids with
    padding positions masked to -100 for the causal LM loss.

    NOTE: Qwen2.5-Omni's processor API may differ from the generic call below.
    If you get a TypeError, check:
        from transformers import AutoProcessor
        p = AutoProcessor.from_pretrained("<model_path>", trust_remote_code=True)
        help(p)
    and adjust the keyword arguments accordingly.
    """

    def __init__(self, processor, sample_rate: int):
        self.processor = processor
        self.sample_rate = sample_rate

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        import soundfile as sf

        texts, audios = [], []
        for entry in batch:
            texts.append(entry.get("text", ""))
            waveform, sr = sf.read(entry["audio_filepath"], dtype="float32")
            if sr != self.sample_rate:
                try:
                    import librosa
                    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                except ImportError:
                    raise RuntimeError(
                        f"Audio at {entry['audio_filepath']} has sr={sr}, expected {self.sample_rate}. "
                        "Install librosa to auto-resample: pip install librosa"
                    )
            audios.append(waveform)

        encoded = self.processor(
            text=texts,
            audios=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        labels = encoded["input_ids"].clone()
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        return encoded

class TalkerDataCollatorPatched:
    def __init__(self, processor, model, sample_rate=24000):
        self.processor = processor
        self.model = model  # Passed in so we can access the audio tokenizer/encoder
        self.sample_rate = sample_rate

    def __call__(self, batch):
        texts = []
        audios = []
        
        # 1. Load and resample audio files exactly like your original code
        for entry in batch:
            texts.append(entry.get("text", ""))
            
            # Load audio file
            waveform, sr = sf.read(entry["audio_filepath"], dtype="float32")
            
            # Resample if sample rate doesn't match
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
                
            audios.append(waveform)

        # 2. Extract input features (spectrograms/prompts) via the processor
        encoded = self.processor(
            text=texts,
            audios=audios,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 3. Extract the discrete target RVQ codes from the raw audios.
        # NOTE: this must run in the main process (dataloader_num_workers=0) because
        # self.model lives on the GPU and is not accessible from DataLoader subprocesses.
        with torch.no_grad():
            audio_codes = None

            if hasattr(self.model, "encode_audio"):
                audio_codes = self.model.encode_audio(audios, sampling_rate=self.sample_rate)

            elif (hasattr(self.model, "talker") and
                  hasattr(self.model.talker, "codec_model")):
                # Qwen3-Omni: the CosyVoice codec lives inside model.talker.codec_model
                codec = self.model.talker.codec_model
                codec_device = next(codec.parameters()).device
                all_codes = []
                for waveform in audios:
                    # codec expects [batch=1, channels=1, time]
                    wav_t = torch.FloatTensor(waveform).unsqueeze(0).unsqueeze(0).to(codec_device)
                    result = codec.encode(wav_t)
                    # result may be (codes, scale) or just codes; codes shape [1, n_cb, T']
                    codes = result[0] if isinstance(result, (list, tuple)) else result
                    all_codes.append(codes.squeeze(0))  # [n_cb, T']
                max_t = max(c.shape[-1] for c in all_codes)
                n_cb = all_codes[0].shape[0]
                padded = torch.zeros(len(all_codes), n_cb, max_t, dtype=torch.long, device=codec_device)
                for i, c in enumerate(all_codes):
                    padded[i, :, :c.shape[-1]] = c
                audio_codes = padded

            elif hasattr(self.model, "audio_encoder"):
                audio_features = self.processor.feature_extractor(audios, sampling_rate=self.sample_rate, return_tensors="pt")
                audio_features = {k: v.to(self.model.device) for k, v in audio_features.items()}
                audio_codes = self.model.audio_encoder.encode(**audio_features)

            else:
                audio_codes = encoded.get("audio_codes", None)
                if audio_codes is None:
                    codec_related = [
                        n for n, _ in self.model.named_modules()
                        if any(k in n.lower() for k in ("codec", "audio", "quantiz", "vq"))
                    ]
                    raise ValueError(
                        "Could not extract 'audio_codes'. Codec-related modules found:\n"
                        f"  {codec_related[:30]}\n"
                        "Update the codec encoding path in TalkerDataCollatorPatched."
                    )

        # 4. Set up standard text labels
        labels = encoded["input_ids"].clone()
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        
        # 5. Inject the extracted discrete codes so our patched forward pass can read them
        if audio_codes is not None:
            # Ensure it's a long tensor for CrossEntropyLoss mapping
            encoded["audio_codes"] = audio_codes.long().to(self.model.device)
        else:
            raise ValueError(
                "Could not extract 'audio_codes' from the batch. Double check your model's audio encoding method name."
            )
            
        # 6. CRITICAL STEP: Delete the raw 'audios' key.
        # This completely stops the HuggingFace Trainer from throwing the "ignoring column" warning!
        if "audios" in encoded:
            del encoded["audios"]

        return encoded
    
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML training config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg  = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg  = cfg["lora"]
    quant_cfg = cfg.get("quantization", {})

    print(f"\n=== Stage 2 Talker fine-tuning (HuggingFace) ===")
    print(f"Config : {args.config}")
    print(f"Base_Model  : {cfg['model_path']}")
    print(f"Finetuned_Model  : {cfg['model_name_or_path']}")

    # --- Model ---
    bnb_config = build_bnb_config(quant_cfg) if quant_cfg else None
    model = _ModelCls.from_pretrained(
        cfg["model_name_or_path"],
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = Qwen3OmniMoeProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)

    # Qwen3OmniMoeForConditionalGeneration is a generation-only wrapper with no forward().
    # Patch it to run thinker → talker with cross-entropy loss over all 15 RVQ codebooks.
    if not any('forward' in cls.__dict__
               for cls in type(model).__mro__
               if cls is not torch.nn.Module):
        thinker_sub = getattr(model, 'thinker', None)
        if thinker_sub is None:
            raise RuntimeError(
                f"{type(model).__name__} has no forward() and no .thinker submodule; "
                "cannot train. Check your transformers version."
            )
        type(model).forward = _patched_talker_forward
        print(f"  Patched {type(model).__name__}.forward → thinker→talker (audio codec loss)")

    # Freeze everything; PEFT will unfreeze adapter params
    for param in model.parameters():
        param.requires_grad = False

    # --- LoRA ---
    regex = lora_cfg["target_modules_regex"]
    target_modules = find_talker_targets(model, regex)
    if not target_modules:
        sys.exit(1)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # thinker is a causal LM; PEFT wrapper exposes 'labels' for eval
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    train_entries = load_manifest(
        data_cfg["train_manifest"],
        min_dur=data_cfg.get("min_duration", 0.5),
        max_dur=data_cfg.get("max_duration", 30.0),
    )
    val_entries = load_manifest(
        data_cfg["val_manifest"],
        min_dur=data_cfg.get("min_duration", 0.5),
        max_dur=data_cfg.get("max_duration", 30.0),
    )
    print(f"Train samples: {len(train_entries)}  |  Val samples: {len(val_entries)}")

    collator = TalkerDataCollatorPatched(processor=processor, model= model, sample_rate=data_cfg["sample_rate"])

    # --- TrainingArguments ---
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_steps=train_cfg["warmup_steps"],
        bf16=train_cfg.get("bf16", True),
        tf32=train_cfg.get("tf32", False),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", False),
        optim=train_cfg.get("optim", "adamw_torch_fused"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 500),
        eval_strategy="steps",
        eval_steps=train_cfg.get("save_steps", 500),
        load_best_model_at_end=train_cfg.get("load_best_model_at_end", True),
        metric_for_best_model=train_cfg.get("metric_for_best_model", "eval_loss"),
        # TalkerDataCollatorPatched holds a GPU model reference — must stay in main process.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        report_to="tensorboard",
        remove_unused_columns=False,
        label_names=["labels"],  # PEFT wrappers don't expose 'labels' in forward signature inspection
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_entries),
        eval_dataset=Dataset.from_list(val_entries),
        data_collator=collator,
        processing_class= processor
    )

    trainer.train()
    trainer.save_model()
    print(f"\nTraining complete. LoRA adapter saved to: {train_cfg['output_dir']}")


if __name__ == "__main__":
    main()

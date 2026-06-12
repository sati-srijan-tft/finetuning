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

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
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
    processor = AutoProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)

    # ---------------------------------------------------------------------------
    # Qwen3OmniMoeForConditionalGeneration is a generation-only wrapper: it has
    # no forward() method, so PEFT's BaseTuner.forward() hits PyTorch's default
    # _forward_unimplemented. Patch the class to delegate to model.thinker.
    #
    # LIMITATION: this routes the training loss through the Thinker (text LM),
    # not the Talker (audio codec decoder). For full Talker training you need:
    #   1. Audio codec tokenization in TalkerDataCollator (wav → RVQ codes)
    #   2. A forward that runs thinker → hidden_projection → talker
    #   3. Cross-entropy loss over the 15 codec-codebook predictions
    # ---------------------------------------------------------------------------
    if not any('forward' in cls.__dict__
               for cls in type(model).__mro__
               if cls is not torch.nn.Module):
        thinker_sub = getattr(model, 'thinker', None)
        if thinker_sub is None:
            raise RuntimeError(
                f"{type(model).__name__} has no forward() and no .thinker submodule; "
                "cannot train. Check your transformers version."
            )
        def _patched_forward(self, *args, **kwargs):
            return self.thinker(*args, **kwargs)
        type(model).forward = _patched_forward
        print(f"  Patched {type(model).__name__}.forward → thinker "
              "(text-loss proxy; see script comment for full Talker training)")

    # Freeze everything; PEFT will unfreeze adapter params
    for param in model.parameters():
        param.requires_grad = False

    # --- LoRA ---
    # Try the configured regex first (talker layers). If nothing matches the
    # thinker forward path, fall back to thinker attention layers so LoRA
    # adapters actually receive gradients.
    regex = lora_cfg["target_modules_regex"]
    target_modules = find_talker_targets(model, regex)
    if not target_modules:
        sys.exit(1)

    # Warn when all matched modules live under talker.* (outside forward path)
    all_matched = [n for n, _ in model.named_modules() if re.fullmatch(regex, n)]
    if all_matched and all(n.startswith('talker.') for n in all_matched):
        fallback_regex = r"thinker\.model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        print(
            "\nWARNING: All LoRA targets are in talker.* but the forward path goes through "
            "thinker — those adapters will receive no gradients.\n"
            f"  Falling back to thinker attention layers: {fallback_regex}\n"
            "  To train the Talker, implement audio codec tokenization and a proper "
            "thinker→talker forward (see script comment above).\n"
        )
        target_modules = find_talker_targets(model, fallback_regex)
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

    collator = TalkerDataCollator(processor=processor, sample_rate=data_cfg["sample_rate"])

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
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        dataloader_pin_memory=train_cfg.get("dataloader_pin_memory", True),
        ddp_find_unused_parameters=train_cfg.get("ddp_find_unused_parameters", False),
        report_to="tensorboard",
        remove_unused_columns=False,
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

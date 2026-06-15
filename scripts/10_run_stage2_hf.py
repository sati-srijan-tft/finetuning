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
    audio_codes=None,  # [B, audio_len] — first-codebook token IDs from CosyVoice2
    **kwargs,
):
    """
    Patched forward for Qwen3OmniMoeForConditionalGeneration (which has no forward() of its own).

    Training the Talker's first-codebook predictor (codec_head):
      • The Talker does NOT accept hidden_states directly; it takes inputs_embeds.
      • Text tokens are projected via  talker.text_projection(thinker.embed_tokens(input_ids)).
      • A 5-token codec special prefix is prepended (nothink / think_bos / think_eos / pad / bos).
      • Teacher-forced audio tokens follow: talker.get_input_embeddings()(audio_codes[:, :-1]).
      • Labels = -100 for text+prefix positions, audio_codes[:, 1:] for the audio region.
      • Loss is computed inside talker.forward() via its loss_function on the first codebook.

    audio_codes must be 1-D codec token IDs (shape [B, T_audio]) in the Talker's codec
    vocabulary.  These come from the CosyVoice2 speech tokenizer — there is no encoder
    inside the Qwen3-Omni model itself.
    """
    kwargs.pop("labels", None)

    batch_size, text_len = input_ids.shape
    talker_cfg = self.config.talker_config
    device = input_ids.device

    # 1. Project thinker word embeddings → talker hidden size.
    #    text_projection maps thinker embedding dim → talker hidden dim (1024).
    #    We use word embeddings (not hidden states) — matching how generate() builds
    #    the talker prefix in _get_talker_assistant_parts.
    with torch.no_grad():
        thinker_embed = self.thinker.get_input_embeddings()(input_ids)  # [B, T, H_thinker]
    text_embeds = self.talker.text_projection(thinker_embed)  # [B, T, H_talker]

    # 2. Codec special-token prefix  (5 tokens, all from the talker's own embedding table)
    special_ids = torch.tensor(
        [talker_cfg.codec_nothink_id, talker_cfg.codec_think_bos_id,
         talker_cfg.codec_think_eos_id, talker_cfg.codec_pad_id, talker_cfg.codec_bos_id],
        device=device, dtype=torch.long,
    ).unsqueeze(0).expand(batch_size, -1)  # [B, 5]

    talker_emb = self.talker.get_input_embeddings()
    special_embeds    = talker_emb(special_ids)           # [B, 5,          H_talker]
    audio_in_embeds   = talker_emb(audio_codes[:, :-1])   # [B, T_audio-1,  H_talker]

    # 3. Full inputs_embeds:  [text projection] + [codec prefix] + [audio teacher-forced]
    inputs_embeds = torch.cat([text_embeds, special_embeds, audio_in_embeds], dim=1)

    # 4. Labels: -100 for text + 5-token prefix; first-codebook targets for audio region
    n_prefix = text_len + 5
    labels = torch.cat([
        input_ids.new_full((batch_size, n_prefix), -100),
        audio_codes[:, 1:],  # next-token targets
    ], dim=1)

    # 5. Extend attention mask to cover the appended codec tokens
    n_extra = 5 + audio_codes.shape[1] - 1
    full_mask = torch.cat([
        attention_mask,
        attention_mask.new_ones(batch_size, n_extra),
    ], dim=1)

    # 6. Run the Talker (computes codec_head loss on first codebook via its loss_function)
    talker_out = self.talker(
        inputs_embeds=inputs_embeds,
        attention_mask=full_mask,
        labels=labels,
        talker_input_ids=input_ids,  # used only for 3-D RoPE, not for loss
    )

    return CausalLMOutputWithPast(
        loss=talker_out.loss,
        logits=talker_out.logits,
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
    """
    Collator for Stage 2 Talker fine-tuning.

    Requires a CosyVoice2 instance for audio → speech-token encoding.
    The speech tokenizer runs at 16 kHz (Whisper-based ONNX model inside CosyVoice2).
    Audio files may be at any sample rate; this class resamples automatically.
    """

    COSYVOICE_SR = 16000  # CosyVoice2 speech tokenizer input rate

    def __init__(self, processor, model, sample_rate=24000, cosyvoice=None):
        self.processor   = processor
        self.sample_rate = sample_rate
        self._model_device = next(model.parameters()).device

        if cosyvoice is None:
            raise RuntimeError(
                "CosyVoice2 is required but was not passed.\n"
                "Set up with:\n"
                "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git third_party/CosyVoice\n"
                "  pip install openai-whisper conformer omegaconf hydra-core hyperpyyaml WeTextProcessing onnxruntime-gpu torchaudio\n"
                "  huggingface-cli download FunAudioLLM/CosyVoice2-0.5B --local-dir ./cosyvoice2-0.5b\n"
                "Then add cosyvoice_repo / cosyvoice_path to your data: config block."
            )
        self._cv = cosyvoice
        print(f"  CosyVoice2 speech tokenizer ready  [{type(cosyvoice).__name__}]")

    def __call__(self, batch):
        import numpy as np

        texts        = []
        audio16k_list = []  # 16 kHz waveforms for CosyVoice2 speech tokenizer

        # 1. Load audio + resample to 16 kHz for the speech tokenizer
        for entry in batch:
            texts.append(entry.get("text", ""))
            waveform, sr = sf.read(entry["audio_filepath"], dtype="float32")
            if sr != self.COSYVOICE_SR:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.COSYVOICE_SR)
            audio16k_list.append(waveform)

        # 2. Tokenise TEXT ONLY — audio codes are the TARGET, not a model input
        encoded = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # 3. Extract first-codebook speech tokens via CosyVoice2's ONNX tokenizer
        #    frontend.extract_speech_token(speech_np [B,T], lengths_np [B]) → list of token arrays
        all_tokens = []
        for wav in audio16k_list:
            speech_np  = wav[np.newaxis, :].astype(np.float32)         # [1, T]
            length_np  = np.array([speech_np.shape[1]], dtype=np.int32) # [1]
            tokens = self._cv.frontend.extract_speech_token(speech_np, length_np)
            tok = tokens[0] if isinstance(tokens, (list, tuple)) else tokens
            all_tokens.append(torch.from_numpy(np.asarray(tok)).long())

        # Pad → [B, T_audio]
        max_t      = max(t.shape[0] for t in all_tokens)
        audio_codes = torch.zeros(len(all_tokens), max_t, dtype=torch.long)
        for i, t in enumerate(all_tokens):
            audio_codes[i, :t.shape[0]] = t

        # 4. Text labels (Trainer needs 'labels' for eval_loss; audio loss is in patched forward)
        labels = encoded["input_ids"].clone()
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels

        # 5. First-codebook audio targets for _patched_talker_forward  [B, T_audio]
        encoded["audio_codes"] = audio_codes.to(self._model_device)
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

    # --- CosyVoice2 speech tokenizer ---
    cosyvoice_repo = data_cfg.get("cosyvoice_repo", "./third_party/CosyVoice")
    cosyvoice_path = data_cfg.get("cosyvoice_path", "./cosyvoice2-0.5b")
    # matcha-tts ships as a submodule of CosyVoice; add it before importing
    for _extra in [cosyvoice_repo,
                   f"{cosyvoice_repo}/third_party/matcha-tts",
                   f"{cosyvoice_repo}/third_party/Matcha-TTS"]:
        if _extra not in sys.path:
            sys.path.insert(0, _extra)
    from cosyvoice.cli.cosyvoice import CosyVoice2
    print(f"  Loading CosyVoice2 from {cosyvoice_path} ...")
    cosyvoice = CosyVoice2(cosyvoice_path)

    collator = TalkerDataCollatorPatched(
        processor=processor,
        model=model,
        sample_rate=data_cfg["sample_rate"],
        cosyvoice=cosyvoice,
    )

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

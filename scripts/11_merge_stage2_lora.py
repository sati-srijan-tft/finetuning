#!/usr/bin/env python3
"""
Stage 2 Post-Training: Merges the Stage 2 Talker LoRA adapter into the
Stage 1 merged model, producing a single standalone HuggingFace checkpoint.

Usage:
    # Auto-detect paths (looks for stage2_talker_h100 first, then stage2_talker_bnb)
    python scripts/11_merge_stage2_lora.py

    # Explicit paths
    python scripts/11_merge_stage2_lora.py \
        --base_model  ./LLaMA-Factory/outputs/stage1_merged \
        --adapter     ./outputs/stage2_talker_h100 \
        --output_dir  ./outputs/final_model

Prerequisites:
    pip install transformers peft accelerate
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoProcessor

try:
    from transformers import Qwen3OmniMoeForConditionalGeneration as _ModelCls
except ImportError:
    from transformers import AutoModel as _ModelCls


CANDIDATE_ADAPTERS = [
    "./outputs/stage2_talker_h100",
    "./outputs/stage2_talker_bnb",
]
DEFAULT_BASE = "./qwen3-omni-full-merged"
DEFAULT_OUTPUT = "./outputs/final_model"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model",  default=None,         help="Path to stage1_merged HF checkpoint")
    p.add_argument("--adapter",     default=None,         help="Path to Stage 2 LoRA adapter directory")
    p.add_argument("--output_dir",  default=DEFAULT_OUTPUT, help="Where to save the final merged model")
    p.add_argument("--dtype",       default="bfloat16",   choices=["bfloat16", "float16", "float32"],
                   help="dtype for loading the base model during merge (default: bfloat16)")
    return p.parse_args()


def resolve_adapter(cli_path: str | None) -> str:
    if cli_path:
        if not Path(cli_path).is_dir():
            print(f"ERROR: adapter path not found: {cli_path}")
            sys.exit(1)
        return cli_path
    for candidate in CANDIDATE_ADAPTERS:
        if Path(candidate).is_dir():
            print(f"Auto-detected adapter: {candidate}")
            return candidate
    print("ERROR: No Stage 2 adapter found. Tried:")
    for c in CANDIDATE_ADAPTERS:
        print(f"  {c}")
    print("Run Stage 2 training first, or pass --adapter <path>.")
    sys.exit(1)


def resolve_base(cli_path: str | None) -> str:
    path = cli_path or DEFAULT_BASE
    if not Path(path).is_dir():
        print(f"ERROR: base model not found: {path}")
        print("       Complete Stage 1 merge (04_merge_lora_adapters.sh) first.")
        sys.exit(1)
    return path


def main():
    args = parse_args()

    base_path    = resolve_base(args.base_model)
    adapter_path = resolve_adapter(args.adapter)
    output_dir   = args.output_dir
    dtype        = getattr(torch, args.dtype)

    print("\n=== Stage 2 LoRA Merge — Final Model ===")
    print(f"Base model  : {base_path}")
    print(f"Adapter     : {adapter_path}")
    print(f"Output      : {output_dir}")
    print(f"Merge dtype : {args.dtype}")
    print()

    # --- Load base model in full precision (no quantization during merge) ---
    print("Loading base model (this may take a few minutes)...")
    model = _ModelCls.from_pretrained(
        base_path,
        torch_dtype=dtype,
        device_map="cpu",      # merge on CPU to avoid VRAM limits
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    print("Base model loaded.")

    # --- Load processor / tokenizer from base ---
    print("Loading processor...")
    processor_config_from_hf = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
    processor = AutoProcessor.from_pretrained(processor_config_from_hf, trust_remote_code=True)

    # --- Apply Stage 2 LoRA adapter and merge ---
    print("Applying Stage 2 LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging weights...")
    model = model.merge_and_unload()
    print("Merge complete.")

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving final model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(output_dir)

    print("\n=== Done. ===")
    print(f"Final model saved to: {output_dir}")
    print("\nTest it with:")
    print(f"  python scripts/08_test_inference.py \\")
    print(f"      --model_path {output_dir} \\")
    print(f"      --load_in_4bit \\")
    print(f"      --prompt \"भारत के बारे में बताओ।\"")


if __name__ == "__main__":
    main()

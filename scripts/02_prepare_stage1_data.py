"""
Stage 1 Data Preparation
Combines text-only JSONL instruction data with IndicVoices ASR samples
into a single ChatML (ShareGPT) JSONL file for LLaMA-Factory.

Usage — text data only:
    python scripts/02_prepare_stage1_data.py \
        --input_dir data/ \
        --output_dir LLaMA-Factory/data/ \
        --split_eval

Usage — with IndicVoices ASR data (Hindi):
    python scripts/02_prepare_stage1_data.py \
        --input_dir data/ \
        --output_dir LLaMA-Factory/data/ \
        --use_indicvoices \
        --max_asr_samples 10000 \
        --split_eval

Dependencies (install before running on GPU instance):
    pip install datasets soundfile librosa numpy
"""

import argparse
import json
import random
import shutil
import os
import sys
from pathlib import Path
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

# login to the huggingface account using it hf token

login(token=os.getenv("HF_TOKEN"))

# --- ASR prompt variations (randomly sampled per example for robustness) ---
ASR_PROMPTS = [
    "इस ऑडियो में क्या कहा जा रहा है? कृपया हिंदी में लिखें।",
    "Please transcribe the Hindi audio.",
    "ऑडियो को ध्यान से सुनें और उसे हिंदी में लिखें।",
    "What is being said in this audio clip? Write it in Hindi.",
    "कृपया इस रिकॉर्डिंग का शब्द-दर-शब्द अनुवाद लिखें।",
]

ASR_SYSTEM = "आप एक सहायक AI सहायक हैं जो हिंदी ऑडियो को सुनकर उसे text में बदलने में सक्षम हैं।"


# ---------------------------------------------------------------------------
# Text-only JSONL helpers
# ---------------------------------------------------------------------------

def validate_sample(sample: dict, idx: int) -> list:
    errors = []
    if "messages" not in sample:
        errors.append(f"[{idx}] Missing 'messages' key")
        return errors

    messages = sample["messages"]
    if not isinstance(messages, list) or len(messages) < 2:
        errors.append(f"[{idx}] 'messages' must have at least 2 entries")
        return errors

    valid_roles = {"system", "user", "assistant"}
    for i, msg in enumerate(messages):
        if "role" not in msg:
            errors.append(f"[{idx}] messages[{i}] missing 'role'")
        elif msg["role"] not in valid_roles:
            errors.append(f"[{idx}] messages[{i}] invalid role: {msg['role']}")
        if "content" not in msg:
            errors.append(f"[{idx}] messages[{i}] missing 'content'")
    return errors


def load_jsonl(path: Path) -> list:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  WARNING: skipping line {line_num} — {e}")
    return samples


def write_jsonl(samples: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# IndicVoices ASR loader
# ---------------------------------------------------------------------------

def load_indicvoices(
    output_audio_dir: Path,
    max_samples: int,
    target_sr: int,
    seed: int,
) -> list:
    """
    Downloads IndicVoices Hindi training split, saves audio to disk,
    and returns ChatML-formatted ASR samples.

    Audio paths in returned samples are relative to LLaMA-Factory/data/
    so they can be resolved by the training framework.
    """
    try:
        import numpy as np
        import soundfile as sf
        from datasets import load_dataset
    except ImportError as e:
        print(f"ERROR: Missing dependency — {e}")
        print("  Install with: pip install datasets soundfile numpy")
        sys.exit(1)

    # librosa for resampling (optional — only needed if SR differs)
    librosa = None
    try:
        import librosa as _librosa
        librosa = _librosa
    except ImportError:
        pass

    print(f"\nLoading IndicVoices (Hindi) — up to {max_samples} samples ...")
    print("  Note: First run downloads ~26GB. Subsequent runs use HF cache.")

    # Use streaming to avoid loading everything into RAM
    ds = load_dataset(
        "ai4bharat/IndicVoices",
        "hindi",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    # Shuffle the stream with a buffer so samples aren't from one speaker
    ds = ds.shuffle(seed=seed, buffer_size=2000)

    output_audio_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    skipped = 0
    rng = random.Random(seed)

    for i, row in enumerate(ds):
        if len(samples) >= max_samples:
            break

        # --- Filter: duration ---
        duration = row.get("duration", 0.0)
        if duration is None or not (0.5 <= duration <= 20.0):
            skipped += 1
            continue

        # --- Pick transcript: prefer normalized, fall back to text ---
        transcript = (
            (row.get("normalized") or "").strip()
            or (row.get("text") or "").strip()
        )
        if not transcript:
            skipped += 1
            continue

        # --- Save audio ---
        wav_filename = f"iv_hindi_{i:07d}.wav"
        wav_path = output_audio_dir / wav_filename

        if not wav_path.exists():
            audio = row["audio_filepath"]   # HF audio: {array, sampling_rate, path}
            audio_array = np.array(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]

            if sr != target_sr:
                if librosa is None:
                    print("  WARNING: librosa not installed — skipping resample. Install with: pip install librosa")
                else:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

            sf.write(str(wav_path), audio_array, sr)

        # --- Build ChatML sample ---
        # Path is relative to LLaMA-Factory/data/ which is where LLaMA-Factory resolves media
        rel_audio_path = f"audio_data/indicvoices/{wav_filename}"
        sample = {
            "messages": [
                {"role": "system", "content": ASR_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"audio": rel_audio_path},
                        {"text": rng.choice(ASR_PROMPTS)},
                    ],
                },
                {"role": "assistant", "content": transcript},
            ]
        }
        samples.append(sample)

        if len(samples) % 500 == 0:
            print(f"  {len(samples)}/{max_samples} ASR samples processed (skipped: {skipped}) ...")

    print(f"  IndicVoices: {len(samples)} samples ready, {skipped} skipped")
    return samples


# ---------------------------------------------------------------------------
# dataset_info.json registration
# ---------------------------------------------------------------------------

def build_dataset_info(output_dir: Path, has_eval: bool) -> None:
    base_entry = {
        "file_name": "data.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    info = {"indic_thinker_data": base_entry}
    if has_eval:
        info["indic_thinker_eval"] = {**base_entry, "file_name": "eval.jsonl"}

    info_path = output_dir / "dataset_info.json"
    merged = {}
    if info_path.exists():
        with info_path.open("r", encoding="utf-8") as f:
            merged = json.load(f)
    merged.update(info)

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"  dataset_info.json updated: {info_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 1 training data (text + ASR)")
    parser.add_argument("--input_dir", default="data",
                        help="Directory with text-only .jsonl files")
    parser.add_argument("--output_dir", default="LLaMA-Factory/data",
                        help="LLaMA-Factory data directory")
    parser.add_argument("--use_indicvoices", action="store_true",
                        help="Download and include IndicVoices Hindi ASR data")
    parser.add_argument("--max_asr_samples", type=int, default=10000,
                        help="Max IndicVoices samples to include (default: 10000). "
                             "Full dataset is ~383K — start small and scale up.")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Audio sample rate for saved files (default: 16000 Hz). "
                             "Qwen Omni audio encoder expects 16kHz input.")
    parser.add_argument("--split_eval", action="store_true",
                        help="Reserve a portion of data for evaluation")
    parser.add_argument("--eval_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []

    # --- 1. Load text-only JSONL instruction data ---
    jsonl_files = list(input_dir.glob("*.jsonl"))
    if jsonl_files:
        print("=== Loading text instruction data ===")
        for jf in jsonl_files:
            print(f"  Loading {jf} ...")
            samples = load_jsonl(jf)
            all_samples.extend(samples)

        print(f"Validating {len(all_samples)} text samples ...")
        errors = []
        for i, s in enumerate(all_samples):
            errors.extend(validate_sample(s, i))
        if errors:
            print(f"\n{len(errors)} validation errors:")
            for e in errors[:20]:
                print(f"  {e}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")
            sys.exit(1)
        print(f"  {len(all_samples)} text samples valid.")
    else:
        print("No text .jsonl files found in input_dir — skipping text data.")

    # --- 2. Load IndicVoices ASR data ---
    if args.use_indicvoices:
        print("\n=== Loading IndicVoices ASR data ===")
        audio_output_dir = output_dir / "audio_data" / "indicvoices"
        asr_samples = load_indicvoices(
            output_audio_dir=audio_output_dir,
            max_samples=args.max_asr_samples,
            target_sr=args.target_sr,
            seed=args.seed,
        )
        all_samples.extend(asr_samples)
        print(f"  ASR samples added. Total samples now: {len(all_samples)}")

    if not all_samples:
        print("ERROR: No samples collected. Add .jsonl files or use --use_indicvoices.")
        sys.exit(1)

    # --- 3. Shuffle and split ---
    random.seed(args.seed)
    random.shuffle(all_samples)

    eval_samples = []
    if args.split_eval:
        n_eval = max(1, int(len(all_samples) * args.eval_ratio))
        eval_samples = all_samples[:n_eval]
        all_samples = all_samples[n_eval:]
        print(f"\nSplit — Train: {len(all_samples)} | Eval: {len(eval_samples)}")

    # --- 4. Write outputs ---
    write_jsonl(all_samples, output_dir / "data.jsonl")
    print(f"Wrote {len(all_samples)} training samples → {output_dir / 'data.jsonl'}")

    if eval_samples:
        write_jsonl(eval_samples, output_dir / "eval.jsonl")
        print(f"Wrote {len(eval_samples)} eval samples → {output_dir / 'eval.jsonl'}")

    build_dataset_info(output_dir, has_eval=bool(eval_samples))
    print("\nData preparation complete.")


if __name__ == "__main__":
    main()

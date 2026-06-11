"""
Stage 2 Audio Data Preparation: SPRINGLab/IndicTTS-Hindi
Downloads the HuggingFace dataset, resamples audio to 24 kHz,
saves .wav files, and writes NeMo-format manifests.

Usage:
    python scripts/00_prepare_indicTTS_hindi.py \
        --output_dir /data/indictts_hindi \
        --manifest_dir /path/to/manifests \
        --split_eval \
        --eval_ratio 0.05

Outputs:
    <output_dir>/
        <split>/
            <speaker_id>_<idx>.wav   (24 kHz mono)
    <manifest_dir>/
        train_manifest.json
        val_manifest.json            (if --split_eval)

Requirements:
    pip install datasets soundfile librosa numpy
"""

import argparse
import json
import random
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from datasets import load_dataset


TARGET_SR = 24_000
MIN_DURATION = 0.5   # seconds
MAX_DURATION = 30.0  # seconds


def resample_audio(array: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return array.astype(np.float32)
    return librosa.resample(array.astype(np.float32), orig_sr=src_sr, target_sr=target_sr)


def process_split(
    ds_split,
    split_name: str,
    output_dir: Path,
    target_sr: int,
    max_samples: int | None = None,
) -> list[dict]:
    wav_dir = output_dir / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    skipped = 0

    if max_samples is not None:
        ds_split = ds_split.select(range(min(max_samples, len(ds_split))))

    for idx, example in enumerate(ds_split):
        text = (example.get("text") or example.get("sentence") or "").strip()
        if not text:
            skipped += 1
            continue

        audio_data = example["audio"]
        array = np.array(audio_data["array"], dtype=np.float32)
        src_sr = int(audio_data["sampling_rate"])

        # Convert to mono if multi-channel
        if array.ndim > 1:
            array = array.mean(axis=0)

        resampled = resample_audio(array, src_sr, target_sr)
        duration = len(resampled) / target_sr

        if not (MIN_DURATION <= duration <= MAX_DURATION):
            skipped += 1
            continue

        # Build filename: prefer speaker_id field if present
        speaker = str(example.get("speaker_id") or example.get("speaker") or "spk")
        wav_name = f"{speaker}_{idx:06d}.wav"
        wav_path = wav_dir / wav_name

        sf.write(str(wav_path), resampled, target_sr, subtype="PCM_16")

        entries.append({
            "audio_filepath": str(wav_path),
            "duration": round(duration, 3),
            "text": text,
        })

        if (idx + 1) % 500 == 0:
            print(f"  [{split_name}] processed {idx + 1}/{len(ds_split)} — valid: {len(entries)}, skipped: {skipped}")

    print(f"  [{split_name}] done — valid: {len(entries)}, skipped: {skipped}")
    return entries


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    duration_h = sum(e["duration"] for e in entries) / 3600
    print(f"  Wrote {len(entries)} entries ({duration_h:.2f} h) → {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare IndicTTS-Hindi audio for Stage 2 talker fine-tuning")
    parser.add_argument("--dataset", default="SPRINGLab/IndicTTS-Hindi", help="HuggingFace dataset ID")
    parser.add_argument("--hf_token", default=None, help="HuggingFace token (if dataset is gated)")
    parser.add_argument("--output_dir", required=True, help="Root directory for saved .wav files")
    parser.add_argument("--manifest_dir", default="./manifests", help="Directory for NeMo manifest files")
    parser.add_argument("--target_sr", type=int, default=TARGET_SR, help="Target sample rate in Hz")
    parser.add_argument("--split_eval", action="store_true", help="Carve out a validation split from train")
    parser.add_argument("--eval_ratio", type=float, default=0.05, help="Fraction reserved for validation")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total examples processed per split (useful for dry-runs)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    manifest_dir = Path(args.manifest_dir)

    print(f"=== IndicTTS-Hindi — Stage 2 Audio Prep ===")
    print(f"Dataset:      {args.dataset}")
    print(f"Output dir:   {output_dir}")
    print(f"Manifest dir: {manifest_dir}")
    print(f"Target SR:    {args.target_sr} Hz")

    load_kwargs = {"token": args.hf_token} if args.hf_token else {}
    dataset = load_dataset(args.dataset, trust_remote_code=True, **load_kwargs)
    print(f"Splits available: {list(dataset.keys())}")

    all_entries: list[dict] = []

    # Process every available split — we merge then re-split ourselves
    for split_name, split_ds in dataset.items():
        print(f"\nProcessing split '{split_name}' ({len(split_ds)} examples)…")
        entries = process_split(split_ds, split_name, output_dir, args.target_sr, args.max_samples)
        all_entries.extend(entries)

    if not all_entries:
        print("ERROR: No valid audio entries found. Check dataset structure.")
        return

    total_h = sum(e["duration"] for e in all_entries) / 3600
    print(f"\nTotal valid entries: {len(all_entries)} ({total_h:.2f} h)")

    random.seed(args.seed)
    random.shuffle(all_entries)

    val_entries: list[dict] = []
    if args.split_eval:
        n_val = max(1, int(len(all_entries) * args.eval_ratio))
        val_entries = all_entries[:n_val]
        all_entries = all_entries[n_val:]

    write_manifest(all_entries, manifest_dir / "train_manifest.json")
    if val_entries:
        write_manifest(val_entries, manifest_dir / "val_manifest.json")

    print("\nDone. Update configs/talker_finetune.yaml with:")
    print(f"  train_ds.manifest_filepath: \"{manifest_dir / 'train_manifest.json'}\"")
    if val_entries:
        print(f"  validation_ds.manifest_filepath: \"{manifest_dir / 'val_manifest.json'}\"")


if __name__ == "__main__":
    main()

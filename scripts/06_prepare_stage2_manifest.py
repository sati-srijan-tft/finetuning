"""
Stage 2 Data Preparation: Generates NeMo-format audio manifests from a directory
of .wav files and their corresponding transcripts.

Expected input directory layout:
    audio_data/
        hindi/
            clip_001.wav
            clip_001.txt        (transcript, one line)
            clip_002.wav
            clip_002.txt
        tamil/
            ...

Usage:
    python scripts/06_prepare_stage2_manifest.py \
        --audio_dir /data/indic_tts \
        --output_dir /path/to/manifests \
        --target_sr 24000 \
        --split_eval

Outputs:
    train_manifest.json
    val_manifest.json  (if --split_eval)
"""

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path


def get_audio_duration(wav_path: Path) -> float:
    """Returns duration in seconds using ffprobe (must be installed)."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        capture_output=True, text=True
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def resample_audio(src: Path, dst: Path, target_sr: int) -> bool:
    """Resamples audio to target sample rate using ffmpeg. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ar", str(target_sr), str(dst)],
        capture_output=True
    )
    return result.returncode == 0


def build_manifest(audio_dir: Path, target_sr: int, resampled_dir: Path) -> list[dict]:
    entries = []
    wav_files = sorted(audio_dir.rglob("*.wav"))

    if not wav_files:
        print(f"  WARNING: No .wav files found in {audio_dir}")
        return entries

    print(f"  Found {len(wav_files)} .wav files in {audio_dir}")

    for wav_path in wav_files:
        # Find transcript: same stem, .txt extension
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  SKIP: No transcript found for {wav_path.name}")
            continue

        transcript = txt_path.read_text(encoding="utf-8").strip()
        if not transcript:
            print(f"  SKIP: Empty transcript for {wav_path.name}")
            continue

        # Resample to target SR
        rel_path = wav_path.relative_to(audio_dir)
        resampled_path = resampled_dir / rel_path
        if not resampled_path.exists():
            if not resample_audio(wav_path, resampled_path, target_sr):
                print(f"  SKIP: Resampling failed for {wav_path.name}")
                continue

        duration = get_audio_duration(resampled_path)
        if duration <= 0:
            print(f"  SKIP: Could not determine duration for {resampled_path.name}")
            continue

        # Filter out very short or very long clips
        if duration < 0.5 or duration > 30.0:
            print(f"  SKIP: Duration {duration:.1f}s out of range for {wav_path.name}")
            continue

        entries.append({
            "audio_filepath": str(resampled_path),
            "duration": round(duration, 3),
            "text": transcript,
        })

    return entries


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(entries)} entries → {path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Stage 2 NeMo audio manifests")
    parser.add_argument("--audio_dir", required=True, help="Root directory of audio + transcript files")
    parser.add_argument("--output_dir", default="./manifests", help="Directory for output manifest files")
    parser.add_argument("--resampled_dir", default=None, help="Directory for resampled audio (default: audio_dir/../resampled)")
    parser.add_argument("--target_sr", type=int, default=24000, help="Target sample rate in Hz (default: 24000)")
    parser.add_argument("--split_eval", action="store_true", help="Reserve 5%% of data for validation")
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    resampled_dir = Path(args.resampled_dir) if args.resampled_dir else audio_dir.parent / "resampled"

    if not audio_dir.exists():
        print(f"ERROR: audio_dir not found: {audio_dir}")
        sys.exit(1)

    # Check ffmpeg/ffprobe available
    for tool in ("ffmpeg", "ffprobe"):
        result = subprocess.run(["which", tool], capture_output=True)
        if result.returncode != 0:
            print(f"ERROR: '{tool}' not found. Install with: apt-get install ffmpeg")
            sys.exit(1)

    print(f"=== Stage 2 Manifest Preparation ===")
    print(f"Audio dir:     {audio_dir}")
    print(f"Resampled dir: {resampled_dir}")
    print(f"Target SR:     {args.target_sr} Hz")

    entries = build_manifest(audio_dir, args.target_sr, resampled_dir)

    if not entries:
        print("ERROR: No valid audio/transcript pairs found.")
        sys.exit(1)

    print(f"\nTotal valid entries: {len(entries)}")
    total_duration = sum(e["duration"] for e in entries)
    print(f"Total audio duration: {total_duration / 3600:.2f} hours")

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(entries)

    val_entries = []
    if args.split_eval:
        n_val = max(1, int(len(entries) * args.eval_ratio))
        val_entries = entries[:n_val]
        entries = entries[n_val:]

    write_manifest(entries, output_dir / "train_manifest.json")
    if val_entries:
        write_manifest(val_entries, output_dir / "val_manifest.json")

    print("\nManifest preparation complete.")
    print(f"Update configs/talker_finetune.yaml with manifest paths in {output_dir}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Repaint editing benchmark for ACE-Step 1.5 (dual-axis protocol).

Stages: prepare (MUSDB18 mixture extraction) -> edit (random regions x
style-pool captions, turbo repaint with explicit chunk mask) -> report
(consistency axis: out-of-region mel L1 + full-track chroma similarity;
fidelity axis: in-region CLAP adherence to the caption; combined ASB).

Example:
    CUDA_VISIBLE_DEVICES=1 uv run python repaint_bench.py --smoke
    CUDA_VISIBLE_DEVICES=1 uv run python repaint_bench.py            # 20 tracks
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")

from loguru import logger
import numpy as np
import soundfile as sf
import torch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from edit_metrics import ClapScorer, asb_scores, chroma_similarity, region_mel_l1  # noqa: E402

SR = 48000
MUSDB_TEST = Path("/data/local/home/zll/musdb18/test")
CLAP_CKPT = "/data/local/tmp/ceval/music_audioset_epoch_15_esc_90.14.pt"
MOODS = ["upbeat", "relaxing", "peaceful", "melancholic"]
GENRES = ["electronic", "jazz", "rock", "folk", "classical"]
TIMBRES = ["synthesizer", "acoustic guitar", "piano", "strings"]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Repaint dual-axis benchmark (MUSDB18)")
    p.add_argument("--musdb", type=Path, default=MUSDB_TEST)
    p.add_argument("--num-tracks", type=int, default=20)
    p.add_argument("--regions", type=int, default=2, help="Random regions per track")
    p.add_argument("--captions", type=int, default=2, help="Captions per region")
    p.add_argument("--trim", type=float, default=40.0)
    p.add_argument("--min-len", type=float, default=8.0, help="Min region length (s)")
    p.add_argument("--max-len", type=float, default=15.0, help="Max region length (s)")
    p.add_argument("--sample-seed", type=int, default=42, help="Seed for region/caption sampling")
    p.add_argument("--seed", type=int, default=42, help="Generation seed")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=7.0, help="CFG scale (turbo forces 1.0)")
    p.add_argument("--shift", type=float, default=1.0, help="Timestep shift (3.0 for base/turbo)")
    p.add_argument("--repaint-mode", default="balanced", choices=["balanced", "aggressive", "conservative"])
    p.add_argument("--repaint-strength", type=float, default=0.5,
                   help="0=conservative (max source injection), 1=aggressive (pure diffusion); balanced mode only")
    p.add_argument("--config", default="acestep-v15-turbo")
    p.add_argument("--clap-ckpt", default=CLAP_CKPT)
    p.add_argument("--clap-device", default="cpu")
    p.add_argument("--out", type=Path, default=REPO / "gradio_outputs" / "repaint_eval")
    p.add_argument("--smoke", action="store_true", help="1 track / 1 region / 1 caption")
    return p


def sample_conditions(args: argparse.Namespace, tracks: list[Path], rng: random.Random):
    """Sample (track, start, end, caption) edit conditions with a fixed RNG."""
    conditions = []
    for track in tracks:
        name = track.name.replace(".stem.mp4", "")
        for _ in range(args.regions):
            length = rng.uniform(args.min_len, args.max_len)
            start = rng.uniform(5.0, args.trim - length - 2.0)
            for _ in range(args.captions):
                caption = (f"A {rng.choice(MOODS)} {rng.choice(GENRES)} music "
                           f"with {rng.choice(TIMBRES)} performance.")
                conditions.append((name, round(start, 2), round(start + length, 2), caption))
    return conditions


def region_band(start: float, trim: float) -> str:
    """Label region position as front/mid/back for grouped reporting."""
    frac = start / trim
    return "front" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "back")


def main() -> None:
    """Run prepare -> edit -> report for the repaint benchmark."""
    args = build_parser().parse_args()
    if args.smoke:
        args.num_tracks, args.regions, args.captions = 1, 1, 1
    src_dir, out_dir = args.out / "sources", args.out / "outputs"
    for d in (src_dir, out_dir):
        d.mkdir(parents=True, exist_ok=True)

    tracks = sorted(args.musdb.glob("*.stem.mp4"))[: args.num_tracks]
    if not tracks:
        logger.error(f"No .stem.mp4 found under {args.musdb}")
        sys.exit(1)

    # ---- prepare sources ----
    sources = {}
    for mp4 in tracks:
        wav = src_dir / (mp4.name.replace(".stem.mp4", ".wav"))
        if not wav.exists():
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4), "-map", "0:a:0",
                 "-t", str(args.trim), "-ar", str(SR), "-ac", "1", str(wav)],
                check=True,
            )
        sources[mp4.name.replace(".stem.mp4", "")] = wav
    logger.info(f"Prepared {len(sources)} source clips")

    conditions = sample_conditions(args, tracks, random.Random(args.sample_seed))
    logger.info(f"{len(conditions)} edit conditions "
                f"({len(tracks)} tracks x {args.regions} regions x {args.captions} captions)")

    # ---- generate edits ----
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(REPO), config_path=args.config, device="auto",
    )
    if not ok:
        logger.error(f"DiT init failed: {status}")
        sys.exit(1)

    outputs = []
    for i, (track, start, end, caption) in enumerate(conditions, 1):
        src = sources[track]
        t0 = time.time()
        params = GenerationParams(
            task_type="repaint", src_audio=str(src), caption=caption,
            repainting_start=start, repainting_end=end,
            chunk_mask_mode="explicit", duration=args.trim,
            enable_normalization=False, seed=args.seed, inference_steps=args.steps,
            guidance_scale=args.guidance, shift=args.shift,
            repaint_mode=args.repaint_mode, repaint_strength=args.repaint_strength,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
        result = generate_music(dit, None, params, config, save_dir=str(out_dir))
        if not result.success:
            logger.error(f"[{i}/{len(conditions)}] FAILED {track} @{start}: "
                         f"{result.error or result.status_message}")
            continue
        path = result.audios[0]["path"]
        outputs.append((track, start, end, caption, Path(path)))
        logger.info(f"[{i}/{len(conditions)}] {track} [{start},{end}] ({time.time() - t0:.1f}s)")

    # ---- score: dual axis + ASB ----
    logger.info("Scoring (mel / chroma / CLAP)...")
    scorer = ClapScorer(ckpt=args.clap_ckpt, device=args.clap_device)
    text_cache: dict[str, np.ndarray] = {}

    def clap_follow(wav_path: Path, caption: str, start: float, end: float) -> float:
        """CLAP cosine between the repainted segment and its caption."""
        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        seg = wav[int(start * sr):int(end * sr)].mean(axis=1)
        emb = scorer.audio_embedding(seg)
        if caption not in text_cache:
            text_cache[caption] = scorer.text_embedding([caption])[0]
        return scorer.cosine(emb, text_cache[caption])

    rows = []
    for track, start, end, caption, edited in outputs:
        src = sources[track]
        rows.append({
            "track": track, "start": start, "end": end,
            "band": region_band(start, args.trim), "caption": caption,
            "seed": args.seed,
            "l1_outside": region_mel_l1(src, edited, start, end),
            "chroma_sim": chroma_similarity(src, edited),
            "clap_follow": clap_follow(edited, caption, start, end),
            "output": str(edited),
        })

    asb = asb_scores([r["clap_follow"] for r in rows], [r["l1_outside"] for r in rows])
    for row, value in zip(rows, asb):
        row["asb"] = value

    csv_path = args.out / "report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stat(key: str) -> str:
        vals = [r[key] for r in rows if r[key] == r[key]]
        if not vals:
            return "nan"
        vals.sort()
        mid = vals[len(vals) // 2]
        return f"mean={sum(vals) / len(vals):.4f} median={mid:.4f}"

    logger.info(f"Report: {csv_path} ({len(rows)} rows)")
    logger.info(f"CONSISTENCY  outside-mel-L1 (lower=better): {stat('l1_outside')}")
    logger.info(f"CONSISTENCY  chroma-sim (higher=better):   {stat('chroma_sim')}")
    logger.info(f"FIDELITY     in-region CLAP (higher=better): {stat('clap_follow')}")
    logger.info(f"BALANCE      ASB (higher=better):           {stat('asb')}")
    for band in ("front", "mid", "back"):
        part = [r for r in rows if r["band"] == band]
        if part:
            logger.info(f"  band={band:5s} n={len(part):3d} "
                        f"outside={sum(r['l1_outside'] for r in part) / len(part):.4f} "
                        f"clap={sum(r['clap_follow'] for r in part) / len(part):.4f}")


if __name__ == "__main__":
    main()

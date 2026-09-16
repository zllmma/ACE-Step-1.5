#!/usr/bin/env python3
"""Batch-generate the Chinese song corpus (100 tracks) for lyric-editing evaluation.

Reads official example JSONs (corpus_official_zh.json) plus hand-written ones
(corpus_new_zh.json), generates each with xl-turbo + 1.7B LM (thinking on),
validates duration/RMS/peak, retries once, and skips outputs that already
exist so interrupted runs resume.

Example:
    CUDA_VISIBLE_DEVICES=1 uv run python gen_corpus.py --pilot 5
    CUDA_VISIBLE_DEVICES=1 uv run python gen_corpus.py            # full 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

from loguru import logger
import soundfile as sf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Generate Chinese song corpus")
    p.add_argument("--out", type=Path, default=REPO / "gen_corpus_zh")
    p.add_argument("--num", type=int, default=100)
    p.add_argument("--pilot", type=int, default=0,
                   help="Pilot mode: only generate the first N tracks")
    p.add_argument("--config", default="acestep-v15-xl-turbo")
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--lm", default="acestep-5Hz-lm-1.7B")
    p.add_argument("--lm-backend", default="vllm")
    p.add_argument("--seed-base", type=int, default=1000)
    return p


def load_items(num: int) -> list[dict]:
    """Load official + handwritten Chinese items up to num."""
    official = json.load(open(REPO / "corpus_official_zh.json"))
    new = json.load(open(REPO / "corpus_new_zh.json"))
    new += json.load(open(REPO / "corpus_new_zh2.json"))
    items = official + new
    for i, it in enumerate(items):
        it["corpus_index"] = i
        it["corpus_source"] = "official" if i < len(official) else "handwritten"
    return items[:num]


def quality_check(wav_path: Path, req_dur: float) -> tuple[bool, float, float, float]:
    """Validate duration/RMS/peak of a generated clip."""
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    dur = wav.shape[0] / sr
    rms = float((wav ** 2).mean()) ** 0.5
    peak = float(abs(wav).max())
    ok = dur >= req_dur * 0.8 and rms > 1e-4 and peak < 0.995
    return ok, dur, rms, peak


def main() -> None:
    """Generate all corpus items with retry and resume support."""
    args = build_parser().parse_args()
    items = load_items(args.num)
    if args.pilot:
        items = [items[i] for i in (0, 20, 60, 73, 85)][: args.pilot]
        logger.info(f"PILOT: tracks {[(i['corpus_index'], i['corpus_source']) for i in items]}")

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    done = set()
    if manifest_path.exists():
        for line in open(manifest_path).read().splitlines()[1:]:
            parts = line.split(",")
            if len(parts) > 8 and parts[8] == "ok":
                done.add(int(parts[0]))
        if done:
            logger.info(f"Resuming: {len(done)} tracks already done")

    logger.info("Initializing DiT (xl-turbo) and LM (1.7B vllm)...")
    dit = AceStepHandler()
    status, ok = dit.initialize_service(project_root=str(REPO), config_path=args.config, device="auto")
    if not ok:
        logger.error(f"DiT init failed: {status}")
        sys.exit(1)
    llm = LLMHandler()
    status, ok = llm.initialize(
        checkpoint_dir=str(REPO / "checkpoints"), lm_model_path=args.lm,
        backend=args.lm_backend, device="auto",
    )
    if not ok:
        logger.error(f"LM init failed: {status}")
        sys.exit(1)

    new_lines = []
    for it in items:
        idx = it["corpus_index"]
        if idx in done:
            continue
        seed = args.seed_base + idx
        params = GenerationParams(
            task_type="text2music", caption=it["caption"], lyrics=it["lyrics"],
            bpm=it.get("bpm"), keyscale=it.get("keyscale", ""),
            timesignature=it.get("timesignature", ""),
            vocal_language=it.get("language", "zh"), duration=it.get("duration"),
            thinking=True, seed=seed, inference_steps=args.steps,
            guidance_scale=args.guidance, shift=args.shift,
        )
        config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")

        wav_path, dur, rms, peak, status_str = "", 0.0, 0.0, 0.0, "failed"
        for attempt in (1, 2):
            t0 = time.time()
            result = generate_music(dit, llm, params, config, save_dir=str(out_dir))
            if not result.success:
                logger.warning(f"[{idx}] attempt {attempt} failed: {result.error or result.status_message}")
                continue
            wav_path = result.audios[0]["path"]
            ok_q, dur, rms, peak = quality_check(Path(wav_path), float(it.get("duration") or 0))
            status_str = "ok" if ok_q else "low_quality"
            logger.info(f"[{idx}] {it['corpus_source']:11s} {dur:.0f}s rms={rms:.3f} "
                        f"({time.time() - t0:.0f}s) -> {Path(wav_path).name}")
            break
        else:
            logger.error(f"[{idx}] both attempts failed, skipping")
        new_lines.append(
            f"{idx},{it['corpus_source']},{seed},{it.get('duration') or ''},{dur:.1f},"
            f"{rms:.4f},{peak:.3f},{wav_path},{status_str}"
        )
        with open(manifest_path, "a") as f:
            if f.tell() == 0:
                f.write("index,source,seed,dur_req,dur_act,rms,peak,output,status\n")
            f.write(new_lines[-1] + "\n")

    n_ok = sum(1 for l in new_lines if l.endswith("ok"))
    logger.info(f"Done: {n_ok} ok / {len(new_lines)} generated this run. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CLI music generation for research experiments (Linux x86_64 + CUDA).

Wraps AceStepHandler + LLMHandler + generate_music with command-line args.
Defaults match the local environment (turbo DiT + 1.7B LM + vllm backend).

Examples:
    python generate.py --caption "upbeat electronic dance music" --duration 30
    python generate.py --example examples/text2music/example_01.json
    python generate.py -c "jazz piano trio" --duration 20 --no-lm --seed 42
"""

import argparse
import json
import os
import sys
import time

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("ALL_PROXY", None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="ACE-Step music generation (research CLI)")
    p.add_argument("-c", "--caption", default="upbeat electronic dance music with heavy bass")
    p.add_argument("--lyrics", default="", help="Lyrics string ('[Instrumental]' for instrumental)")
    p.add_argument("--lyrics-file", default=None, help="Read lyrics from a text file")
    p.add_argument("--example", default=None, help="Load caption/lyrics/metadata from an example JSON")
    p.add_argument("--task", default="text2music",
                   choices=["text2music", "repaint", "cover", "cover-nofsq", "extract", "lego", "complete"])
    p.add_argument("--src-audio", default=None, help="Source audio for cover/repaint/extract tasks")
    p.add_argument("--instruction", default="", help="Task instruction (lego/extract/complete)")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--bpm", type=int, default=None)
    p.add_argument("--keyscale", default="")
    p.add_argument("--timesignature", default="")
    p.add_argument("--language", default="unknown")
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--shift", type=float, default=3.0, help="Timestep shift (3.0 recommended for turbo)")
    p.add_argument("--guidance", type=float, default=7.0, help="CFG scale (ignored by turbo)")
    p.add_argument("--no-thinking", action="store_true", help="Disable LM CoT (faster, no metadata refinement)")
    p.add_argument("--audio-format", default="flac", choices=["flac", "wav", "wav32", "mp3", "opus", "aac"])
    p.add_argument("--out", default=os.path.join("gradio_outputs", "generate"))
    # Model selection
    p.add_argument("--config", default="acestep-v15-turbo", help="DiT model name")
    p.add_argument("--lm", default="acestep-5Hz-lm-1.7B", help="LM model name")
    p.add_argument("--backend", default="vllm", choices=["vllm", "pt"])
    p.add_argument("--no-lm", action="store_true", help="Skip LM entirely (DiT-only generation)")
    p.add_argument("--offload", action="store_true", help="Enable CPU offload (VRAM < 20GB)")
    return p


def load_example_args(args: argparse.Namespace) -> None:
    """Overlay caption/lyrics/metadata from an example JSON onto args."""
    with open(args.example, "r", encoding="utf-8") as f:
        ex = json.load(f)
    args.caption = ex.get("caption", args.caption)
    args.lyrics = ex.get("lyrics", args.lyrics)
    if ex.get("bpm") is not None:
        args.bpm = ex["bpm"]
    args.keyscale = ex.get("keyscale", args.keyscale)
    args.timesignature = ex.get("timesignature", args.timesignature)
    args.language = ex.get("language", args.language)
    if ex.get("duration") is not None:
        args.duration = ex["duration"]


def main() -> None:
    """Parse args, initialize handlers, generate music, print results."""
    args = build_parser().parse_args()
    if args.lyrics_file:
        with open(args.lyrics_file, "r", encoding="utf-8") as f:
            args.lyrics = f.read()
    if args.example:
        load_example_args(args)

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.out, exist_ok=True)
    thinking = not args.no_thinking and not args.no_lm

    t0 = time.time()
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=project_root,
        config_path=args.config,
        device="auto",
        offload_to_cpu=args.offload,
    )
    if not ok:
        logger.error(f"DiT init failed: {status}")
        sys.exit(1)
    logger.info(f"DiT [{args.config}] loaded in {time.time() - t0:.1f}s")

    llm = None
    if not args.no_lm:
        from acestep.llm_inference import LLMHandler

        t0 = time.time()
        llm = LLMHandler()
        status, ok = llm.initialize(
            checkpoint_dir=os.path.join(project_root, "checkpoints"),
            lm_model_path=args.lm,
            backend=args.backend,
            device="auto",
        )
        if not ok:
            logger.error(f"LM init failed: {status}")
            sys.exit(1)
        logger.info(f"LM [{args.lm}/{args.backend}] loaded in {time.time() - t0:.1f}s")
    else:
        logger.info("LM skipped (--no-lm), DiT-only generation")

    params = GenerationParams(
        task_type=args.task,
        caption=args.caption,
        lyrics=args.lyrics,
        instrumental=(args.lyrics == "" and not thinking),
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        vocal_language=args.language,
        duration=args.duration,
        inference_steps=args.steps,
        shift=args.shift,
        guidance_scale=args.guidance,
        seed=args.seed,
        thinking=thinking,
        src_audio=args.src_audio,
        instruction=args.instruction,
    )
    config = GenerationConfig(
        batch_size=args.batch,
        audio_format=args.audio_format,
        use_random_seed=(args.seed < 0),
    )

    logger.info(f"Generating: task={args.task} duration={args.duration}s batch={args.batch} "
                f"seed={args.seed if args.seed >= 0 else 'random'} thinking={thinking}")
    t0 = time.time()
    result = generate_music(dit, llm, params, config, save_dir=args.out)
    elapsed = time.time() - t0

    if not result.success:
        logger.error(f"Generation FAILED ({elapsed:.1f}s): {result.error or result.status_message}")
        sys.exit(1)

    costs = result.extra_outputs.get("time_costs", {}) or {}
    logger.info(f"Done in {elapsed:.1f}s "
                f"(lm1={costs.get('lm_phase1_time', 0):.1f}s lm2={costs.get('lm_phase2_time', 0):.1f}s "
                f"dit={costs.get('dit_total_time_cost', 0):.1f}s)")
    for audio in result.audios:
        logger.info(f"  seed={audio['params'].get('seed')} -> {audio['path']}")


if __name__ == "__main__":
    main()

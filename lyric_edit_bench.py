#!/usr/bin/env python3
"""Lyric-editing (flow-edit) benchmark for ACE-Step 1.5.

Protocol per condition (from lyric_edit_conditions.json):
  1. EDIT  - flow_edit_morph with edited lyrics on the source song
  2. REGEN - plain text2music regeneration with the edited lyrics (baseline:
             "edited by recomposing" -- should break melody/accompaniment)
  3. SCORE - whisper transcription of the edit -> new-keyword hit rate and
             old-keyword residual rate (fidelity axis); chroma similarity and
             CQT1-PCC against the source (consistency axis), also computed
             for the regen baseline for contrast.

Example:
    CUDA_VISIBLE_DEVICES=1 uv run python lyric_edit_bench.py --probe
    CUDA_VISIBLE_DEVICES=1 uv run python lyric_edit_bench.py
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHAUDIO_USE_BACKEND", "ffmpeg")

from loguru import logger
import numpy as np

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402
from acestep.llm_inference import LLMHandler  # noqa: E402
from edit_metrics import chroma_similarity, cqt_pcc  # noqa: E402

CONDITIONS = REPO / "lyric_edit_conditions.json"
MODELS = {
    "turbo": {"config": "acestep-v15-turbo", "steps": 8, "guidance": 1.0, "shift": 3.0},
    "xlturbo": {"config": "acestep-v15-xl-turbo", "steps": 8, "guidance": 1.0, "shift": 3.0},
    "xlbase": {"config": "acestep-v15-xl-base", "steps": 50, "guidance": 7.0, "shift": 3.0},
}
PUNCT = re.compile(r"[，。！？；：、,.!?;:\s\"'“”‘’\[\]()（）<>《》\-—…]")


def normalize(text: str) -> str:
    """Strip punctuation/whitespace for tolerant keyword matching."""
    return PUNCT.sub("", text)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(description="Lyric-edit (flow-edit) benchmark")
    p.add_argument("--model", choices=list(MODELS), default=None,
                   help="Skip probe and force a model")
    p.add_argument("--num", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--asr-model", default="openai/whisper-small")
    p.add_argument("--out", type=Path, default=REPO / "lyric_edit_eval")
    p.add_argument("--probe", action="store_true",
                   help="Only run conditions[0] on all candidate models, then pick")
    return p


def whisper_transcribe(pipeline, wav_path: Path) -> str:
    """Transcribe a full song, forcing Chinese decoding (long-form enabled)."""
    out = pipeline(str(wav_path), return_timestamps=True,
                   generate_kwargs={"language": "chinese"})
    return normalize(out.get("text", ""))


def keyword_stats(transcript: str, new_keys: list[str], old_keys: list[str]):
    """Return (new_hit_rate, old_residual_rate) by substring containment."""
    new_hit = sum(1 for k in new_keys if normalize(k) in transcript)
    old_res = sum(1 for k in old_keys if normalize(k) in transcript)
    return new_hit / max(1, len(new_keys)), old_res / max(1, len(old_keys))


def flow_edit(dit, llm, cond: dict, model_cfg: dict, args, out_dir: Path) -> Path:
    """Run one flow-edit generation and return the output path."""
    params = GenerationParams(
        task_type="text2music", src_audio=cond["source_wav"],
        caption=cond["caption"], lyrics=cond["edited_lyrics"],
        flow_edit_morph=True,
        flow_edit_source_caption=cond["caption"],
        flow_edit_source_lyrics=cond["orig_lyrics"],
        thinking=False, duration=cond["duration"],
        seed=args.seed, inference_steps=model_cfg["steps"],
        guidance_scale=model_cfg["guidance"], shift=model_cfg["shift"],
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    result = generate_music(dit, llm, params, config, save_dir=str(out_dir))
    if not result.success:
        raise RuntimeError(result.error or result.status_message)
    return Path(result.audios[0]["path"])


def regen_baseline(dit, llm, cond: dict, model_cfg: dict, args, out_dir: Path) -> Path:
    """Regenerate with edited lyrics and no source (baseline)."""
    params = GenerationParams(
        task_type="text2music", caption=cond["caption"], lyrics=cond["edited_lyrics"],
        thinking=False, duration=cond["duration"],
        seed=args.seed, inference_steps=model_cfg["steps"],
        guidance_scale=model_cfg["guidance"], shift=model_cfg["shift"],
    )
    config = GenerationConfig(batch_size=1, use_random_seed=False, audio_format="wav")
    result = generate_music(dit, llm, params, config, save_dir=str(out_dir))
    if not result.success:
        raise RuntimeError(result.error or result.status_message)
    return Path(result.audios[0]["path"])


def main() -> None:
    """Run probe (optional) -> edit/regen -> transcribe -> score."""
    args = build_parser().parse_args()
    conds = json.load(open(CONDITIONS))[: args.num]
    for c in conds:
        matches = sorted((REPO / "gen_corpus_zh").glob(f"{int(c['title_file_stub']):03d}_*.wav"))
        if not matches:
            logger.error(f"source wav missing for condition {c['index']}")
            sys.exit(1)
        c["source_wav"] = str(matches[0])

    model_name = args.model
    candidates = [model_name] if model_name else list(MODELS)
    asr = None  # lazy init after generation so downloads don't block model choice

    edit_dir = args.out / "edits"
    regen_dir = args.out / "regen"
    for d in (edit_dir, regen_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.probe:
        asr = None
        best_name, best_score = None, -1.0
        for name in candidates:
            cfg = MODELS[name]
            dit = AceStepHandler()
            status, ok = dit.initialize_service(
                project_root=str(REPO), config_path=cfg["config"], device="auto")
            if not ok:
                logger.error(f"{name}: DiT init failed: {status}")
                continue
            c0 = conds[0]
            t0 = time.time()
            edit_wav = flow_edit(dit, None, c0, cfg, args, edit_dir / f"probe_{name}")
            dt = time.time() - t0
            if asr is None:
                from transformers import pipeline as hf_pipeline
                asr = hf_pipeline("automatic-speech-recognition",
                                  model=args.asr_model, device="cpu")
            transcript_probe = whisper_transcribe(asr, edit_wav)
            new_hit, old_res = keyword_stats(transcript_probe,
                                             c0["eval_new_keywords"], c0["eval_old_keywords"])
            chroma = chroma_similarity(c0["source_wav"], edit_wav)
            score = new_hit - old_res
            logger.info(f"PROBE {name}: {dt:.0f}s new_hit={new_hit:.2f} old_res={old_res:.2f} "
                        f"chroma={chroma:.3f} transcript[:80]={transcript_probe[:80]}")
            if score > best_score:
                best_name, best_score = name, score
        logger.info(f"PROBE verdict: best model = {best_name} (score {best_score:.2f}) "
                    f"-- rerun with --model {best_name}")
        return

    # ---- full run with chosen model ----
    cfg = MODELS[model_name]
    dit = AceStepHandler()
    status, ok = dit.initialize_service(
        project_root=str(REPO), config_path=cfg["config"], device="auto")
    if not ok:
        logger.error(f"DiT init failed: {status}")
        sys.exit(1)
    llm = None
    if asr is None:
        from transformers import pipeline as hf_pipeline
        asr = hf_pipeline("automatic-speech-recognition",
                          model=args.asr_model, device="cpu")

    rows = []
    for i, cond in enumerate(conds, 1):
        try:
            edit_wav = flow_edit(dit, llm, cond, cfg, args, edit_dir)
            regen_wav = regen_baseline(dit, llm, cond, cfg, args, regen_dir)
        except RuntimeError as e:
            logger.error(f"[{i}/{len(conds)}] generation failed: {e}")
            continue
        transcript = whisper_transcribe(asr, edit_wav)
        transcript_r = whisper_transcribe(asr, regen_wav)
        new_hit, old_res = keyword_stats(transcript,
                                         cond["eval_new_keywords"], cond["eval_old_keywords"])
        new_hit_r, _ = keyword_stats(transcript_r,
                                     cond["eval_new_keywords"], cond["eval_old_keywords"])
        rows.append({
            "index": cond["index"], "model": model_name,
            "caption": cond["caption"][:60],
            "new_hit": new_hit, "old_residual": old_res,
            "follow_asb": new_hit * (1 - old_res),
            "chroma_edit": chroma_similarity(cond["source_wav"], edit_wav),
            "cqt_pcc_edit": cqt_pcc(cond["source_wav"], edit_wav),
            "chroma_regen": chroma_similarity(cond["source_wav"], regen_wav),
            "cqt_pcc_regen": cqt_pcc(cond["source_wav"], regen_wav),
            "clap_probe_regen": new_hit_r,
            "edit_wav": str(edit_wav), "regen_wav": str(regen_wav),
            "transcript": transcript[:200],
        })
        logger.info(f"[{i}/{len(conds)}] new_hit={new_hit:.2f} old_res={old_res:.2f} "
                    f"cqt_edit={rows[-1]['cqt_pcc_edit']:.3f} "
                    f"cqt_regen={rows[-1]['cqt_pcc_regen']:.3f}")

    csv_path = args.out / "report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def mean(key):
        vals = [r[key] for r in rows if r[key] == r[key]]
        return sum(vals) / len(vals) if vals else float("nan")

    logger.info(f"Report: {csv_path} ({len(rows)} rows, model={model_name})")
    logger.info(f"FIDELITY   new_hit={mean('new_hit'):.2f} old_residual={mean('old_residual'):.2f}")
    logger.info(f"CONSISTENCY edit: chroma={mean('chroma_edit'):.3f} cqt_pcc={mean('cqt_pcc_edit'):.3f}")
    logger.info(f"REGEN CTRL: chroma={mean('chroma_regen'):.3f} cqt_pcc={mean('cqt_pcc_regen'):.3f} "
                f"(lower chroma/cqt than edit proves flow-edit preserves structure)")


if __name__ == "__main__":
    main()

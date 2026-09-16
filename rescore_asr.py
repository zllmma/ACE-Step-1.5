#!/usr/bin/env python3
"""Re-score the lyric-edit benchmark with Qwen3-ASR (singing-trained ASR).

Re-transcribes the already-generated edit/regen audio from
lyric_edit_eval/report.csv with Qwen3-ASR-1.7B (the same ASR family used by
LyricEditBench, singing-voice trained), recomputes the keyword-following
metrics, and appends them as new columns without touching whisper columns.

Run inside an isolated env (needs transformers>=5.13):
    uv run --with "transformers>=5.13.0" python rescore_asr.py
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent

PUNCT = re.compile(r"[，。！？；：、,.!?;:\s\"'“”‘’\[\]()（）<>《》\-—…]")


def normalize(text: str) -> str:
    """Strip punctuation/whitespace for tolerant keyword matching."""
    return PUNCT.sub("", text)

ASR_MODEL = "Qwen/Qwen3-ASR-1.7B-hf"
REPORT = REPO / "lyric_edit_eval" / "report.csv"
CONDITIONS = REPO / "lyric_edit_conditions.json"


def keyword_stats(transcript: str, new_keys: list[str], old_keys: list[str]):
    """Substring containment on normalized text (same protocol as whisper pass)."""
    t = normalize(transcript)
    new_hit = sum(1 for k in new_keys if normalize(k) in t)
    old_res = sum(1 for k in old_keys if normalize(k) in t)
    return new_hit / max(1, len(new_keys)), old_res / max(1, len(old_keys))


def clean_decode(text: str) -> str:
    """Strip chat scaffolding around the <asr_text> payload."""
    if "<asr_text>" in text:
        text = text.split("<asr_text>")[-1]
    for marker in ("</asr_text>", "<|im_end|>", "<|endoftext|>"):
        text = text.replace(marker, "")
    return text.strip()


def main() -> None:
    """Re-transcribe report rows with Qwen3-ASR and update the CSV."""
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    rows = list(csv.DictReader(open(REPORT)))
    conds = {int(c["index"]): c for c in json.load(open(CONDITIONS))}

    processor = AutoProcessor.from_pretrained(ASR_MODEL)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        ASR_MODEL, dtype=torch.bfloat16, device_map="cuda"
    )

    def transcribe(wav_path: str) -> str:
        """Transcribe one full song via chat-template + audio input."""
        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": wav_path},
            {"type": "text", "text": "Transcribe the singing voice in Chinese."},
        ]}]
        text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], audio=[wav_path], return_tensors="pt").to(model.device, model.dtype)
        out = model.generate(**inputs, max_new_tokens=4000)
        return clean_decode(processor.batch_decode(out, skip_special_tokens=True)[0])

    t0 = time.time()
    for i, r in enumerate(rows, 1):
        cond = conds[int(r["index"])]
        tr_edit = transcribe(r["edit_wav"])
        tr_regen = transcribe(r["regen_wav"])
        new_hit, old_res = keyword_stats(tr_edit, cond["eval_new_keywords"], cond["eval_old_keywords"])
        new_hit_r, _ = keyword_stats(tr_regen, cond["eval_new_keywords"], cond["eval_old_keywords"])
        r["new_hit_qwen"] = f"{new_hit:.2f}"
        r["old_residual_qwen"] = f"{old_res:.2f}"
        r["follow_asb_qwen"] = f"{new_hit * (1 - old_res):.3f}"
        r["new_hit_regen_qwen"] = f"{new_hit_r:.2f}"
        r["transcript_qwen"] = tr_edit[:300]
        logger_mins = (time.time() - t0) / 60
        print(f"[{i}/{len(rows)}] new_hit_qwen={new_hit:.2f} old_res={old_res:.2f} "
              f"({logger_mins:.1f}min) tr[:100]={tr_edit[:100]}", flush=True)

    fields = list(rows[0].keys())
    with open(REPORT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    def mean(key):
        vals = [float(r[key]) for r in rows if r[key] not in ("", None)]
        return sum(vals) / len(vals) if vals else float("nan")

    print(f"UPDATED {REPORT}")
    print(f"QWEN  new_hit={mean('new_hit_qwen'):.2f} old_residual={mean('old_residual_qwen'):.2f} "
          f"follow_asb={mean('follow_asb_qwen'):.3f}")
    print(f"WHISPER(old) new_hit={mean('new_hit'):.2f} old_residual={mean('old_residual'):.2f}")
    print(f"REGEN-CTRL new_hit={mean('new_hit_regen_qwen'):.2f}")


if __name__ == "__main__":
    main()

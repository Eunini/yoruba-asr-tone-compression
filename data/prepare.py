"""Audit the actual WAXAL release and materialize reproducible audio manifests."""

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf
from math import gcd
from scipy.signal import resample_poly
from datasets import Audio, load_dataset
from huggingface_hub import HfApi

DATASET = "google/WaxalNLP"
REVISION = "5f4d8ca24f2b9d168b2ee545f1febaaff4b40580"


def prepare(out: Path, splits: list[str], limit: int | None, revision: str = REVISION):
    info = HfApi().dataset_info(DATASET, revision=revision)
    configs = [c["config_name"] for c in info.card_data.to_dict()["configs"]]
    if "yor_tts" not in configs:
        raise RuntimeError("Reviewed Yoruba TTS configuration disappeared")
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "dataset": DATASET,
        "revision": info.sha,
        "config": "yor_tts",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "yoruba_asr_available": "yor_asr" in configs,
        "task": "TTS recordings repurposed for ASR; not an official Yoruba ASR subset",
        "license": "CC-BY-SA-4.0 (Media Trust subset; verify upstream before redistribution)",
        "partial": limit is not None,
        "splits": {},
    }
    texts, speakers = {}, {}
    for split in splits:
        rows = load_dataset(DATASET, "yor_tts", revision=revision, split=split, streaming=True)
        rows = rows.cast_column("audio", Audio(decode=False))
        seconds, n, marked = 0.0, 0, 0
        texts[split], speakers[split] = set(), set()
        audio_dir = out / split
        audio_dir.mkdir(exist_ok=True)
        with (out / f"{split}.jsonl").open("w") as stream:
            for i, row in enumerate(rows):
                if limit is not None and i >= limit:
                    break
                audio = row["audio"]
                if not audio.get("bytes"):
                    raise ValueError("Expected embedded audio bytes, refusing an unreviewed path")
                signal, rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
                if signal.ndim == 2:
                    signal = signal.mean(axis=1)
                if rate != 16000:
                    divisor = gcd(rate, 16000)
                    signal = resample_poly(signal, 16000 // divisor, rate // divisor)
                    rate = 16000
                duration = len(signal) / rate
                path = audio_dir / f"{i:06d}.wav"
                sf.write(path, signal, rate, subtype="PCM_16")
                reference = row["text"]
                record = {
                    "id": row["id"],
                    "speaker_id": row["speaker_id"],
                    "audio": str(path.resolve()),
                    "reference": reference,
                    "duration_seconds": duration,
                    "sample_rate": rate,
                    "source_revision": revision,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                texts[split].add(hashlib.sha256(reference.encode()).hexdigest())
                speakers[split].add(row["speaker_id"])
                import unicodedata

                marked += any(c in "\u0300\u0301" for c in unicodedata.normalize("NFD", reference))
                seconds += duration
                n += 1
        audit["splits"][split] = {
            "rows": n,
            "hours": seconds / 3600,
            "seconds": seconds,
            "unique_speakers": len(speakers[split]),
            "tone_marked_rows": marked,
        }
        print(json.dumps({split: audit["splits"][split]}), flush=True)
    audit["overlap"] = {}
    for i, a in enumerate(splits):
        for b in splits[i + 1 :]:
            audit["overlap"][f"{a}:{b}"] = {
                "exact_transcripts": len(texts[a] & texts[b]),
                "speakers": len(speakers[a] & speakers[b]),
            }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    # Clean only training data. Keep the published validation and test sets intact.
    if "train" in splits and "validation" in splits and "test" in splits:
        held_out = texts["validation"] | texts["test"]
        path = out / "train.jsonl"
        training = [json.loads(line) for line in path.read_text().splitlines()]
        retained = [
            r
            for r in training
            if hashlib.sha256(r["reference"].encode()).hexdigest() not in held_out
        ]
        audit["training_deduplication"] = {
            "original": len(training),
            "retained": len(retained),
            "removed": len(training) - len(retained),
        }
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in retained))
        (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/prepared"))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    prepare(args.out, args.splits, args.limit)

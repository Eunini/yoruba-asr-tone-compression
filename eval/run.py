"""One manifest, four inference backends, identical metrics and CPU timing."""

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import score
from eval.artifacts import fingerprint


def backend(args):
    if args.backend == "ct2":
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.threads,
            download_root=args.model_cache,
        )

        def transcribe(path):
            segments, _ = model.transcribe(
                path,
                language="yo",
                task="transcribe",
                beam_size=1,
                temperature=0,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            return " ".join(s.text.strip() for s in segments)

        return transcribe
    if args.backend in {"whisper", "mms"}:
        import torch
        from transformers import (
            AutoProcessor,
            Wav2Vec2ForCTC,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        if args.backend == "mms":
            processor = AutoProcessor.from_pretrained(args.model)
            processor.tokenizer.set_target_lang("yor")
            model = Wav2Vec2ForCTC.from_pretrained(
                args.model, target_lang="yor", ignore_mismatched_sizes=True
            )
        else:
            processor = WhisperProcessor.from_pretrained(
                args.model, language="yo", task="transcribe"
            )
            model = WhisperForConditionalGeneration.from_pretrained(args.model)
            model.generation_config.language = "yo"
            model.generation_config.task = "transcribe"
            model.generation_config.forced_decoder_ids = None
        model.to(args.device).eval()

        def transcribe(path):
            signal, sr = sf.read(path, dtype="float32")
            if signal.ndim != 1:
                signal = signal.mean(axis=1)
            if sr != 16000:
                divisor = gcd(sr, 16000)
                signal = resample_poly(signal, 16000 // divisor, sr // divisor)
            chunks = (
                [signal[i : i + 480000] for i in range(0, len(signal), 480000)]
                if args.backend in {"whisper", "mms"}
                else [signal]
            )
            texts = []
            for chunk in chunks:
                inputs = processor(chunk, sampling_rate=16000, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    if args.backend == "mms":
                        tokens = model(**inputs).logits.argmax(-1)
                    else:
                        tokens = model.generate(**inputs, max_length=448, do_sample=False)
                texts.append(processor.batch_decode(tokens, skip_special_tokens=True)[0])
            return " ".join(texts)

        return transcribe
    if args.backend == "cpp":

        def transcribe(path):
            with tempfile.TemporaryDirectory() as tmp:
                prefix = str(Path(tmp) / "transcript")
                subprocess.run(
                    [
                        args.cpp_binary,
                        "-m",
                        args.model,
                        "-f",
                        path,
                        "-l",
                        "yo",
                        "-t",
                        str(args.threads),
                        "-bs",
                        "1",
                        "-bo",
                        "1",
                        "-otxt",
                        "-of",
                        prefix,
                        "-nt",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return Path(prefix + ".txt").read_text().strip()

        return transcribe
    raise ValueError(args.backend)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", choices=["ct2", "whisper", "mms", "cpp"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-cache", default="models")
    parser.add_argument("--cpp-binary", default="whisper-cli")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--server-hourly-usd", type=float)
    parser.add_argument("--price-source")
    args = parser.parse_args()
    if args.server_hourly_usd is not None and (args.server_hourly_usd < 0 or not args.price_source):
        parser.error("A nonnegative hourly price requires a public --price-source URL")
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        parser.error("Manifest is empty")
    start = time.perf_counter()
    transcribe = backend(args)
    load_seconds = time.perf_counter() - start
    transcribe(rows[0]["audio"])  # Fixed warmup excluded from steady-state timing.
    args.out.mkdir(parents=True, exist_ok=True)
    predictions = []
    with (args.out / "predictions.jsonl").open("w") as stream:
        for row in rows:
            start = time.perf_counter()
            hypothesis = transcribe(row["audio"])
            predictions.append(
                {
                    "id": row["id"],
                    "reference": row["reference"],
                    "hypothesis": hypothesis,
                    "duration_seconds": row["duration_seconds"],
                    "processing_seconds": time.perf_counter() - start,
                }
            )
            stream.write(json.dumps(predictions[-1], ensure_ascii=False) + "\n")
            stream.flush()
    seconds = sum(r["processing_seconds"] for r in predictions)
    duration = sum(r["duration_seconds"] for r in predictions)
    result = {
        "status": "measured",
        "artifacts": fingerprint(args.model),
        "model": args.model,
        "backend": args.backend,
        "compute_type": args.compute_type if args.backend == "ct2" else "backend default",
        "device": args.device,
        "threads": args.threads,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "partial": args.limit is not None,
        "load_seconds": load_seconds,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "evaluation_set_sha256": hashlib.sha256(
            json.dumps(
                [{k: r[k] for k in ["id", "reference", "duration_seconds"]} for r in rows],
                ensure_ascii=False,
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "metric_policy": "explicit-tone-sequence edits conditional on matching words; v1",
        "segmentation": (
            "consecutive nonoverlapping 30s chunks"
            if args.backend in {"whisper", "mms"}
            else "backend-native long-audio segmentation"
        ),
        "processing_seconds": seconds,
        "audio_seconds": duration,
        "real_time_factor": seconds / duration,
        "cost_per_audio_hour_usd": (
            seconds / duration * args.server_hourly_usd
            if args.server_hourly_usd is not None
            else None
        ),
        "server_hourly_usd": args.server_hourly_usd,
        "price_source": args.price_source,
        **score([r["reference"] for r in predictions], [r["hypothesis"] for r in predictions]),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

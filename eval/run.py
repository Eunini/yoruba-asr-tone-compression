"""One manifest, four inference backends, identical metrics and CPU timing."""

import argparse
import hashlib
import json
import platform
import contextlib
import subprocess
import tempfile
import time
import os
import importlib.metadata
import fcntl
from datetime import datetime, timezone
from pathlib import Path

from eval.metrics import score
from eval.artifacts import fingerprint
from eval.audio import chunks
from eval.journal import prepare
from train.model_revisions import REVISIONS


def backend(args):
    if args.backend == "ct2":
        from faster_whisper import WhisperModel

        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.threads,
            download_root=args.model_cache,
            revision=args.revision,
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
            processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
            processor.tokenizer.set_target_lang("yor")
            model = Wav2Vec2ForCTC.from_pretrained(
                args.model, ignore_mismatched_sizes=True, revision=args.revision
            )
            model.load_adapter("yor", revision=args.revision)
        else:
            processor = WhisperProcessor.from_pretrained(
                args.model, language="yo", task="transcribe", revision=args.revision
            )
            model = WhisperForConditionalGeneration.from_pretrained(
                args.model, revision=args.revision
            )
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
            texts = []
            for chunk in chunks(signal):
                if args.backend == "mms" and len(chunk) < 400:
                    # Wav2Vec2's convolutional receptive field needs 400 samples.
                    import numpy as np

                    chunk = np.pad(chunk, (0, 400 - len(chunk)))
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
        if args.cpp_server_url:
            import httpx

            # Start whisper-server once with this exact --model and --threads.
            # Multipart form fields match the pinned upstream server interface.
            client = httpx.Client(timeout=600)

            def transcribe(path):
                with open(path, "rb") as audio:
                    response = client.post(
                        args.cpp_server_url.rstrip("/") + "/inference",
                        files={
                            "file": (Path(path).name, audio, "audio/wav"),
                            **{
                                k: (None, v)
                                for k, v in {
                                    "language": "yo",
                                    "response_format": "json",
                                    "beam_size": "1",
                                    "best_of": "1",
                                    "temperature": "0",
                                    "temperature_inc": "0",
                                    "vad": "false",
                                    "max_context": "0",
                                    "no_timestamps": "true",
                                }.items()
                            },
                        },
                    )
                response.raise_for_status()
                return response.json()["text"].strip()

            transcribe.close = client.close
            return transcribe

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
    parser.add_argument(
        "--revision", help="Pinned public weights commit; known study models have defaults"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model-cache", default="models")
    parser.add_argument("--cpp-binary", default="whisper-cli")
    parser.add_argument(
        "--cpp-server-url", help="Persistent whisper-server root, e.g. http://127.0.0.1:8080"
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--resume", action="store_true", help="Resume only an identical recorded run"
    )
    parser.add_argument("--server-hourly-usd", type=float)
    parser.add_argument("--price-source")
    args = parser.parse_args()
    args.revision = args.revision or REVISIONS.get(args.model)
    if args.server_hourly_usd is not None and (args.server_hourly_usd < 0 or not args.price_source):
        parser.error("A nonnegative hourly price requires a public --price-source URL")
    if args.backend == "cpp" and not args.cpp_server_url and args.server_hourly_usd is not None:
        parser.error(
            "Steady-state cost requires --cpp-server-url; whisper-cli reloads weights per file"
        )
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        parser.error("Manifest is empty")
    artifacts = fingerprint(args.model)
    source_files = [
        Path(__file__),
        Path(__file__).with_name("metrics.py"),
        Path(__file__).with_name("audio.py"),
        Path(__file__).with_name("journal.py"),
    ]
    versions = {}
    for package in ("jiwer", "faster-whisper", "ctranslate2", "transformers", "torch", "httpx"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    identity = {
        "model": args.model,
        "artifacts": artifacts,
        "revision": args.revision,
        "backend": args.backend,
        "device": args.device,
        "compute_type": args.compute_type,
        "threads": args.threads,
        "cpp_server_url": args.cpp_server_url,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "limit": args.limit,
        "server_hourly_usd": args.server_hourly_usd,
        "price_source": args.price_source,
        "platform": platform.platform(),
        "packages": versions,
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    }
    identity["audio_sha256"] = {}
    for row in rows:
        with open(row["audio"], "rb") as stream:
            identity["audio_sha256"][row["id"]] = hashlib.file_digest(stream, "sha256").hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out / ".run.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    predictions = prepare(args.out, identity, rows, args.resume)
    if len(predictions) == len(rows) and (args.out / "summary.json").exists():
        print("Identical completed run already saved; no inference repeated")
        return
    start = time.perf_counter()
    transcribe = backend(args)
    load_seconds = time.perf_counter() - start
    with contextlib.ExitStack() as stack:
        if hasattr(transcribe, "close"):
            stack.callback(transcribe.close)
        transcribe(rows[0]["audio"])  # Same fixed warmup after every model load.
        with (args.out / "load-events.jsonl").open("a") as events:
            events.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "load_seconds": load_seconds,
                        "existing_predictions": len(predictions),
                    }
                )
                + "\n"
            )
        stream = stack.enter_context((args.out / "predictions.jsonl").open("a"))
        for row in rows[len(predictions) :]:
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
            os.fsync(stream.fileno())
    seconds = sum(r["processing_seconds"] for r in predictions)
    duration = sum(r["duration_seconds"] for r in predictions)
    result = {
        "status": "measured",
        "artifacts": artifacts,
        "model": args.model,
        "source_revision": args.revision,
        "backend": args.backend,
        "compute_type": args.compute_type if args.backend == "ct2" else "backend default",
        "device": args.device,
        "threads": args.threads,
        "platform": platform.platform(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "partial": args.limit is not None,
        "load_seconds": load_seconds,
        "load_timing_scope": (
            "HTTP client initialization; persistent whisper.cpp model was loaded before this run"
            if args.backend == "cpp" and args.cpp_server_url
            else "backend initialization and model loading"
        ),
        "load_events_file": "load-events.jsonl",
        "timing_scope": (
            "process startup and model load per utterance; not steady-state"
            if args.backend == "cpp" and not args.cpp_server_url
            else "warm persistent model; excludes initial load and one warmup"
        ),
        "cpp_server_url": args.cpp_server_url,
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
            "nonoverlapping <=30s chunks; final tail >=1s for split audio; MMS pads original <400-sample inputs"
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
    temporary = args.out / "summary.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out / "summary.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

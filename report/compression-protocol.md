# Converted-model CPU protocol

The three validation-selected checkpoints retain their original artifact hashes.
Measure five converted conditions per size: CT2 float32, CT2 int8, whisper.cpp
f16 weights, q5_0 and q4_0. The three existing Hugging Face GPU evaluations remain
separate. This yields 12 primary model representations and six backend controls;
15 converted conditions need new CPU inference. F16 describes whisper.cpp weight
storage, not a claim that all CPU arithmetic uses fp16.

Every condition uses all 177 original test recordings, seed-independent greedy
decoding, Yoruba language selection, no VAD, and the documented native backend
segmentation. CT2 has temperature zero and no previous-text conditioning;
whisper.cpp uses a persistent CPU server, no fallback, beam/best-of one and zero
text context. Models load once per session; a fixed first-recording warmup is
excluded from steady-state time. Server startup and client initialization have
separate timing receipts. No test recording is removed after seeing predictions.

`eval.compression` verifies the conversion receipts before starting and evaluates
one condition at a time with two CPU threads. `eval.run --resume` requires matching
model/configuration hashes, audio-file hashes, source hashes, package versions,
manifest, decoding settings, platform and price evidence. Completed predictions
remain byte-for-byte unchanged. Corrupt tails or different inputs cause an error
instead of automatic repair. A directory lock prevents concurrent writers.

For each quantized condition, compare its own backend float control on the
intersection of correctly aligned reference-word positions. Report common-word
and common-tone coverage, both common-support TERs and their difference, plus
5,000 paired utterance bootstrap resamples (seed 2026, percentile 95% interval).
Report ordinary WER and marginal TER alongside these quantities. Undefined tone
denominators remain null. Cross-backend differences do not identify a pure
quantization effect. Bootstrap intervals measure utterance variation, not runtime
repeatability or population-wide Yoruba performance.

## Dedicated AWS run

`serve/infra` creates one private-profile EC2 `m7i-flex.large` in us-east-1,
with 30 GB encrypted gp3, one public IPv4, IMDSv2, and SSH restricted to the
controller's /32. Models and audio are transferred over SSH; no AWS, Hugging Face
or API credentials are sent to the machine. There is no public inference endpoint.

The controller accepts freshly checked public AWS list prices, a positive runtime
of at most 48 hours, and a compute/storage/IPv4 budget. Its default ceiling is $6;
this is independent of the owner's reported credits and excludes tax and traffic.
The dated price evidence includes the source offer URLs and component prices.
RTF is inference wall time divided by actual audio duration; USD/audio hour is
RTF multiplied by the recorded server hourly list price. Flex performance is
variable, so this is one operating point, not a guaranteed service latency.

A verified on-instance timer terminates the instance at its finite lifetime even
if the controller disappears. The controller copies results every minute, then
destroys its exclusive Terraform state in a finally block on success or failure.
Root disks delete on termination. Preserve the controller's actual teardown
receipt; do not claim cleanup merely because a timer was configured. State,
provisioning logs, SSH host keys and local account files remain outside git.

```bash
# Refresh Project 3's unauthenticated AWS offer-file price snapshot first.
python -m serve.cloud_benchmark --account-id YOUR_INTENDED_ACCOUNT \
  --converted /external/verified/converted-v6 \
  --manifest data/prepared/test.jsonl --prices report/aws-public-prices.json \
  --terraform /absolute/path/to/terraform --max-hours 48 --budget-usd 6 \
  --out report/runs/compression-aws
```

These are execution methods. Only completed summaries establish measurements.
Five-recording engineering checks in `conversion-validation.json` validate model
loading and transcription; they are excluded from final scientific comparisons.

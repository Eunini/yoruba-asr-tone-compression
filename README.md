# Do tones break first when a Yorùbá speech model shrinks?

**Measured so far:** the complete WAXAL Yoruba data audit and a pretrained tiny
CPU baseline. The training and compression study is still in progress; there is
no supported claim yet that compression causes tone loss.

WAXAL contains **`yor_tts`, not `yor_asr`** at the pinned revision. These are
recordings collected for TTS and repurposed here for ASR. The 2026-10-25 audit
decoded every recording with the `datasets` library:

| Published split | Recordings | Measured audio hours | Speakers |
| --- | ---: | ---: | ---: |
| Train | 2,233 | 19.169 | 8 |
| Validation | 201 | 1.785 | 8 |
| Test | 177 | 1.562 | 8 |

Every speaker appears in all splits. There are four exact transcript overlaps
between training and test, and five between training and validation. Preparation
removes held-out transcript duplicates from the training manifest. This experiment
does **not** measure generalization to unseen speakers.

| Model | Precision | Test recordings | WER | Conditional TER | Matched words | Local RTF | Cloud cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| faster-whisper-tiny | float32 | 177 | 123.97% | 100.00% | 1.85% | 0.153 | Not measured |
| faster-whisper-tiny | int8 | 177 | 122.16% | 100.00% | 2.57% | 0.063 | Not measured |

![Cost versus tone error](report/results/cost-tone.png)

The measurement table is in [the report](report/report.md); machine-readable
observations are in [report/runs](report/runs). `null` costs mean no cloud price or
controlled cloud measurement has been supplied. They do not mean free inference.

```bash
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,inference]'
pytest -q
python -m data.prepare                    # Allow about 4 GB including 16-kHz audio
python -m eval.run --manifest data/prepared/test.jsonl --backend ct2 \
  --model Systran/faster-whisper-tiny --compute-type int8 --out report/runs/tiny-int8
# Exact Hugging Face baselines need the training dependencies:
pip install -e '.[training]'
python -m eval.run --manifest data/prepared/test.jsonl --backend whisper \
  --model openai/whisper-small --out report/runs/whisper-small
python -m eval.run --manifest data/prepared/test.jsonl --backend mms \
  --model facebook/mms-1b-all --out report/runs/mms-yor
```

`yo` is Whisper's Yoruba code; MMS uses the `yor` adapter. Long audio is decoded
with the backend's documented segmentation policy; do not compare runs with
different policies as if quantization were their only difference.

On a Kaggle GPU, with Internet enabled and this repository checked out:

```bash
pip install -e '.[training,inference]'
python -m data.prepare
for size in tiny base small; do
  python -m train.finetune --size "$size" --out "checkpoints/$size"
  bash scripts/compress.sh "checkpoints/$size/best" "models/$size" /path/to/whisper.cpp /path/to/whisper
done
```

Checkpoints are selected by validation WER. Utterances longer than 30 seconds are
excluded as whole examples during training and validation checkpoint selection;
the counts are saved. The complete test set remains intact. Each size has four
conditions—float, int8, q5_0 and q4_0—so the complete matrix has **12 versions**.
Record the exact source commits of both converter repositories.

```bash
python -m eval.run --manifest data/prepared/test.jsonl --backend cpp \
  --model models/tiny/ggml-model-q4_0.bin --cpp-binary /path/to/whisper-cli \
  --out report/runs/tiny-q4
python -m eval.error_review report/runs/tiny-q4/predictions.jsonl
python -m eval.report report/runs/*/summary.json
```

For CPU cloud runs, supply `--server-hourly-usd` and a public `--price-source` URL.
Cost per audio hour is `real_time_factor × hourly_server_price`. Keep hardware,
thread count, decoding settings, sample set and warmup fixed across comparisons.
The local shared-host timings are exploratory and cannot establish cloud costs.

WER removes only U+0300 and U+0301 after Unicode NFD; underdots in ẹ, ọ and ṣ
survive. TER compares explicit-tone sequences **only inside words that match
after tone removal**, normalized by explicit reference marks in those words.
It is always accompanied by matched-word and matched-tone coverage. The optional
`reference_normalized_tone_edit_rate` uses all reference tones as its denominator;
that quantity can become misleadingly small when almost no words match.

Unmarked mid tones, tone positions within the same mark sequence, punctuation,
and incomplete reference diacritics remain limitations. A vowel-aligned tone
metric and native-speaker review are necessary before strong linguistic claims.

Read [background](report/background.md), [methods and results](report/report.md),
and [remaining work](report/completion.md). Dataset/model licenses are separate
from the MIT code license. No trained model has been uploaded yet.

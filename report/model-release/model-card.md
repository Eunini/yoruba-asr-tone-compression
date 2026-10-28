---
language: [yo]
license: cc-by-sa-4.0
base_model: openai/whisper-tiny
base_model_relation: finetune
datasets: [google/WaxalNLP]
pipeline_tag: automatic-speech-recognition
library_name: transformers
tags: [whisper, yoruba, tone, research]
---
# Yoruba WAXAL Whisper-tiny — measured research checkpoint

**Research result: 88.76% tone-stripped word error rate on the complete 177-clip
WAXAL Yoruba test split. This checkpoint is not validated as a production
recognizer.** The complete compression/cost study remains in progress.

This release makes a validation-selected fine-tuned checkpoint reproducible,
including its weak result. It does not outperform the study's MMS Yoruba baseline
(58.41% WER). The model is intended for studying tone preservation, decoding and
compression effects; successful loading or transcription is not evidence of
accurate Yoruba recognition.

## Training and selection

Base: `openai/whisper-tiny` at
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Data: `google/WaxalNLP`, config
`yor_tts`, revision `5f4d8ca24f2b9d168b2ee545f1febaaff4b40580`. This is a
**TTS corpus repurposed for ASR**, not an official WAXAL Yoruba ASR subset.

The original splits contain 2,233 training clips (19.169 h), 201 validation clips
(1.785 h) and 177 test clips (1.562 h). Removing training transcript overlaps leaves
2,216 training clips. Whole-utterance duration/context filtering retains 1,691
training and 156 validation clips; long audio is not cropped under a full label.
All 177 original test recordings remain in the final evaluation.

Kaggle v6 training used one visible T4, fp16 mixed precision, seed 2026,
learning rate 1e-5, batch size four with four-step gradient accumulation,
100 warmup steps and 2,000 total optimizer steps. Validation and checkpoint
saving ran every 100 steps. The **step-800 checkpoint** had the lowest held-out
validation WER (61.0785%); the test split did not select it. Non-reentrant
gradient checkpointing was enabled. See `selection.json` for exact hashes,
filter counts and recorded training history.

## Measured full-test result

| Quantity | Value |
| --- | ---: |
| Test recordings | 177 |
| Audio duration | 5,622.144 s |
| Tone-stripped WER | 88.7646% |
| Conditional explicit-tone sequence error rate | 37.1726% |
| Matched-word coverage | 31.6960% |
| Matched-reference-tone coverage | 27.4512% |

WER removes only Unicode combining grave and acute marks after NFD; Yoruba
underdots remain. Tone error counts edits in explicit tone sequences only for
aligned, tone-stripped matching words. Unmatched words are excluded from its
denominator, so **37.17% is not an all-transcript tone-accuracy score**. Read it
together with WER and low coverage. Unmarked mid tones are not directly scored.

`evaluation.json` records the selected-weight hashes, complete test identity,
decoding/evaluation scope and counts. Test identity:
`2e71d67cf2d5de5c1953bc646ca1550887b31de11f2b6370e82beacaf9cc8529`.
The source model's GPU timing is not a CPU serving cost result. Converted-model
measurements and paired-support analysis belong to the separate study.

## Using the checkpoint

Install PyTorch, Transformers 4.57.6 and the audio dependencies used by its ASR
pipeline, including FFmpeg for file decoding. A short-clip demonstration is:

```python
from transformers import pipeline

asr = pipeline(
    "automatic-speech-recognition",
    model="Ineeoluwa/whisper-tiny-yoruba-waxal-research",
    device=-1,
)
result = asr("example.wav", generate_kwargs={"language": "yo", "task": "transcribe"})
print(result["text"])
```

This example does not reproduce the whole benchmark. Use the versioned source
and recorded dataset/evaluation settings for that:
https://github.com/Eunini/yoruba-asr-tone-compression

Weights use safetensors. Optimizer state and pickle-based `training_args.bin`
are excluded from this release. `release-manifest.json` identifies included
artifacts, their SHA256 values and the publication source revision.

## Limitations and intended use

Eight speakers overlap across source splits. These results do not demonstrate
unseen-speaker, dialect, noisy-conversation or real-user generalization. The
small scripted TTS corpus and exclusion of long training clips further limit
interpretation. Low matched-tone coverage can make conditional TER misleading.
Whisper may hallucinate text or omit diacritics. Human review of the speech
errors and the full compression/cost study remain pending.

Use for reproducible research and error investigation. Medical, legal, financial,
access-control and other consequential uses have not been validated. Avoid
presenting this checkpoint as a reliable transcription service.

## Attribution and licenses

Credit the original OpenAI Whisper authors and Google/partners' WAXAL
contributors, especially Media Trust for this Yoruba TTS subset. The pinned
dataset card identifies CC-BY-SA-4.0 for the Media Trust data. Fine-tuning
contributions and this card are released under CC BY-SA 4.0, with upstream
Apache-2.0 and MIT notices retained separately. This does not relabel the original
base weights or imply endorsement. Read `NOTICE.md` and all included license
files. Raw WAXAL recordings/transcripts are not redistributed here.

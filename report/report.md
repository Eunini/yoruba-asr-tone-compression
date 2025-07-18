# Do tones break first when a Yorùbá speech model shrinks?

Status: preliminary measurements; causal compression results pending.

## Question

Does reducing model size or numerical precision disproportionately damage explicit
Yoruba tone marks, and what is the resulting accuracy–serving-cost trade-off?
Model size, fine-tuning and quantization are separate experimental factors.

## Data

`google/WaxalNLP`, `yor_tts`, revision
`5f4d8ca24f2b9d168b2ee545f1febaaff4b40580`. The measured original splits contain
2,233 / 201 / 177 recordings and 19.169 / 1.785 / 1.562 hours. The audit finds
eight shared speakers in every split and exact transcript overlaps. Training
preparation removes held-out duplicates; the original test set is preserved.
Only 175 of 177 test references contain any explicit acute/grave tone mark;
having some marks does not establish complete tone annotation.

## Method

Use corpus WER after stripping acute and grave combining marks, retaining
underdots. Align words using `jiwer`, then compare the tone-mark sequences in
matching pairs using Levenshtein distance. Normalize TER by reference marks in
those matching words and report coverage. Rates can exceed 100% through
insertions. Undefined denominators yield `null`.

Select each fine-tuned checkpoint by held-out validation WER, never test scores.
For training, remove entire utterances exceeding Whisper's 30-second window
rather than cropping audio against unmodified full labels. Report exclusion
counts and retain a full-test long-audio evaluation. Evaluate float, int8,
q5_0 and q4_0 for tiny, base and small using documented backend policies.

## Preliminary results

The generated table below is updated from completed `summary.json` files. These
pretrained measurements are not fine-tuned model results. They cannot answer the
compression question on their own.

<!-- RESULTS -->
| Model | Precision | Test recordings | WER | Conditional TER | Matched words | Local RTF | Cloud cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| faster-whisper-small | float32 | 177 | 110.76% | 100.00% | 1.01% | 0.710 | Not measured |
| faster-whisper-tiny | float32 | 177 | 123.97% | 100.00% | 1.85% | 0.153 | Not measured |
| faster-whisper-tiny | int8 | 177 | 122.16% | 100.00% | 2.57% | 0.063 | Not measured |
<!-- /RESULTS -->

## Cost chart

![Cost versus tone error](results/cost-tone.png)

Points require a measured CPU run and a public hourly server price. Missing cost
evidence produces an explicitly empty chart rather than fabricated points.
Local timings include inference and transcript materialization after one warmup,
exclude model loading, and were obtained on a shared host. They are exploratory.

## Error analysis

`python -m eval.error_review ...` creates a seeded sample of 50 utterances with
columns for dropped tones, wrong tones, code-switching and wrong words. Multiple
types may apply to one utterance; report both utterance incidence and the chosen
event-count policy. The native/manual coding step is pending. Automatically
identifying an explicit-tone deletion is not equivalent to judging lexical
meaning or language mixing.

## Limitations

This is repurposed TTS speech, not a verified spontaneous Yoruba ASR subset.
Speakers overlap across splits. References may omit marks. Conditional TER omits
unmatched words and therefore must be read with WER and coverage. Tone sequences
ignore mark position when two positions produce the same sequence, and unmarked
mid tones are not scored. Tokenization, segmentation and decoding differences can
confound backend comparisons. Full-model, quantized-model and cross-domain runs
remain necessary. No cloud cost or best fine-tuned model is claimed.

## Conclusion

The data audit and metric implementation are complete. The final answer requires
the full model matrix, GPU fine-tuning, controlled CPU serving measurements,
native error review and a cross-domain evaluation. CV percentages remain blank
until that evidence exists.

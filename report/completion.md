# Completion record

- Completed: paper verification, full source-data audit, split hours, speaker and
  transcript overlap checks, tone-aware metrics and tests, reproducible data,
  inference, training, conversion, reporting and manual-review code.
- Local CPU baseline results are included only when their run has completed.
- Completed: pinned full-test Hugging Face Whisper-small and MMS Yoruba baselines.
- Pending: Kaggle training
  for three sizes, checkpoint selection, all conversions, controlled cloud cost
  measurement, manual/native review of 50 outputs, and model publication.
- Future work: a speaker-independent corpus such as FLEURS or YORÙLECT as a distinct
  generalization test. Do not silently substitute it for the published WAXAL test.
- Kaggle phone verification and GPU/internet access are resolved. The training
  protocol now uses one visible GPU with non-reentrant gradient checkpointing;
  the corrected run is advancing. Earlier failed stages remain recorded.

The scientific project is not complete merely because its code passes tests.

# Completion record

- Completed: paper verification, full source-data audit, split hours, speaker and
  transcript overlap checks, tone-aware metrics and tests, reproducible data,
  inference, training, conversion, reporting and manual-review code.
- Local CPU baseline results are included only when their run has completed.
- Completed: pinned full-test Hugging Face Whisper-small and MMS Yoruba baselines.
- Completed: all three 2,000-step Kaggle training runs, validation-only checkpoint
  selection and full 177-recording evaluations; selected source weights privately
  persisted and converted to CT2 int8/float32 and whisper.cpp f16/q5_0/q4_0.
- Pending: converted-model inference validation and scoring, paired common-word
  tone analysis, controlled CPU cost measurement, manual/native review of 50
  outputs, final chart/report and model publication.
- Update: CT2 int8 and whisper.cpp q4 tiny passed five-recording real-audio
  engineering checks. The resumable 15-condition CPU runner and paired-support
  analysis are implemented and tested; full CPU results remain pending. The
  dedicated AWS controller includes price/runtime guards, periodic result sync,
  a verified termination timer and Terraform teardown. See compression-protocol.md.
- Future work: a speaker-independent corpus such as FLEURS or YORÙLECT as a distinct
  generalization test. Do not silently substitute it for the published WAXAL test.
- Kaggle phone verification and GPU/internet access are resolved. The training
  protocol now uses one visible GPU with non-reentrant gradient checkpointing;
  the corrected version 6 completed successfully. Earlier failed stages remain recorded.

The scientific project is not complete merely because its code passes tests.

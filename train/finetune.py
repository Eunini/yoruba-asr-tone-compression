"""Run on a Kaggle GPU; validation selects checkpoints, test is never loaded."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from eval.metrics import score
from eval.artifacts import fingerprint
from train.model_revisions import REVISIONS


@dataclass
class Collator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        import torch

        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt"
        )
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt"
        )
        ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if torch.all(ids[:, 0] == self.decoder_start_token_id):
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def main():
    import torch
    from datasets import Audio, Dataset
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
        set_seed,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=["tiny", "base", "small"], required=True)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required for the planned fp16 experiment")
    set_seed(args.seed)
    model_id = f"openai/whisper-{args.size}"
    revision = REVISIONS[model_id]
    processor = WhisperProcessor.from_pretrained(
        model_id, language="yo", task="transcribe", revision=revision
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id, revision=revision)
    model.generation_config.language = "yo"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    exclusions = {}

    def load(split):
        rows = [json.loads(x) for x in (args.data / f"{split}.jsonl").read_text().splitlines()]
        # Whisper training has a 30s input limit. Exclude whole long utterances;
        # never crop audio while leaving its full transcript as the label.
        usable = [r for r in rows if r["duration_seconds"] <= 30]
        exclusions[split] = {
            "original": len(rows),
            "retained": len(usable),
            "excluded_longer_than_30s": len(rows) - len(usable),
        }
        if not usable:
            raise ValueError(f"No <=30s examples in {split}")
        ds = Dataset.from_list(usable).cast_column("audio", Audio(sampling_rate=16000))

        def preprocess(row):
            audio = row["audio"]
            return {
                "input_features": processor.feature_extractor(
                    audio["array"], sampling_rate=16000
                ).input_features[0],
                "labels": processor.tokenizer(row["reference"]).input_ids,
            }

        prepared = ds.map(preprocess, remove_columns=ds.column_names)
        usable = prepared.filter(
            lambda row: len(row["labels"]) <= model.config.max_target_positions
        )
        exclusions[split]["excluded_over_decoder_token_limit"] = len(prepared) - len(usable)
        exclusions[split]["retained_after_all_filters"] = len(usable)
        if not len(usable):
            raise ValueError(f"No examples fit the decoder context in {split}")
        return usable

    def metrics(prediction):
        labels = np.where(
            prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids
        )
        values = score(
            processor.tokenizer.batch_decode(labels, skip_special_tokens=True),
            processor.tokenizer.batch_decode(prediction.predictions, skip_special_tokens=True),
        )
        return {k: v for k, v in values.items() if v is not None}

    training = Seq2SeqTrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=4,
        learning_rate=1e-5,
        warmup_steps=100,
        max_steps=args.steps,
        fp16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        save_only_model=True,
        predict_with_generate=True,
        generation_max_length=448,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training,
        train_dataset=load("train"),
        eval_dataset=load("validation"),
        data_collator=Collator(processor, model.config.decoder_start_token_id),
        compute_metrics=metrics,
        processing_class=processor,
    )
    trainer.train()
    trainer.model.config.use_cache = True
    trainer.save_model(str(args.out / "best"))
    processor.save_pretrained(args.out / "best")
    (args.out / "selection.json").write_text(
        json.dumps(
            {
                "checkpoint": trainer.state.best_model_checkpoint,
                "artifacts": fingerprint(args.out / "best"),
                "validation_wer": trainer.state.best_metric,
                "seed": args.seed,
                "base_model": model_id,
                "base_model_revision": revision,
                "selection_metric": "validation WER; report TER jointly; test was not used",
                "length_exclusions": exclusions,
                "training_history": trainer.state.log_history,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

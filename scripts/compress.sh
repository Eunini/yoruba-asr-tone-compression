#!/usr/bin/env bash
set -euo pipefail
model_dir=${1:?usage: compress.sh MODEL_DIR OUTPUT_DIR WHISPER_CPP_DIR OPENAI_WHISPER_DIR}
output_dir=${2:?output directory required}
cpp_dir=${3:?reviewed whisper.cpp checkout required}
whisper_dir=${4:?reviewed openai/whisper checkout required}
mkdir -p "$output_dir"
# Current whisper.cpp converter reads pytorch_model.bin; export legacy weights first.
# The selected checkpoint stays immutable; conversion uses a separate export.
python - "$model_dir" "$output_dir/hf-bin" <<'PY'
import sys
from transformers import AutoTokenizer, WhisperForConditionalGeneration, WhisperProcessor
WhisperForConditionalGeneration.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2], safe_serialization=False)
WhisperProcessor.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
AutoTokenizer.from_pretrained(sys.argv[1], use_fast=True).save_pretrained(sys.argv[2])
PY
ct2-transformers-converter --model "$output_dir/hf-bin" --output_dir "$output_dir/ct2-int8" --quantization int8 --copy_files tokenizer.json preprocessor_config.json
python "$cpp_dir/models/convert-h5-to-ggml.py" "$output_dir/hf-bin" "$whisper_dir" "$output_dir"
for quant in q5_0 q4_0; do
  "$cpp_dir/build/bin/quantize" "$output_dir/ggml-model.bin" "$output_dir/ggml-model-$quant.bin" "$quant"
done

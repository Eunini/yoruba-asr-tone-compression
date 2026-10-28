# Attribution and changes

This release was prepared for the Eunini / Ineeoluwa Yoruba ASR tone-compression
research project. Fine-tuning contributions and the accompanying model card are
released under CC BY-SA 4.0 (`LICENSE`). This does not replace upstream rights or
license notices.

The base is OpenAI's `openai/whisper-tiny`, revision
`169d4a4341b33bc18d8881c4b69c2e104e1cc0af`. Its Hugging Face model card identifies
Apache-2.0; that license is retained in `LICENSE-BASE-APACHE-2.0.txt`. The original
OpenAI Whisper release also supplies an MIT license, retained verbatim in
`LICENSE-UPSTREAM-WHISPER-MIT.txt`. No endorsement by OpenAI is implied.

Training uses Google and partners' WAXAL dataset, specifically the Media Trust
Yoruba TTS recordings and transcripts, licensed CC-BY-SA-4.0 in the dataset card.
Credit Media Trust, the WAXAL contributors and the original speakers. Source:
https://huggingface.co/datasets/google/WaxalNLP/tree/5f4d8ca24f2b9d168b2ee545f1febaaff4b40580

Changes: recordings were repurposed for recognition; overlapping training
transcripts were removed, examples exceeding the training duration/context were
excluded, and Whisper weights were fine-tuned. A validation-selected checkpoint,
processor/tokenizer artifacts and measured evaluation metadata are distributed.
Raw recordings and transcripts are obtained separately from WAXAL. The model
weights differ from the original base checkpoint. No claim is made that the
training-data license is the only legal consideration for downstream use.

# Background, checked 2026-10-25

**WAXAL (Diack et al., 2026).** The paper presents a community-collected African
speech corpus covering both ASR and TTS, rather than a uniform ASR collection for
every listed language. Its contribution is the scale of transcribed speech,
language coverage and local annotation partnerships. For this project, the
critical distinction is that Yoruba is listed among TTS languages. The current
release exposes `yor_tts` with train, validation and test splits. The dataset card
also assigns provider-specific licenses, so the paper's general licensing summary
does not replace checking the actual Yoruba subset.
[Paper](https://arxiv.org/pdf/2602.02734),
[dataset card](https://huggingface.co/datasets/google/WaxalNLP).

**Tone in Yoruba ASR (Olusanya, 2026).** The study compares MMS-1B-all,
Whisper-small and a Yoruba-specific ASR model using tone-marked and unmarked text.
Its evaluation finds substantially better error scores on unmarked text and
argues that current modeling and tokenization do not adequately handle Yoruba
diacritics. This motivates reporting lexical accuracy separately from tone
accuracy: a single WER score can mix different problems. Its results motivate
this experiment but do not establish how quantization changes tones; that is the
new question to be tested here.
[Paper](https://aclanthology.org/2026.loreslm-1.14/).

**Voices Unheard / YORÙLECT (Ahia et al., 2024).** This paper introduces parallel
text and speech resources spanning Standard Yoruba and regional dialects. It
reports substantial disparities across dialects and improvements after
adaptation. In its ASR analysis, the authors specifically observe that Whisper
transcriptions lack diacritics and can be overly segmented, contributing to poor
WER. This warns against treating a narrow standard-language, shared-speaker
corpus as evidence about all Yoruba speakers, and motivates an additional dialect
or FLEURS evaluation before publication.
[Paper](https://arxiv.org/pdf/2406.19564),
[authors' repository](https://github.com/orevaahia/yorulect).

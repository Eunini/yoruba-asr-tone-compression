"""Tone-stripped WER and conditional, word-aligned explicit-tone edits."""

import unicodedata
from dataclasses import asdict, dataclass

import jiwer

TONES = frozenset(("\u0300", "\u0301"))


def strip_tones(text: str) -> str:
    """Remove only grave/acute marks. Underdots and all other marks survive."""
    return unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", text) if c not in TONES)
    )


def normalize(text: str) -> str:
    # Punctuation is deliberately retained; this policy is identical for both texts.
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def tones(word: str) -> list[str]:
    return [c for c in unicodedata.normalize("NFD", word) if c in TONES]


def edit_distance(left, right) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@dataclass
class Counts:
    word_errors: int = 0
    reference_words: int = 0
    matched_words: int = 0
    tone_edits: int = 0
    reference_tones: int = 0
    matched_reference_tones: int = 0

    def result(self) -> dict:
        return {
            **asdict(self),
            "wer": self.word_errors / self.reference_words if self.reference_words else None,
            "tone_error_rate": (
                self.tone_edits / self.matched_reference_tones
                if self.matched_reference_tones
                else None
            ),
            "reference_normalized_tone_edit_rate": (
                self.tone_edits / self.reference_tones
                if self.reference_tones and self.matched_words
                else None
            ),
            "conditional_tone_error_rate": (
                self.tone_edits / self.matched_reference_tones
                if self.matched_reference_tones
                else None
            ),
            "matched_word_coverage": (
                self.matched_words / self.reference_words if self.reference_words else None
            ),
            "matched_tone_coverage": (
                self.matched_reference_tones / self.reference_tones
                if self.reference_tones
                else None
            ),
        }


def count_pair(reference: str, hypothesis: str) -> Counts:
    ref, hyp = normalize(reference), normalize(hypothesis)
    aligned = jiwer.process_words(strip_tones(ref), strip_tones(hyp))
    result = Counts(
        word_errors=aligned.substitutions + aligned.deletions + aligned.insertions,
        reference_words=len(ref.split()),
        reference_tones=len(tones(ref)),
    )
    rw, hw = ref.split(), hyp.split()
    for chunk in aligned.alignments[0]:
        if chunk.type != "equal":
            continue
        for r, h in zip(
            rw[chunk.ref_start_idx : chunk.ref_end_idx],
            hw[chunk.hyp_start_idx : chunk.hyp_end_idx],
            strict=True,
        ):
            result.matched_words += 1
            result.matched_reference_tones += len(tones(r))
            result.tone_edits += edit_distance(tones(r), tones(h))
    return result


def score(references: list[str], hypotheses: list[str]) -> dict:
    total = Counts()
    if not references or len(references) != len(hypotheses):
        raise ValueError("Provide nonempty, equally sized reference/hypothesis lists")
    for ref, hyp in zip(references, hypotheses, strict=True):
        for key, value in asdict(count_pair(ref, hyp)).items():
            setattr(total, key, getattr(total, key) + value)
    return total.result()

# SPDX-License-Identifier: Apache-2.0
"""VoiceBBPE tokenizer for SpeechifyTTS (vLLM-free port).

Adapted from ``vllm_omni/.../speechify_t5_tts/tokenizer.py`` with the vLLM
``TokenizerLike`` / renderer / registry coupling removed. Behaviour preserved:

- Always wraps encodings as ``[encoder_start] + ids + [encoder_end]`` where
  ``encoder_start == number_text_tokens`` (read from the AR config) and
  ``encoder_end == [STOP] == 0`` (matches GPTTTS
  ``prepare_inputs_for_encoder_decoder``).
- Multilingual recipes (MoE 4B MTL, vocab 15006) use ``general_cleaners`` +
  ``production_cleaners`` and do NOT append the ``[BR5]`` line-break marker;
  EN BBPE recipes append a trailing ``[BR5]``.
- ``[SPACE]`` substitution for word boundaries; ``return_word_index`` yields
  speechmark word backtracking (requires PyICU; degrades with a clear error).
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from transformers import BatchEncoding, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

_whitespace_re = re.compile(r"\s+")
_punctuation_re = re.compile(r"[,.!?;:]")
_trailing_space_re = re.compile(r".+\s+$")


def sub_with_index(
    pattern: str,
    repl: str,
    string: str,
    indices: list[int] | None = None,
) -> tuple[str, list[int] | None]:
    """Regex substitution that propagates a character->word index mapping."""
    if indices is None:
        return re.sub(pattern, repl, string), None

    assert len(indices) == len(string), (
        f"Length of indices {len(indices)} and string {len(string)} mismatch"
    )
    offset = 0
    new_string = ""
    backtrack_indices: list[int] = []
    for match in re.finditer(pattern, string):
        start, end = match.span()
        new_string += string[offset:start]
        backtrack_indices += indices[offset:start]
        replacement = repl(match) if callable(repl) else repl
        replacement = match.expand(replacement)
        if len(replacement) > (end - start):
            new_indices = indices[start:end]
            last_index = new_indices[-1]
            new_indices += (len(replacement) - (end - start)) * [last_index]
        else:
            new_indices = indices[start : start + len(replacement)]
        backtrack_indices += new_indices
        new_string += replacement
        offset = end
    new_string += string[offset:]
    backtrack_indices += indices[offset:]
    return new_string, backtrack_indices


class Segmenter:
    """ICU-based word segmenter (lazy ICU import; speechmarks only)."""

    def __init__(self, locale: str = "en", granularity: str = "word"):
        self._locale_str = locale
        self.granularity = granularity

    def segment(self, text: str) -> list[dict[str, Any]]:
        try:
            from icu import BreakIterator, Locale
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                "PyICU is required for word segmentation (speechmarks). "
                "Install it with: pip install PyICU"
            ) from exc
        locale = Locale(self._locale_str)
        if self.granularity == "word":
            iterator = BreakIterator.createWordInstance(locale)
        else:
            raise ValueError(f"Unsupported granularity: {self.granularity}")
        iterator.setText(text)
        segments: list[dict[str, Any]] = []
        start = iterator.first()
        for end in iterator:
            segment = text[start:end]
            added_segment = False
            if re.match(_whitespace_re, segment):
                if segments:
                    segments[-1]["value"] += segment
                    added_segment = True
            elif re.match(_punctuation_re, segment):
                if segments and not re.match(_trailing_space_re, segments[-1]["value"]):
                    segments[-1]["value"] += segment
                    added_segment = True
            if not added_segment:
                segments.append({"value": segment, "startIndex": start})
            start = end
        return segments

    @staticmethod
    def char_to_word_index(segments: list[dict[str, Any]]) -> list[int]:
        indices: list[int] = []
        for i, seg in enumerate(segments):
            indices += [i] * len(seg["value"])
        return indices


class SpeechifyTTSTokenizer:
    """VoiceBBPE tokenizer for SpeechifyT5 TTS models."""

    def __init__(self, model_path: str | Path):
        self._tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(model_path) / "tokenizer.json"),
            model_max_length=1024,
        )
        self._vocab = self._tokenizer.get_vocab()
        self._id2token = {v: k for k, v in self._vocab.items()}

        self._stop_token_id = self._vocab.get("[STOP]", 0)
        self._unk_token_id = self._vocab.get("[UNK]", 1)
        self._space_token_id = self._vocab.get("[SPACE]", 2)
        self._br5_token_id = self._vocab.get("[BR5]")

        nt = self._lookup_number_text_tokens(model_path)
        if nt is None:
            nt = len(self._vocab)
            logger.warning(
                "SpeechifyTTSTokenizer: no AR config.json found under %s; "
                "falling back to number_text_tokens=len(vocab)=%d.",
                model_path,
                nt,
            )
        self._number_text_tokens = nt

        # [encoder_start] = number_text_tokens, [encoder_end] = [STOP] = 0.
        self._bos_token_id = self._number_text_tokens
        self._eos_token_id = self._stop_token_id
        self._pad_token_id = self._stop_token_id

        explicit_is_mtl = self._lookup_is_multilingual(model_path)
        if explicit_is_mtl is not None:
            self._is_multilingual = bool(explicit_is_mtl)
        else:
            self._is_multilingual = len(self._vocab) > 2000
            logger.warning(
                "SpeechifyTTSTokenizer: no is_multilingual field in AR config "
                "under %s; falling back to len(vocab)>2000 -> %s.",
                model_path,
                self._is_multilingual,
            )

        self._truncation_side = "left"
        self.name_or_path = str(model_path)
        self.word_segmenter = Segmenter(locale="en", granularity="word")

        logger.info(
            "SpeechifyTTSTokenizer: vocab=%d number_text_tokens=%d BOS=%d EOS=%d "
            "is_multilingual=%s",
            len(self._vocab),
            self._number_text_tokens,
            self._bos_token_id,
            self._eos_token_id,
            self._is_multilingual,
        )

    @staticmethod
    def _lookup_config_field(model_path: str | Path, field: str):
        candidates = (
            Path(model_path) / "config.json",
            Path(model_path) / "ar" / "config.json",
            Path(model_path).parent / "ar" / "config.json",
        )
        for c in candidates:
            try:
                if c.is_file():
                    with c.open() as fh:
                        cfg = json.load(fh)
                    if field in cfg:
                        return cfg[field]
            except (OSError, ValueError):
                continue
        return None

    @classmethod
    def _lookup_number_text_tokens(cls, model_path: str | Path) -> int | None:
        nt = cls._lookup_config_field(model_path, "number_text_tokens")
        if isinstance(nt, int) and nt > 0:
            return nt
        return None

    @classmethod
    def _lookup_is_multilingual(cls, model_path: str | Path) -> bool | None:
        val = cls._lookup_config_field(model_path, "is_multilingual")
        return None if val is None else bool(val)

    @classmethod
    def from_pretrained(cls, path_or_repo_id: str | Path, *args, **kwargs):
        return cls(str(path_or_repo_id))

    def _preprocess_text(self, text: str) -> str:
        if self._is_multilingual:
            from sglang_omni.models.speechify_tts.text_preprocessing import (
                preprocess_multilingual_text,
            )

            return preprocess_multilingual_text(text, production=True)
        from sglang_omni.models.speechify_tts.text_preprocessing import preprocess_text

        return preprocess_text(text)

    def create_backtrack_mapping(
        self, tokens: Any, word_indices: list[int]
    ) -> list[int]:
        backtrack: list[int] = []
        for _tid, (s, _e) in zip(tokens.ids, tokens.offsets):
            if s < len(word_indices):
                backtrack.append(word_indices[s])
            else:
                backtrack.append(word_indices[-1] if word_indices else 0)
        return backtrack

    def encode(
        self,
        text: str,
        truncation: bool | None = None,
        max_length: int | None = None,
        add_special_tokens: bool = True,
        return_word_index: bool = False,
        normalized_text: str | None = None,
        normalized_c2w: list[int] | None = None,
        original_words: list[dict[str, Any]] | None = None,
    ):
        word_indices: list[int] | None = None
        words_out: list[dict[str, Any]] = []
        if return_word_index:
            if (
                normalized_text is not None
                and normalized_c2w is not None
                and original_words is not None
            ):
                words_out = original_words
                text_to_tokenize, word_indices = sub_with_index(
                    r"\s+", "[SPACE]", normalized_text, normalized_c2w
                )
            else:
                words_out = self.word_segmenter.segment(text)
                c2w = self.word_segmenter.char_to_word_index(words_out)
                text_to_tokenize, word_indices = sub_with_index(
                    r"\s+", "[SPACE]", text, c2w
                )
        else:
            text_to_tokenize = self._preprocess_text(text)

        force = os.environ.get("VLLM_OMNI_FORCE_TRAILING_BR5")
        if force == "1":
            append_br5 = True
        elif force == "0":
            append_br5 = False
        else:
            append_br5 = not self._is_multilingual
        if append_br5:
            text_to_tokenize = text_to_tokenize + "[BR5]"

        tokens = self._tokenizer.backend_tokenizer.encode(text_to_tokenize)
        token_ids = tokens.ids
        token_ids = [self._bos_token_id] + token_ids + [self._eos_token_id]

        if truncation and max_length is not None and len(token_ids) > max_length:
            if self._truncation_side == "left":
                token_ids = [self._bos_token_id] + token_ids[-(max_length - 1) :]
            else:
                token_ids = token_ids[: max_length - 1] + [self._eos_token_id]

        if not return_word_index:
            return token_ids

        backtrack = self.create_backtrack_mapping(tokens, word_indices)
        for w in words_out:
            w["value"] = w["value"].strip()
        last_word = max(len(words_out) - 1, 0)
        word_backtrack = [0] + backtrack + [last_word]
        return token_ids, word_backtrack, words_out

    def decode(self, ids, skip_special_tokens: bool = False) -> str:
        if isinstance(ids, int):
            ids = [ids]
        text = self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        text = text.replace("[SPACE]", " ")
        if skip_special_tokens:
            text = text.replace("[STOP]", "").replace("[UNK]", "")
        return text

    def __call__(
        self,
        text: str | list[str],
        text_pair: str | None = None,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> BatchEncoding:
        texts = [text] if isinstance(text, str) else text
        all_input_ids = [
            self.encode(t, truncation=truncation, max_length=max_length) for t in texts
        ]
        return BatchEncoding({"input_ids": all_input_ids})

    def num_special_tokens_to_add(self) -> int:
        return 2

    @property
    def all_special_tokens(self) -> list[str]:
        return ["[STOP]", "[UNK]", "[SPACE]", "<BOS>"]

    @property
    def all_special_ids(self) -> list[int]:
        return [
            self._stop_token_id,
            self._unk_token_id,
            self._space_token_id,
            self._bos_token_id,
        ]

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def is_fast(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    @property
    def max_token_id(self) -> int:
        return max(self._vocab.values())

    @property
    def max_chars_per_token(self) -> int:
        return max((len(t) for t in self._vocab), default=1)

    @property
    def truncation_side(self) -> str:
        return self._truncation_side

    @truncation_side.setter
    def truncation_side(self, value: str) -> None:
        self._truncation_side = value

    def __len__(self) -> int:
        return self.vocab_size

    def __hash__(self) -> int:
        return hash(id(self))

    def get_vocab(self) -> dict[str, int]:
        return self._vocab.copy()

    def get_added_vocab(self) -> dict[str, int]:
        return {
            "[STOP]": self._stop_token_id,
            "[UNK]": self._unk_token_id,
            "[SPACE]": self._space_token_id,
            "<BOS>": self._bos_token_id,
        }

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._vocab.get(tokens, self._unk_token_id)
        return [self._vocab.get(t, self._unk_token_id) for t in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        special_ids = set(self.all_special_ids) if skip_special_tokens else set()
        tokens = []
        for id_ in ids:
            if id_ in special_ids:
                continue
            tokens.append(self._id2token.get(id_, "[UNK]"))
        return tokens

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return "".join(tokens).replace("[SPACE]", " ")

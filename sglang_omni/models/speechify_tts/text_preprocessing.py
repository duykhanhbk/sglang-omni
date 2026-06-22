# SPDX-License-Identifier: Apache-2.0
"""Text preprocessing for SpeechifyTTS, ported from GPTTTS VoiceBpeTokenizer.

Replicates the ``production=True`` pipeline:
1. convert_to_ascii (unidecode)
2. expand_numbers (inflect)
3. expand_abbreviations
4. collapse_whitespace
5. bracket/quote normalization
6. production rules (space-before-punct, parens->apostrophes, colons/slashes->commas)

``inflect`` / ``unidecode`` are imported lazily so the multilingual MoE 4B path
(``general_cleaners`` + ``production_cleaners``) works even if they are absent.
"""

from __future__ import annotations

import re

_whitespace_re = re.compile(r"\s+")

_comma_number_re = re.compile(r"([0-9][0-9\,]+[0-9])")
_decimal_number_re = re.compile(r"([0-9]+\.[0-9]+)")
_pounds_re = re.compile(r"£([0-9\,]*[0-9]+)")
_dollars_re = re.compile(r"\$([0-9\.\,]*[0-9]+)")
_ordinal_re = re.compile(r"[0-9]+(st|nd|rd|th)")
_number_re = re.compile(r"[0-9]+")

_abbreviations = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

DIGIT_MAP = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}

_inflect = None


def _get_inflect():
    global _inflect
    if _inflect is None:
        import inflect

        _inflect = inflect.engine()
    return _inflect


def _remove_commas(m):
    return m.group(1).replace(",", "")


def _expand_decimal_point(m):
    return m.group(1).replace(".", " point ")


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split(".")
    if len(parts) > 2:
        return match + " dollars"
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s, %s %s" % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = "dollar" if dollars == 1 else "dollars"
        return "%s %s" % (dollars, dollar_unit)
    elif cents:
        cent_unit = "cent" if cents == 1 else "cents"
        return "%s %s" % (cents, cent_unit)
    else:
        return "zero dollars"


def _expand_ordinal(m):
    return _get_inflect().number_to_words(m.group(0))


def _convert_digit_to_readable_text(digit):
    return " ".join(DIGIT_MAP[i] for i in digit)


def _expand_number(m):
    num = int(m.group(0))
    num_str = m.group(0)
    eng = _get_inflect()
    if num > 1000 and num < 3000:
        if num == 2000:
            return "two thousand"
        elif num > 2000 and num < 2010:
            return "two thousand " + eng.number_to_words(num % 100)
        elif num % 100 == 0:
            return eng.number_to_words(num // 100) + " hundred"
        else:
            return eng.number_to_words(num, andword="", zero="oh", group=2).replace(
                ", ", " "
            )
    elif len(num_str) > 30:
        return _convert_digit_to_readable_text(num_str)
    else:
        return eng.number_to_words(num, andword="")


def convert_to_ascii(text: str) -> str:
    from unidecode import unidecode_expect_ascii

    return unidecode_expect_ascii(text)


def expand_numbers(text: str) -> str:
    text = _comma_number_re.sub(_remove_commas, text)
    text = _pounds_re.sub(r"\1 pounds", text)
    text = _dollars_re.sub(_expand_dollars, text)
    text = _decimal_number_re.sub(_expand_decimal_point, text)
    text = _ordinal_re.sub(_expand_ordinal, text)
    text = _number_re.sub(_expand_number, text)
    return text


def expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    return text


def collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text)


def english_cleaners(text: str) -> str:
    text = convert_to_ascii(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    text = text.replace("[", "(")
    text = text.replace("]", ")")
    text = text.replace("{", "(")
    text = text.replace("}", ")")
    text = text.replace('"', "'")
    return text


def production_cleaners(text: str) -> str:
    text = re.sub(r"\s+([,!?.\;])(?<!$)", r"\1", text)
    text = re.sub(r"[\(\)]", "'", text)
    text = re.sub(r"[:;/]", ",", text)
    return text


def general_cleaners(text: str) -> str:
    text = collapse_whitespace(text)
    text = re.sub(r"\.\.\.", ",", text)
    text = text.replace("…", ",")
    return text


def preprocess_text(text: str, normalize: bool = False) -> str:
    if normalize:
        text = expand_numbers(text)
    text = text.replace(" ", "[SPACE]")
    return text


def preprocess_multilingual_text(text: str, *, production: bool = True) -> str:
    text = general_cleaners(text)
    if production:
        text = production_cleaners(text)
    text = text.replace(" ", "[SPACE]")
    return text

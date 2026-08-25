"""Shared constants, paths and text utilities.

Deliberately stdlib-only so that `data_prep.py` and `lid.py` can run on the
local M1 (Python 3.14, no third-party packages installed) while the training
code runs on Colab/Kaggle.
"""

import os
import re
import unicodedata

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = DATA_DIR                       # data/en_hi, data/en_bn
CLEAN_DIR = os.path.join(DATA_DIR, "clean")
SPM_DIR = os.path.join(ROOT, "spm")
DICTS_DIR = os.path.join(ROOT, "dicts")
OUTS_DIR = os.path.join(ROOT, "outs")
SUBMISSION_DIR = os.path.join(ROOT, "submission")
LOGS_DIR = os.path.join(ROOT, "logs")
VAL_JSON = os.path.join(ROOT, "val_data1.json")

# --------------------------------------------------------------------------
# Language identifiers
# --------------------------------------------------------------------------

# Directory name -> (language code, target-language tag token)
PAIRS = {
    "en_bn": "bn",
    "en_hi": "hi",
}

# Tag token prepended to every source sentence. This is what makes a single
# multilingual model able to serve both directions (Johnson et al., 2017,
# "Google's Multilingual Neural Machine Translation System", arXiv:1611.04558).
LANG_TAG = {"hi": "<2hi>", "bn": "<2bn>"}
TAG_TOKENS = ["<2hi>", "<2bn>"]

# Top-level keys of val_data1.json -> language code
VAL_SECTIONS = {
    "English-Bengali": "bn",
    "English-Hindi": "hi",
}

# --------------------------------------------------------------------------
# SentencePiece special token ids.
#
# These are pinned explicitly here, passed to spm_train as --pad_id/--unk_id/
# --bos_id/--eos_id, and asserted in gates.py. The all-<unk> failure of the
# previous attempt (Capstone_v5) was caused by these drifting between the
# tokenizer, the dataset and the decoder.
# --------------------------------------------------------------------------

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3

# --------------------------------------------------------------------------
# Unicode ranges
# --------------------------------------------------------------------------

DEVANAGARI = (0x0900, 0x097F)
BENGALI = (0x0980, 0x09FF)

SCRIPT_RANGE = {"hi": DEVANAGARI, "bn": BENGALI}

# Letters that exist in the Bengali Unicode block but are Assamese-only.
# The Hindi training file contains a handful of Assamese rows; a plain
# block-membership test cannot catch those, which is why lid.py exists.
ASSAMESE_ONLY = set("ৰৱ")  # ৰ (RA WITH MIDDLE DIAGONAL), ৱ (RA WITH LOWER DIAGONAL)

DANDA = "।"          # । DEVANAGARI DANDA
BENGALI_NUM_FOUR = "৷"  # ৷ widely used as a danda in Bengali corpora
PASEQ = "׀"          # ׀ HEBREW PUNCTUATION PASEQ - corruption, never legitimate here

# Zero-width characters that carry no orthographic meaning and should go.
ZERO_WIDTH_STRIP = "﻿​⁠­"
# ZWNJ (U+200C) and ZWJ (U+200D) are deliberately NOT stripped: they are
# meaningful in Devanagari/Bengali conjunct formation (e.g. मुक्‍त).

_WS_RE = re.compile(r"\s+")
_DIGIT_PREFIX_RE = re.compile(r"^\s*\d+\s*:\s*")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]")


def strip_digit_prefix(text):
    """Remove the scraped line-number artefact, e.g. '65: शेष सोयाबीन...'.

    Present on 670 Hindi training targets (0.87% of the Hindi training set).
    """
    return _DIGIT_PREFIX_RE.sub("", text)


def normalise(text, lang=None, fold_bengali_danda=False):
    """Normalise one field.

    Fixes only unambiguous corruption. Notably it does NOT change the
    tokenisation style: ~10% of the corpus is Moses/IndicNLP pre-tokenised
    (space before punctuation) and val_data1.json is mixed the same way, so the
    hidden references almost certainly preserve both styles. Forcing one style
    would systematically mismatch them.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_RE.sub(" ", text)
    for ch in ZERO_WIDTH_STRIP:
        text = text.replace(ch, "")

    if lang == "bn":
        # U+05C0 is a Hebrew paseq standing in for a danda on 4,528 lines.
        text = text.replace(PASEQ, DANDA)
        if fold_bengali_danda:
            text = text.replace(BENGALI_NUM_FOUR, DANDA)
    elif lang == "hi":
        # ASCII pipe used as a danda; final character of 1,153 targets.
        text = text.replace("|", DANDA)

    text = _WS_RE.sub(" ", text).strip()
    return text


def script_ratio(text, lang):
    """Fraction of *letter* characters that fall in the expected script block."""
    lo, hi = SCRIPT_RANGE[lang]
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    inside = sum(1 for c in letters if lo <= ord(c) <= hi)
    return inside / len(letters)


def read_tsv(path):
    """Yield (source, target) pairs from a 2-column, header-less TSV."""
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"{path}:{lineno} has {len(parts)} fields, expected 2")
            yield parts[0], parts[1]


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for src, tgt in rows:
            fh.write(f"{src}\t{tgt}\n")


def tag_source(src, lang):
    """Prepend the target-language tag: '<2hi> Some English sentence'."""
    return f"{LANG_TAG[lang]} {src}"

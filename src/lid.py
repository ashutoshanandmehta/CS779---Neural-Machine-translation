"""Character n-gram language identification, used as a data-cleaning tool.

This is the useful form of the "language detection" idea. It is *not* a router:
the source side of this corpus is always English, so a detector reading the
input cannot tell you whether Hindi or Bengali is wanted -- that is target-side
metadata, and val_data1.json already supplies it as a top-level key.

What a detector *can* do is find rows whose target is not in the language the
file claims. The corpus has exactly that problem: `data/en_hi/train.tsv`
contains Assamese-script targets and `data/en_bn/train.tsv` contains
Devanagari ones, in both cases alongside a semantic misalignment.

Two complementary checks:

  1. `CharNGramLID` - multinomial Naive Bayes over character 1..3-grams,
     classes {en, hi, bn}. Catches cross-script contamination.
  2. `has_assamese` - Assamese and Bengali share a Unicode block, so no
     block-membership or n-gram model trained on this corpus can separate
     them. The reliable signal is the two Assamese-only letters.

Pure stdlib, so it runs on the local machine with no installs.
"""

import argparse
import collections
import json
import math
import os
import random

from common import (
    ASSAMESE_ONLY,
    CLEAN_DIR,
    DICTS_DIR,
    PAIRS,
    RAW_DIR,
    normalise,
    read_tsv,
)

CLASSES = ("en", "hi", "bn")
NGRAM_ORDERS = (1, 2, 3)
MAX_FEATURES_PER_CLASS = 60000


def ngrams(text, orders=NGRAM_ORDERS):
    """Character n-grams over a space-padded string."""
    text = f" {text.strip()} "
    for n in orders:
        for i in range(len(text) - n + 1):
            yield text[i : i + n]


def has_assamese(text):
    """True if the text uses a letter that exists in Assamese but not Bengali."""
    return any(ch in ASSAMESE_ONLY for ch in text)


class CharNGramLID:
    """Multinomial Naive Bayes with add-alpha smoothing over character n-grams."""

    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.log_prior = {}
        self.log_prob = {}          # class -> {ngram: log P(ngram | class)}
        self.log_unseen = {}        # class -> log P(unseen ngram | class)

    # -- training ----------------------------------------------------------

    def fit(self, samples):
        """samples: iterable of (text, class_label)."""
        counts = {c: collections.Counter() for c in CLASSES}
        docs = collections.Counter()

        for text, label in samples:
            docs[label] += 1
            counts[label].update(ngrams(text))

        # Trim the long tail so the saved model stays small and fast to load.
        for c in CLASSES:
            if len(counts[c]) > MAX_FEATURES_PER_CLASS:
                counts[c] = collections.Counter(
                    dict(counts[c].most_common(MAX_FEATURES_PER_CLASS))
                )

        vocab = set()
        for c in CLASSES:
            vocab.update(counts[c])
        v = len(vocab)

        total_docs = sum(docs.values())
        for c in CLASSES:
            self.log_prior[c] = math.log(max(docs[c], 1) / total_docs)
            denom = sum(counts[c].values()) + self.alpha * v
            self.log_prob[c] = {
                g: math.log((k + self.alpha) / denom) for g, k in counts[c].items()
            }
            self.log_unseen[c] = math.log(self.alpha / denom)

        return self

    # -- inference ---------------------------------------------------------

    def scores(self, text):
        out = {}
        grams = list(ngrams(text))
        for c in CLASSES:
            table = self.log_prob[c]
            unseen = self.log_unseen[c]
            out[c] = self.log_prior[c] + sum(table.get(g, unseen) for g in grams)
        return out

    def predict(self, text):
        s = self.scores(text)
        return max(s, key=s.get)

    # -- persistence -------------------------------------------------------

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "alpha": self.alpha,
                    "log_prior": self.log_prior,
                    "log_prob": self.log_prob,
                    "log_unseen": self.log_unseen,
                },
                fh,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        m = cls(alpha=blob["alpha"])
        m.log_prior = blob["log_prior"]
        m.log_prob = blob["log_prob"]
        m.log_unseen = blob["log_unseen"]
        return m


# --------------------------------------------------------------------------


def build_samples(limit_per_class=20000, seed=13):
    """Collect (text, label) training data straight from the raw corpus.

    English comes from the source side of both files; Hindi and Bengali from
    their respective target sides. The corpus contamination we are trying to
    find is a few dozen rows out of 142k, far too little to poison the model.
    """
    rng = random.Random(seed)
    buckets = {c: [] for c in CLASSES}

    for pair_dir, lang in PAIRS.items():
        path = os.path.join(RAW_DIR, pair_dir, "train.tsv")
        for src, tgt in read_tsv(path):
            buckets["en"].append(normalise(src))
            buckets[lang].append(normalise(tgt, lang))

    samples = []
    for c in CLASSES:
        rng.shuffle(buckets[c])
        samples.extend((t, c) for t in buckets[c][:limit_per_class] if t)
    rng.shuffle(samples)
    return samples


def main():
    ap = argparse.ArgumentParser(description="Train the char n-gram LID model.")
    ap.add_argument("--out", default=os.path.join(DICTS_DIR, "lid_charngram.json"))
    ap.add_argument("--limit-per-class", type=int, default=20000)
    ap.add_argument("--holdout", type=int, default=2000)
    args = ap.parse_args()

    samples = build_samples(args.limit_per_class)
    holdout, train = samples[: args.holdout], samples[args.holdout :]
    print(f"train={len(train)}  holdout={len(holdout)}")

    model = CharNGramLID().fit(train)

    correct = sum(1 for t, y in holdout if model.predict(t) == y)
    print(f"holdout accuracy: {correct}/{len(holdout)} = {correct / len(holdout):.4f}")

    confusion = collections.Counter(
        (y, model.predict(t)) for t, y in holdout if model.predict(t) != y
    )
    if confusion:
        print("errors (true -> predicted):")
        for (y, p), k in confusion.most_common():
            print(f"  {y} -> {p}: {k}")

    model.save(args.out)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()

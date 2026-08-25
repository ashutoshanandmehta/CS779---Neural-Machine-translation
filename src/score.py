"""Local scorer that mimics the competition grader.

The grader (`evaluate.py` in the repo root) sums three metrics:

    total_score = bleu + rouge + chrf

and it does NOT use sacrebleu. Its BLEU helper calls
`nltk.translate.bleu_score.corpus_bleu` over `nltk.word_tokenize` output with
uniform weights and no smoothing, returning a 0..1 value -- a considerably
harsher number than sacrebleu's 0..100. Tuning against sacrebleu and reporting
against this would be measuring two different things, so the implementations
below follow the grader's semantics.

Everything is pure stdlib so this runs on the local machine. If `nltk` /
`sacrebleu` happen to be installed, `--cross-check` compares against them.

Caveat worth stating in any write-up: `chrfscore.py` is not present anywhere
in the project, so the chrF here follows sacrebleu's chrF++ definition
(char n-grams to order 6, word n-grams to order 2, beta=2). It is a
best-effort reconstruction, not a verified match.
"""

import argparse
import collections
import json
import math
import os
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import ROOT  # noqa: E402

# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

try:  # prefer the grader's exact tokeniser when it is available
    import nltk

    nltk.data.find("tokenizers/punkt")
    _HAVE_NLTK = True
except Exception:
    _HAVE_NLTK = False


def _fallback_tokenize(text):
    """Whitespace split, then peel leading and trailing punctuation.

    A plain `\\w+|[^\\w\\s]` regex is wrong for Devanagari and Bengali. Matras and
    the halant are Unicode categories Mn and Mc, `str.isalnum()` is False for
    them, so `\\w` does not match and every word shatters at its combining marks:
    'নমস্তে' becomes ['নমস', '্', 'ত', 'ে']. That scores BLEU near character level
    and roughly doubles it. Peeling punctuation off whitespace tokens keeps words
    intact and stays close to nltk.word_tokenize on this data.
    """
    out = []
    for word in text.split():
        i = 0
        while i < len(word) and unicodedata.category(word[i]).startswith("P"):
            out.append(word[i])
            i += 1
        j = len(word)
        tail = []
        while j > i and unicodedata.category(word[j - 1]).startswith("P"):
            tail.append(word[j - 1])
            j -= 1
        if j > i:
            out.append(word[i:j])
        out.extend(reversed(tail))
    return out


def word_tokenize(text, use_nltk=True):
    if use_nltk and _HAVE_NLTK:
        return nltk.word_tokenize(text)
    return _fallback_tokenize(text)


# --------------------------------------------------------------------------
# BLEU  (equivalent to nltk.translate.bleu_score.corpus_bleu, no smoothing)
# --------------------------------------------------------------------------


def _ngram_counts(tokens, n):
    return collections.Counter(
        tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
    )


def corpus_bleu(hypotheses, references, max_n=4):
    """Corpus BLEU-4, uniform weights, no smoothing. Returns 0..1.

    `references` is a list of single reference strings (this task has one
    reference per sentence).
    """
    numerators = [0] * (max_n + 1)
    denominators = [0] * (max_n + 1)
    hyp_len = ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        h = word_tokenize(hyp)
        r = word_tokenize(ref)
        hyp_len += len(h)
        ref_len += len(r)
        for n in range(1, max_n + 1):
            hc = _ngram_counts(h, n)
            rc = _ngram_counts(r, n)
            overlap = sum(min(c, rc[g]) for g, c in hc.items())
            numerators[n] += overlap
            denominators[n] += max(sum(hc.values()), 0)

    if numerators[1] == 0:
        return 0.0

    # NLTK's method0 substitutes sys.float_info.min for a zero precision,
    # which drives the geometric mean to ~0.
    logs = []
    for n in range(1, max_n + 1):
        if denominators[n] == 0:
            return 0.0
        p = numerators[n] / denominators[n]
        logs.append(0.25 * math.log(p if p > 0 else sys.float_info.min))

    # Brevity penalty
    if hyp_len > ref_len:
        bp = 1.0
    elif hyp_len == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_len / hyp_len)

    return bp * math.exp(math.fsum(logs))


# --------------------------------------------------------------------------
# ROUGE-L  (formula from the `rouge` package used by the grader)
# --------------------------------------------------------------------------


def _lcs_length(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l(hypotheses, references):
    """Mean ROUGE-L F over the corpus, following pltrdy/rouge's _f_lcs."""
    total = 0.0
    n = 0
    for hyp, ref in zip(hypotheses, references):
        h = word_tokenize(hyp)
        r = word_tokenize(ref)
        n += 1
        if not h or not r:
            continue
        llcs = _lcs_length(r, h)
        if llcs == 0:
            continue
        r_lcs = llcs / len(r)
        p_lcs = llcs / len(h)
        beta = p_lcs / (r_lcs + 1e-12)
        num = (1 + beta ** 2) * r_lcs * p_lcs
        den = r_lcs + (beta ** 2) * p_lcs
        total += num / (den + 1e-12)
    return total / max(n, 1)


# --------------------------------------------------------------------------
# chrF++  (sacrebleu definition: char n-grams to 6, word n-grams to 2, beta=2)
# --------------------------------------------------------------------------


def _chrf_stats(hyp, ref, char_order=6, word_order=2):
    stats = []
    h_chars = "".join(hyp.split())
    r_chars = "".join(ref.split())
    for n in range(1, char_order + 1):
        hc = _ngram_counts(list(h_chars), n)
        rc = _ngram_counts(list(r_chars), n)
        match = sum(min(c, rc[g]) for g, c in hc.items())
        stats.append((match, sum(hc.values()), sum(rc.values())))

    h_words = hyp.split()
    r_words = ref.split()
    for n in range(1, word_order + 1):
        hc = _ngram_counts(h_words, n)
        rc = _ngram_counts(r_words, n)
        match = sum(min(c, rc[g]) for g, c in hc.items())
        stats.append((match, sum(hc.values()), sum(rc.values())))
    return stats


def chrf(hypotheses, references, char_order=6, word_order=2, beta=2.0):
    """Corpus-level chrF++ in 0..1 (sacrebleu reports the same number x100)."""
    n_orders = char_order + word_order
    agg = [[0, 0, 0] for _ in range(n_orders)]
    for hyp, ref in zip(hypotheses, references):
        for i, (m, ht, rt) in enumerate(_chrf_stats(hyp, ref, char_order, word_order)):
            agg[i][0] += m
            agg[i][1] += ht
            agg[i][2] += rt

    precisions, recalls = [], []
    for m, ht, rt in agg:
        precisions.append(m / ht if ht else 0.0)
        recalls.append(m / rt if rt else 0.0)

    avg_p = sum(precisions) / n_orders
    avg_r = sum(recalls) / n_orders
    if avg_p + avg_r == 0:
        return 0.0
    b2 = beta ** 2
    return (1 + b2) * avg_p * avg_r / (b2 * avg_p + avg_r)


# --------------------------------------------------------------------------


def score_all(hypotheses, references):
    b = corpus_bleu(hypotheses, references)
    r = rouge_l(hypotheses, references)
    c = chrf(hypotheses, references)
    return {
        "bleu": round(b, 4),
        "rouge_l": round(r, 4),
        "chrf": round(c, 4),
        "total": round(round(b, 3) + round(r, 3) + round(c, 3), 4),
        "n": len(hypotheses),
    }


def cross_check(hypotheses, references):
    """Compare against the real libraries when they are installed."""
    out = {}
    try:
        import sacrebleu

        out["sacrebleu_bleu_x100"] = round(
            sacrebleu.corpus_bleu(hypotheses, [references]).score, 3
        )
        out["sacrebleu_chrf++_x100"] = round(
            sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 3
        )
    except Exception as exc:
        out["sacrebleu"] = f"unavailable ({type(exc).__name__})"

    if _HAVE_NLTK:
        try:
            from nltk.translate.bleu_score import corpus_bleu as nltk_bleu

            out["nltk_bleu"] = round(
                nltk_bleu(
                    [[nltk.word_tokenize(r)] for r in references],
                    [nltk.word_tokenize(h) for h in hypotheses],
                ),
                4,
            )
        except Exception as exc:
            out["nltk_bleu"] = f"unavailable ({type(exc).__name__})"
    else:
        out["nltk_bleu"] = "unavailable (nltk/punkt not installed)"
    return out


def read_hyp_ref(hyp_path, ref_path):
    with open(hyp_path, encoding="utf-8") as fh:
        hyps = [ln.rstrip("\n") for ln in fh]
    langs, refs = [], []
    with open(ref_path, encoding="utf-8") as fh:
        for ln in fh:
            src, tgt = ln.rstrip("\n").split("\t")
            langs.append(src.split(" ", 1)[0].strip("<>2"))
            refs.append(tgt)
    if len(hyps) != len(refs):
        sys.exit(f"length mismatch: {len(hyps)} hypotheses vs {len(refs)} references")
    return hyps, refs, langs


def main():
    ap = argparse.ArgumentParser(description="Score translations like the grader does.")
    ap.add_argument("--hyp", required=True, help="one translation per line")
    ap.add_argument("--ref", required=True, help="tagged TSV, e.g. data/clean/combined/dev.tsv")
    ap.add_argument("--cross-check", action="store_true")
    ap.add_argument("--out", default=None, help="write scores as JSON")
    args = ap.parse_args()

    hyps, refs, langs = read_hyp_ref(args.hyp, args.ref)

    results = {"overall": score_all(hyps, refs)}
    for lang in sorted(set(langs)):
        idx = [i for i, l in enumerate(langs) if l == lang]
        results[lang] = score_all([hyps[i] for i in idx], [refs[i] for i in idx])

    if args.cross_check:
        results["cross_check"] = cross_check(hyps, refs)

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

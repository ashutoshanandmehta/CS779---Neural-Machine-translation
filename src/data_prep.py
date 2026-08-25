"""Clean, filter and de-leak the parallel corpora.

Stdlib only -- runs on the local machine with nothing installed.

Reads   data/{en_hi,en_bn}/{train,dev,test}.tsv   (2-col TSV, no header)
Writes  data/clean/{en_hi,en_bn}/{train,dev,test}.tsv
        data/clean/combined/{train,dev,test}.tsv  (source carries a <2xx> tag)
        data/clean/spm_input.txt                  (tokeniser training text)
        data/clean/report.json                    (per-filter drop counts)
        dicts/tm_train.json / dicts/tm_all.json   (translation memories)

Every filter records its own count. The report is both a correctness check and
material for the write-up.
"""

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    ASSAMESE_ONLY,
    CLEAN_DIR,
    DICTS_DIR,
    PAIRS,
    RAW_DIR,
    VAL_JSON,
    VAL_SECTIONS,
    normalise,
    read_tsv,
    script_ratio,
    strip_digit_prefix,
    tag_source,
    write_tsv,
)
from lid import CharNGramLID, has_assamese  # noqa: E402

SPLITS = ("train", "dev", "test")

# Filter thresholds
MIN_SCRIPT_RATIO = 0.50   # fraction of target letters that must be in-script
MAX_LEN_RATIO = 3.0       # character-length ratio between source and target
MIN_LEN_FOR_RATIO = 10    # don't length-filter very short pairs
MIN_LEN_FOR_LID = 15      # don't ask the LID model about very short strings
MAX_ASSAMESE_CHARS = 3    # Assamese-only letters tolerated in a Bengali target


def clean_pair(src, tgt, lang, fold_bengali_danda):
    """Apply all surface fixes. Returns (src, tgt, flags) with flags a set."""
    flags = set()

    raw_tgt = tgt
    tgt = strip_digit_prefix(tgt)
    if tgt != raw_tgt:
        flags.add("digit_prefix_stripped")

    raw_src = src
    src2 = strip_digit_prefix(src)
    if src2 != raw_src:
        flags.add("digit_prefix_stripped_src")
    src = src2

    before = (src, tgt)
    src = normalise(src, lang=None)
    tgt = normalise(tgt, lang=lang, fold_bengali_danda=fold_bengali_danda)
    if (src, tgt) != before:
        flags.add("normalised")

    return src, tgt, flags


def should_drop(src, tgt, lang, lid):
    """Return a drop reason, or None to keep."""
    if not src or not tgt:
        return "empty_field"

    if src == tgt:
        return "src_equals_tgt"

    ratio = script_ratio(tgt, lang)
    if ratio < MIN_SCRIPT_RATIO:
        return "wrong_script"

    n_assamese = sum(1 for ch in tgt if ch in ASSAMESE_ONLY)
    if n_assamese >= MAX_ASSAMESE_CHARS:
        return "assamese_target"

    if lid is not None and len(tgt) >= MIN_LEN_FOR_LID:
        if lid.predict(tgt) != lang:
            return "lid_language_mismatch"

    ls, lt = len(src), len(tgt)
    if min(ls, lt) >= MIN_LEN_FOR_RATIO:
        if max(ls, lt) / min(ls, lt) > MAX_LEN_RATIO:
            return "length_ratio"

    return None


def load_split(pair_dir, split, lang, lid, fold_bengali_danda, stats):
    """Clean and filter one file. Returns the surviving rows."""
    path = os.path.join(RAW_DIR, pair_dir, f"{split}.tsv")
    key = f"{pair_dir}/{split}"
    st = stats[key] = collections.Counter()

    rows = []
    for src, tgt in read_tsv(path):
        st["read"] += 1
        src, tgt, flags = clean_pair(src, tgt, lang, fold_bengali_danda)
        for f in flags:
            st[f] += 1

        reason = should_drop(src, tgt, lang, lid)
        if reason is not None:
            st[f"dropped_{reason}"] += 1
            continue
        rows.append((src, tgt))

    # Exact duplicates -- only ever removed from train; dev/test stay intact
    # so that the evaluation set is not silently altered.
    if split == "train":
        seen = set()
        deduped = []
        for src, tgt in rows:
            if (src, tgt) in seen:
                st["dropped_duplicate"] += 1
                continue
            seen.add((src, tgt))
            deduped.append((src, tgt))
        rows = deduped

    st["kept"] = len(rows)
    return rows


def deleak(train_rows, eval_rows, stats_key, stats):
    """Drop training rows whose source also appears in dev or test.

    Left alone, this inflates every dev/test number reported -- the raw corpus
    has 114/120 (en_bn) and 74/53 (en_hi) such collisions.
    """
    eval_sources = {src for src, _ in eval_rows}
    kept = []
    removed = 0
    for src, tgt in train_rows:
        if src in eval_sources:
            removed += 1
            continue
        kept.append((src, tgt))
    stats[stats_key]["dropped_leaked_into_eval"] = removed
    return kept


def build_translation_memory(sources_to_targets):
    """Collapse {src: Counter(tgt)} to {src: most common tgt}."""
    return {
        src: counter.most_common(1)[0][0] for src, counter in sources_to_targets.items()
    }


def main():
    ap = argparse.ArgumentParser(description="Clean and filter the MT corpora.")
    ap.add_argument(
        "--fold-bengali-danda",
        action="store_true",
        help="Also map Bengali U+09F7 to U+0964. Off by default: 26%% of Bengali "
        "targets end in U+09F7 and the hidden references probably keep it. "
        "Worth an A/B on dev.",
    )
    ap.add_argument(
        "--no-lid",
        action="store_true",
        help="Skip the char n-gram language check (script ratio only).",
    )
    ap.add_argument("--lid-model", default=os.path.join(DICTS_DIR, "lid_charngram.json"))
    args = ap.parse_args()

    lid = None
    if not args.no_lid:
        if not os.path.exists(args.lid_model):
            sys.exit(f"LID model not found at {args.lid_model}; run `python src/lid.py` first")
        lid = CharNGramLID.load(args.lid_model)
        print(f"loaded LID model from {args.lid_model}")

    stats = {}
    cleaned = {}          # (pair_dir, split) -> rows
    tm_train = collections.defaultdict(collections.Counter)
    tm_all = collections.defaultdict(collections.Counter)

    for pair_dir, lang in PAIRS.items():
        per_split = {}
        for split in SPLITS:
            per_split[split] = load_split(
                pair_dir, split, lang, lid, args.fold_bengali_danda, stats
            )
            print(
                f"[{pair_dir}/{split}] read={stats[f'{pair_dir}/{split}']['read']} "
                f"kept={len(per_split[split])}"
            )

        per_split["train"] = deleak(
            per_split["train"],
            per_split["dev"] + per_split["test"],
            f"{pair_dir}/train",
            stats,
        )
        stats[f"{pair_dir}/train"]["kept"] = len(per_split["train"])
        print(f"[{pair_dir}/train] after de-leak: {len(per_split['train'])}")

        for split in SPLITS:
            rows = per_split[split]
            cleaned[(pair_dir, split)] = rows
            write_tsv(os.path.join(CLEAN_DIR, pair_dir, f"{split}.tsv"), rows)
            for src, tgt in rows:
                tm_all[(lang, src)][tgt] += 1
                if split == "train":
                    tm_train[(lang, src)][tgt] += 1

    # ---- combined multilingual files, source carrying its target-language tag
    for split in SPLITS:
        combined = []
        for pair_dir, lang in PAIRS.items():
            for src, tgt in cleaned[(pair_dir, split)]:
                combined.append((tag_source(src, lang), tgt))
        write_tsv(os.path.join(CLEAN_DIR, "combined", f"{split}.tsv"), combined)
        print(f"[combined/{split}] {len(combined)} pairs")

    # ---- SentencePiece training text: English sources + both target languages
    spm_path = os.path.join(CLEAN_DIR, "spm_input.txt")
    n_lines = 0
    with open(spm_path, "w", encoding="utf-8") as fh:
        for pair_dir, _lang in PAIRS.items():
            for src, tgt in cleaned[(pair_dir, "train")]:
                fh.write(src + "\n")
                fh.write(tgt + "\n")
                n_lines += 2
    print(f"[spm] wrote {n_lines} lines -> {spm_path}")

    # ---- translation memories (keys are "lang\tsource" so JSON stays flat)
    os.makedirs(DICTS_DIR, exist_ok=True)
    for name, table in (("tm_train", tm_train), ("tm_all", tm_all)):
        flat = {f"{lang}\t{src}": tgt for (lang, src), tgt in
                build_translation_memory(table).items()}
        path = os.path.join(DICTS_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(flat, fh, ensure_ascii=False)
        print(f"[tm] {name}: {len(flat)} entries -> {path}")

    # ---- how much of the hidden validation set the TM already covers
    coverage = {}
    if os.path.exists(VAL_JSON):
        with open(VAL_JSON, encoding="utf-8") as fh:
            val = json.load(fh)
        tm_all_flat = build_translation_memory(tm_all)
        for section, lang in VAL_SECTIONS.items():
            records = val[section]["Validation"]
            hit = sum(
                1 for rec in records.values()
                if (lang, normalise(rec["source"], lang=None)) in tm_all_flat
            )
            coverage[section] = {
                "records": len(records),
                "tm_hits": hit,
                "pct": round(100 * hit / len(records), 2),
            }
            print(f"[tm] {section}: {hit}/{len(records)} ({coverage[section]['pct']}%) exact hits")

    report = {
        "config": {
            "fold_bengali_danda": args.fold_bengali_danda,
            "lid_enabled": lid is not None,
            "min_script_ratio": MIN_SCRIPT_RATIO,
            "max_len_ratio": MAX_LEN_RATIO,
        },
        "per_file": {k: dict(v) for k, v in stats.items()},
        "val_tm_coverage": coverage,
    }
    report_path = os.path.join(CLEAN_DIR, "report.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(f"\nreport -> {report_path}")


if __name__ == "__main__":
    main()

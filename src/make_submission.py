"""Build `answer.csv` for the Codabench scorer.

Format, reconstructed from the notebook that produced a file the grader
accepted plus the parsing in `~/Downloads/ROUGE/rougescore.py`:

    csv.writer(f, delimiter="\\t", quoting=csv.QUOTE_ALL)
    w.writerow(["ID", "Translation"])

So despite the `.csv` extension the file is TAB-delimited with every field
quoted, and the Bengali block (ids 147532-157367) precedes the Hindi block
(505511-517053). All 21,379 ids must be present.

Applies the exact-match translation memory built by data_prep.py before
writing: ~5% of the validation sources appear verbatim in the training data,
and a memorised reference beats anything the model will produce for them.
"""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    DICTS_DIR,
    ROOT,
    SUBMISSION_DIR,
    VAL_JSON,
    VAL_SECTIONS,
    normalise,
)


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_tm(path):
    if not path or not os.path.exists(resolve(path)):
        return {}
    with open(resolve(path), encoding="utf-8") as fh:
        flat = json.load(fh)
    tm = {}
    for key, tgt in flat.items():
        lang, _, src = key.partition("\t")
        tm[(lang, src)] = tgt
    return tm


def main():
    ap = argparse.ArgumentParser(description="Write submission/answer.csv.")
    ap.add_argument("--hyps", required=True,
                    help="JSON from `decode.py --val-json`")
    ap.add_argument("--out", default=os.path.join(SUBMISSION_DIR, "answer.csv"))
    ap.add_argument("--tm", default=os.path.join(DICTS_DIR, "tm_all.json"),
                    help="translation memory; pass '' to disable")
    args = ap.parse_args()

    with open(VAL_JSON, encoding="utf-8") as fh:
        val = json.load(fh)
    with open(resolve(args.hyps), encoding="utf-8") as fh:
        hyps = json.load(fh)

    tm = load_tm(args.tm)
    if tm:
        print(f"translation memory: {len(tm)} entries")

    rows = []
    missing = []
    tm_hits = 0
    # Bengali first, then Hindi -- matching the accepted submission's order.
    for section in ("English-Bengali", "English-Hindi"):
        lang = VAL_SECTIONS[section]
        records = val[section]["Validation"]
        section_hyps = hyps.get(section, {})
        for rec_id, rec in records.items():
            src = normalise(rec["source"])
            text = tm.get((lang, src))
            if text is not None:
                tm_hits += 1
            else:
                text = section_hyps.get(rec_id)
                if text is None:
                    missing.append((section, rec_id))
                    text = ""
            rows.append((rec_id, " ".join(str(text).split())))
        print(f"  {section}: {len(records)} rows")

    expected = sum(len(val[s]["Validation"]) for s in VAL_SECTIONS)
    assert len(rows) == expected, f"row count {len(rows)} != {expected}"

    got_ids = {r[0] for r in rows}
    want_ids = {i for s in VAL_SECTIONS for i in val[s]["Validation"]}
    assert got_ids == want_ids, (
        f"id mismatch: {len(want_ids - got_ids)} missing, {len(got_ids - want_ids)} extra"
    )

    print(f"translation-memory hits: {tm_hits}/{len(rows)} "
          f"({100 * tm_hits / len(rows):.2f}%)")
    if missing:
        print(f"WARNING: {len(missing)} ids had no hypothesis and were written empty; "
              f"first few: {missing[:5]}")
    empty = sum(1 for _, t in rows if not t.strip())
    if empty:
        print(f"WARNING: {empty} rows are empty -- these score zero")

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", quoting=csv.QUOTE_ALL)
        w.writerow(["ID", "Translation"])
        w.writerows(rows)

    with open(out, encoding="utf-8") as fh:
        n_lines = sum(1 for _ in fh)
    print(f"\nwrote {out}")
    print(f"  {n_lines} lines ({expected} rows + 1 header)")
    assert n_lines == expected + 1, f"line count {n_lines} != {expected + 1}"


if __name__ == "__main__":
    main()

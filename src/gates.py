"""Pre-training verification gates.

Every previous attempt on this data trained to completion and produced garbage:
one emitted almost nothing but <unk> (a token-id mismatch), another collapsed
its vocabulary to ~2.5k pieces and drove the loss to 0.005 by learning only
PAD. Both were silent for hours of GPU time. These gates make those failure
modes loud and immediate.

    G1  tokenizer round-trips real corpus text
    G2  special ids agree across tokenizer / common.py / dataset
    G3  UNK rate is negligible on train, dev, and the hidden val sources
    G4  the model can overfit 64 sentence pairs to near-zero loss and
        reproduce them under greedy decoding          <-- the decisive one
    G5  the decode probe path runs and emits non-degenerate text
    G6  the scorer returns ~1.0 when scoring references against themselves

G1-G3 and G6 need only sentencepiece; G4-G5 need torch. Run what you can
locally, then run the whole thing once on the GPU before the real job starts.

    python src/gates.py              # all available gates
    python src/gates.py --skip-torch # tokenizer + scorer only
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    BOS_ID,
    CLEAN_DIR,
    EOS_ID,
    LANG_TAG,
    PAD_ID,
    ROOT,
    SPM_DIR,
    UNK_ID,
    VAL_JSON,
    VAL_SECTIONS,
    normalise,
    read_tsv,
)

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"

results = []


def record(name, ok, detail=""):
    status = PASS if ok else FAIL
    results.append((name, status, detail))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def skip(name, why):
    results.append((name, SKIP, why))
    print(f"[{SKIP}] {name} -- {why}")


# --------------------------------------------------------------------------


def gate1_roundtrip(sp, n=1000, seed=7):
    """decode(encode(x)) == x on real corpus lines."""
    rng = random.Random(seed)
    ok_all = True
    for pair_dir in ("en_bn", "en_hi"):
        path = os.path.join(CLEAN_DIR, pair_dir, "train.tsv")
        rows = list(read_tsv(path))
        sample = rng.sample(rows, min(n, len(rows)))

        texts = [s for s, _ in sample] + [t for _, t in sample]
        exact = sum(1 for x in texts if sp.decode(sp.encode(x, out_type=int)) == x)
        rate = exact / len(texts)
        ok_all &= record(
            f"G1 round-trip {pair_dir}",
            rate > 0.99,
            f"{exact}/{len(texts)} exact = {rate:.4f}",
        )
    return ok_all


def gate2_ids(sp, expected_vocab=16000):
    ok = True
    ok &= record("G2 pad_id", sp.pad_id() == PAD_ID, f"{sp.pad_id()} (expect {PAD_ID})")
    ok &= record("G2 unk_id", sp.unk_id() == UNK_ID, f"{sp.unk_id()} (expect {UNK_ID})")
    ok &= record("G2 bos_id", sp.bos_id() == BOS_ID, f"{sp.bos_id()} (expect {BOS_ID})")
    ok &= record("G2 eos_id", sp.eos_id() == EOS_ID, f"{sp.eos_id()} (expect {EOS_ID})")
    ok &= record(
        "G2 vocab size",
        sp.get_piece_size() == expected_vocab,
        f"{sp.get_piece_size()} (expect {expected_vocab})",
    )
    for lang, tag in LANG_TAG.items():
        tid = sp.piece_to_id(tag)
        ok &= record(
            f"G2 tag {tag}",
            tid != sp.unk_id() and sp.id_to_piece(tid) == tag,
            f"id={tid}",
        )
    return ok


def gate3_unk_rate(sp, threshold=0.005, sample=5000, seed=11):
    rng = random.Random(seed)
    ok = True

    for split in ("train", "dev"):
        path = os.path.join(CLEAN_DIR, "combined", f"{split}.tsv")
        rows = list(read_tsv(path))
        rows = rng.sample(rows, min(sample, len(rows)))
        texts = [s.split(" ", 1)[1] for s, _ in rows] + [t for _, t in rows]
        ids = sp.encode(texts, out_type=int)
        total = sum(len(x) for x in ids)
        unks = sum(sum(1 for i in x if i == UNK_ID) for x in ids)
        rate = unks / max(total, 1)
        ok &= record(
            f"G3 unk rate {split}", rate < threshold, f"{unks}/{total} = {rate:.5f}"
        )

    if os.path.exists(VAL_JSON):
        with open(VAL_JSON, encoding="utf-8") as fh:
            val = json.load(fh)
        for section in VAL_SECTIONS:
            srcs = [normalise(r["source"]) for r in val[section]["Validation"].values()]
            srcs = rng.sample(srcs, min(sample, len(srcs)))
            ids = sp.encode(srcs, out_type=int)
            total = sum(len(x) for x in ids)
            unks = sum(sum(1 for i in x if i == UNK_ID) for x in ids)
            rate = unks / max(total, 1)
            ok &= record(
                f"G3 unk rate val/{section}",
                rate < threshold,
                f"{unks}/{total} = {rate:.5f}",
            )
    return ok


def gate6_scorer():
    """Scoring references against themselves must be a perfect score."""
    from score import chrf, corpus_bleu, rouge_l

    path = os.path.join(CLEAN_DIR, "combined", "dev.tsv")
    refs = [t for _, t in list(read_tsv(path))[:500]]

    b = corpus_bleu(refs, refs)
    r = rouge_l(refs, refs)
    c = chrf(refs, refs)
    ok = record("G6 self-BLEU == 1", abs(b - 1.0) < 1e-6, f"{b:.6f}")
    ok &= record("G6 self-ROUGE == 1", abs(r - 1.0) < 1e-3, f"{r:.6f}")
    ok &= record("G6 self-chrF == 1", abs(c - 1.0) < 1e-6, f"{c:.6f}")

    # And a deliberately bad system must not score well.
    shifted = refs[1:] + refs[:1]
    b_bad = corpus_bleu(shifted, refs)
    ok &= record("G6 shifted-BLEU ~ 0", b_bad < 0.05, f"{b_bad:.6f}")
    return ok


def gate4_overfit(sp, n_pairs=64, steps=200, device_name=None):
    """Train on a tiny batch until it is memorised, then greedily decode it.

    If the plumbing between tokenizer, dataset, model, loss and decoder is
    wrong anywhere, this fails in about a minute. Both previous failures would
    have been caught here.
    """
    import torch

    from dataset import collate
    from model import LabelSmoothingLoss, Transformer
    from train import greedy_decode, pick_device, strip_special

    device = pick_device(device_name)
    print(f"     (G4 device: {device})")
    torch.manual_seed(0)

    from dataset import ParallelDataset

    ds = ParallelDataset(os.path.join(CLEAN_DIR, "combined", "train.tsv"), sp, max_len=64)
    # Take a language-balanced slice of short sentences.
    idx_bn = [i for i in range(len(ds)) if ds.langs[i] == "bn" and ds.size(i) <= 24]
    idx_hi = [i for i in range(len(ds)) if ds.langs[i] == "hi" and ds.size(i) <= 24]
    idxs = (idx_bn[: n_pairs // 2] + idx_hi[: n_pairs // 2])[:n_pairs]
    batch = [ds[i] for i in idxs]
    src, tgt_in, tgt_out, _ = collate(batch)
    src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)

    # Small model, no dropout: we *want* it to memorise.
    model = Transformer(
        vocab_size=sp.get_piece_size(),
        d_model=256, n_heads=4, n_enc_layers=2, n_dec_layers=2,
        d_ff=512, dropout=0.0, max_len=64,
    ).to(device)
    criterion = LabelSmoothingLoss(sp.get_piece_size(), smoothing=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.98), eps=1e-9)

    model.train()
    last = None
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        logits = model(src, tgt_in)
        loss_sum, n_tok = criterion(logits, tgt_out)
        loss = loss_sum / n_tok
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach())
        if step % 50 == 0:
            print(f"     step {step:3d}  loss {last:.4f}")

    ok = record("G4 overfit loss -> 0", last < 0.05, f"final loss {last:.4f}")

    ys = greedy_decode(model, src, 64, device)
    exact = 0
    for row, i in zip(ys, idxs):
        hyp = sp.decode(strip_special(row.tolist()))
        if hyp.strip() == ds.raw_tgt[i].strip():
            exact += 1
    ok &= record(
        "G4 greedy reproduces the batch",
        exact >= int(0.9 * len(idxs)),
        f"{exact}/{len(idxs)} exact",
    )

    # Beam search is the path actually used at inference, so it gets the same
    # proof. This also exercises beam reordering and the incremental cache.
    from decode import beam_search, greedy

    beam_ids = beam_search(model, src, sp, beam=5, alpha=0.6, max_len=64,
                           no_repeat_ngram=0, use_cache=True)
    beam_exact = sum(
        1 for ids, i in zip(beam_ids, idxs)
        if sp.decode(ids).strip() == ds.raw_tgt[i].strip()
    )
    ok &= record(
        "G4 beam-5 reproduces the batch",
        beam_exact >= int(0.9 * len(idxs)),
        f"{beam_exact}/{len(idxs)} exact",
    )

    # Cached and uncached decoding must agree exactly.
    a = greedy(model, src[:16], 64, use_cache=True)
    b = greedy(model, src[:16], 64, use_cache=False)
    same = sum(
        1 for x, y in zip(a, b)
        if strip_special(x.tolist()) == strip_special(y.tolist())
    )
    ok &= record("G4 decoder cache matches uncached", same == 16, f"{same}/16 identical")

    # G5: the probe path itself must run and produce non-degenerate text.
    sample = [sp.decode(strip_special(r.tolist())) for r in ys[:8]]
    degenerate = all((not s.strip()) or s.count("<unk>") > 3 for s in sample)
    ok &= record("G5 probe output is non-degenerate", not degenerate,
                 f"e.g. {sample[0][:60]!r}")
    return ok


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Run the pre-training verification gates.")
    ap.add_argument("--spm", default=os.path.join(SPM_DIR, "joint16k.model"))
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overfit-steps", type=int, default=200)
    args = ap.parse_args()

    from tokenizer import load as load_spm

    sp = load_spm(args.spm)

    print("=" * 66)
    gate2_ids(sp, args.vocab_size)
    print("-" * 66)
    gate1_roundtrip(sp)
    print("-" * 66)
    gate3_unk_rate(sp)
    print("-" * 66)
    gate6_scorer()
    print("-" * 66)

    if args.skip_torch:
        skip("G4 overfit-64", "--skip-torch")
        skip("G5 probe", "--skip-torch")
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            skip("G4 overfit-64", "torch not installed (run this on the GPU box)")
            skip("G5 probe", "torch not installed (run this on the GPU box)")
        else:
            gate4_overfit(sp, steps=args.overfit_steps, device_name=args.device)

    print("=" * 66)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    print(f"{n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\nDO NOT START TRAINING. Failing gates:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

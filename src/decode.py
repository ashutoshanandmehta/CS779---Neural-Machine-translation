"""Batched beam-search decoding.

Handles both evaluation (a tagged TSV -> one hypothesis per line, ready for
score.py) and inference (val_data1.json -> a JSON of translations keyed by the
competition ids, ready for make_submission.py).

Supports checkpoint averaging, which reliably buys a little quality for free:
`--average-last 5` loads the last N saved steps and means their weights.

Incremental decoding caches decoder self-attention keys and values. That is
easy to get subtly wrong, so `--verify-cache` asserts that cached greedy
decoding produces byte-identical output to the uncached path.
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import (  # noqa: E402
    BOS_ID,
    EOS_ID,
    LANG_TAG,
    PAD_ID,
    ROOT,
    VAL_JSON,
    VAL_SECTIONS,
    normalise,
)
from model import Transformer  # noqa: E402
from tokenizer import load as load_spm  # noqa: E402


def resolve(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Checkpoint loading
# --------------------------------------------------------------------------


def load_model(paths, device):
    """Load one checkpoint, or average several."""
    blobs = [torch.load(resolve(p), map_location="cpu", weights_only=False) for p in paths]
    cfg = blobs[0]["config"]
    mcfg, dcfg = cfg["model"], cfg["data"]

    state = {k: v.clone().float() for k, v in blobs[0]["model"].items()}
    if len(blobs) > 1:
        for blob in blobs[1:]:
            for k, v in blob["model"].items():
                state[k] += v.float()
        for k in state:
            state[k] /= len(blobs)
        print(f"averaged {len(blobs)} checkpoints")

    sp = load_spm(resolve(dcfg["spm_model"]))
    model = Transformer(
        vocab_size=sp.get_piece_size(),
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_enc_layers=mcfg["n_enc_layers"],
        n_dec_layers=mcfg["n_dec_layers"],
        d_ff=mcfg["d_ff"],
        dropout=0.0,
        max_len=dcfg["max_len"],
        tie_embeddings=mcfg["tie_embeddings"],
    )
    model.load_state_dict(state)
    model.to(device).eval()
    return model, sp, cfg


def find_checkpoints(save_dir, average_last):
    save_dir = resolve(save_dir)
    files = [f for f in os.listdir(save_dir) if f.startswith("step") and f.endswith(".pt")]
    files.sort(key=lambda f: int(f[4:-3]))
    if not files:
        raise FileNotFoundError(f"no step*.pt checkpoints in {save_dir}")
    return [os.path.join(save_dir, f) for f in files[-average_last:]]


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def length_penalty(length, alpha):
    """GNMT length normalisation (Wu et al. 2016, eq. 14)."""
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(model, src, sp, beam=5, alpha=0.6, max_len=128,
                no_repeat_ngram=3, use_cache=True):
    """Return a list of token-id lists, one best hypothesis per input row."""
    device = src.device
    b = src.size(0)
    k = beam
    vocab = model.vocab_size

    memory, mem_mask = model.encode(src)
    # (B, S, D) -> (B*K, S, D)
    memory = memory.unsqueeze(1).expand(-1, k, -1, -1).reshape(b * k, *memory.shape[1:])
    mem_mask = mem_mask.unsqueeze(1).expand(-1, k, -1, -1, -1).reshape(
        b * k, *mem_mask.shape[1:]
    )

    ys = torch.full((b * k, 1), BOS_ID, dtype=torch.long, device=device)
    scores = torch.full((b, k), float("-inf"), device=device)
    scores[:, 0] = 0.0
    scores = scores.view(-1)

    caches = model.new_caches() if use_cache else None
    finished = [[] for _ in range(b)]
    alive = torch.ones(b, dtype=torch.bool, device=device)

    for step in range(max_len - 1):
        if use_cache:
            h = model.decode(ys[:, -1:], memory, mem_mask, offset=step, caches=caches)
        else:
            h = model.decode(ys, memory, mem_mask)
        logits = model.out_proj(h[:, -1]).float()
        logp = torch.log_softmax(logits, dim=-1)

        if no_repeat_ngram and ys.size(1) >= no_repeat_ngram:
            _block_repeats(logp, ys, no_repeat_ngram)

        cand = (scores.unsqueeze(1) + logp).view(b, k * vocab)
        top_scores, top_idx = cand.topk(k, dim=-1)
        beam_idx = top_idx // vocab
        token_idx = top_idx % vocab

        flat_beam = (torch.arange(b, device=device).unsqueeze(1) * k + beam_idx).view(-1)
        ys = torch.cat([ys[flat_beam], token_idx.view(-1, 1)], dim=1)
        scores = top_scores.view(-1)

        if use_cache:
            for layer in caches:
                layer["self"]["k"] = layer["self"]["k"][flat_beam]
                layer["self"]["v"] = layer["self"]["v"][flat_beam]

        # Retire beams that just produced EOS.
        eos = token_idx.view(-1) == EOS_ID
        if bool(eos.any()):
            for pos in eos.nonzero(as_tuple=False).flatten().tolist():
                sent = pos // k
                if len(finished[sent]) < k:
                    norm = float(scores[pos]) / length_penalty(ys.size(1), alpha)
                    finished[sent].append((norm, ys[pos, 1:-1].tolist()))
            scores = scores.masked_fill(eos, float("-inf"))

        for i in range(b):
            if len(finished[i]) >= k:
                alive[i] = False
        if not bool(alive.any()):
            break

    # Anything that never emitted EOS falls back to its best partial beam.
    out = []
    for i in range(b):
        if finished[i]:
            out.append(max(finished[i], key=lambda x: x[0])[1])
        else:
            best = int(scores.view(b, k)[i].argmax())
            row = ys.view(b, k, -1)[i, best, 1:].tolist()
            out.append([t for t in row if t not in (EOS_ID, PAD_ID)])
    return out


def _block_repeats(logp, ys, n):
    """Forbid completing an n-gram that already occurred in the same beam."""
    prefix = ys[:, -(n - 1):]
    for row in range(ys.size(0)):
        seq = ys[row].tolist()
        pref = tuple(prefix[row].tolist())
        banned = {
            seq[i + n - 1]
            for i in range(len(seq) - n + 1)
            if tuple(seq[i : i + n - 1]) == pref
        }
        if banned:
            logp[row, list(banned)] = float("-inf")


@torch.no_grad()
def greedy(model, src, max_len=128, use_cache=True):
    device = src.device
    b = src.size(0)
    memory, mem_mask = model.encode(src)
    ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    caches = model.new_caches() if use_cache else None

    for step in range(max_len - 1):
        if use_cache:
            h = model.decode(ys[:, -1:], memory, mem_mask, offset=step, caches=caches)
        else:
            h = model.decode(ys, memory, mem_mask)
        nxt = model.out_proj(h[:, -1]).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == EOS_ID)
        if bool(done.all()):
            break
    return ys


def ids_to_text(sp, ids):
    clean = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i not in (BOS_ID, PAD_ID):
            clean.append(i)
    return sp.decode(clean)


# --------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------


def encode_sources(sp, items, max_len):
    """items: list of (lang, source_text) -> list of id lists."""
    tag_ids = {lang: sp.piece_to_id(tok) for lang, tok in LANG_TAG.items()}
    encoded = sp.encode([t for _, t in items], out_type=int)
    return [
        [tag_ids[lang]] + ids[: max_len - 2] + [EOS_ID]
        for (lang, _), ids in zip(items, encoded)
    ]


def read_tsv_items(path):
    items = []
    with open(resolve(path), encoding="utf-8") as fh:
        for line in fh:
            tagged, _tgt = line.rstrip("\n").split("\t")
            tag, _, rest = tagged.partition(" ")
            lang = next(l for l, t in LANG_TAG.items() if t == tag)
            items.append((lang, rest))
    return items


def read_val_items(path):
    with open(resolve(path), encoding="utf-8") as fh:
        val = json.load(fh)
    items, keys = [], []
    for section, lang in VAL_SECTIONS.items():
        for rec_id, rec in val[section]["Validation"].items():
            items.append((lang, normalise(rec["source"])))
            keys.append((section, rec_id))
    return items, keys


def run(model, sp, items, device, batch_size, beam, alpha, max_len, no_repeat, use_cache):
    """Decode in length-sorted batches, then restore the original order."""
    encoded = encode_sources(sp, items, max_len)
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))
    hyps = [None] * len(encoded)

    t0 = time.time()
    for start in range(0, len(order), batch_size):
        chunk = order[start : start + batch_size]
        smax = max(len(encoded[i]) for i in chunk)
        src = torch.full((len(chunk), smax), PAD_ID, dtype=torch.long)
        for r, i in enumerate(chunk):
            src[r, : len(encoded[i])] = torch.tensor(encoded[i])
        src = src.to(device)

        if beam <= 1:
            rows = greedy(model, src, max_len, use_cache)
            outs = [ids_to_text(sp, r.tolist()) for r in rows]
        else:
            outs = [
                sp.decode(ids)
                for ids in beam_search(model, src, sp, beam, alpha, max_len,
                                       no_repeat, use_cache)
            ]
        for i, text in zip(chunk, outs):
            hyps[i] = text

        done = start + len(chunk)
        if done % (batch_size * 20) == 0 or done >= len(order):
            el = time.time() - t0
            rate = done / max(el, 1e-9)
            print(f"  {done}/{len(order)}  {el / 60:.1f}m  "
                  f"({rate:.1f}/s, eta {(len(order) - done) / max(rate, 1e-9) / 60:.1f}m)")
    return hyps


def verify_cache(model, sp, items, device, max_len):
    """Cached and uncached greedy decoding must agree exactly."""
    encoded = encode_sources(sp, items[:16], max_len)
    smax = max(len(e) for e in encoded)
    src = torch.full((len(encoded), smax), PAD_ID, dtype=torch.long)
    for r, e in enumerate(encoded):
        src[r, : len(e)] = torch.tensor(e)
    src = src.to(device)

    a = [ids_to_text(sp, r.tolist()) for r in greedy(model, src, max_len, use_cache=True)]
    b = [ids_to_text(sp, r.tolist()) for r in greedy(model, src, max_len, use_cache=False)]
    bad = [(x, y) for x, y in zip(a, b) if x != y]
    if bad:
        print(f"[FAIL] cache verification: {len(bad)}/{len(a)} mismatched")
        for x, y in bad[:3]:
            print(f"   cached  : {x[:90]}")
            print(f"   uncached: {y[:90]}")
        return False
    print(f"[PASS] cache verification: {len(a)}/{len(a)} identical")
    return True


def main():
    ap = argparse.ArgumentParser(description="Decode with beam search.")
    ap.add_argument("--checkpoint", nargs="*", default=None)
    ap.add_argument("--save-dir", default="outs")
    ap.add_argument("--average-last", type=int, default=0,
                    help="average the last N step checkpoints in --save-dir")
    ap.add_argument("--input", default=None, help="tagged TSV to translate")
    ap.add_argument("--val-json", action="store_true", help="translate val_data1.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beam", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--verify-cache", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")

    if args.checkpoint:
        paths = args.checkpoint
    elif args.average_last:
        paths = find_checkpoints(args.save_dir, args.average_last)
        print(f"averaging: {[os.path.basename(p) for p in paths]}")
    else:
        paths = [os.path.join(resolve(args.save_dir), "best.pt")]
    model, sp, cfg = load_model(paths, device)

    if args.val_json:
        items, keys = read_val_items(VAL_JSON)
    elif args.input:
        items, keys = read_tsv_items(args.input), None
    else:
        sys.exit("give --input or --val-json")
    if args.limit:
        items = items[: args.limit]
        keys = keys[: args.limit] if keys else None
    print(f"{len(items)} sentences to translate")

    if args.verify_cache and not verify_cache(model, sp, items, device, args.max_len):
        sys.exit(1)

    hyps = run(model, sp, items, device, args.batch_size, args.beam, args.alpha,
               args.max_len, args.no_repeat_ngram, not args.no_cache)

    out = resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if keys is not None:
        blob = {}
        for (section, rec_id), text in zip(keys, hyps):
            blob.setdefault(section, {})[rec_id] = text
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=1)
    else:
        with open(out, "w", encoding="utf-8") as fh:
            for h in hyps:
                fh.write(h.replace("\n", " ") + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

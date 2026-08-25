"""Training loop.

Designed to survive free-tier session limits: full optimiser state is written
every eval, and `--resume` picks up exactly where it left off.

Two habits here exist specifically because the earlier attempts on this data
produced fluent-looking loss curves alongside degenerate output:

  * a decode probe runs at every evaluation and prints real translations, so a
    collapsed model is visible within minutes rather than after a full run;
  * loss is normalised per *token*, not per batch, which is the only
    meaningful number when batch sentence-count varies with a token budget.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import BOS_ID, EOS_ID, PAD_ID, ROOT  # noqa: E402
from dataset import ParallelDataset, make_loader  # noqa: E402
from model import LabelSmoothingLoss, Transformer  # noqa: E402
from tokenizer import load as load_spm  # noqa: E402


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def load_config(path):
    with open(resolve(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def noam_lr(step, d_model, warmup, scale=1.0):
    """Vaswani et al. schedule: linear warmup then inverse-sqrt decay."""
    step = max(step, 1)
    return scale * (d_model ** -0.5) * min(step ** -0.5, step * warmup ** -1.5)


@torch.no_grad()
def greedy_decode(model, src, max_len, device):
    """Small greedy decoder used only for the training-time probe."""
    model.eval()
    src = src.to(device)
    memory, mem_mask = model.encode(src)
    b = src.size(0)
    ys = torch.full((b, 1), BOS_ID, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        h = model.decode(ys, memory, mem_mask)
        nxt = model.out_proj(h[:, -1]).argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
        ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == EOS_ID)
        if bool(done.all()):
            break
    return ys


def strip_special(ids):
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i not in (BOS_ID, PAD_ID):
            out.append(i)
    return out


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total, tokens = 0.0, 0
    for src, tgt_in, tgt_out, _ in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in)
        loss, n = criterion(logits, tgt_out)
        total += float(loss)
        tokens += n
    return total / max(tokens, 1)


@torch.no_grad()
def probe(model, dataset, sp, device, n=5, max_len=128):
    """G5: print a handful of real translations. Guards against silent collapse."""
    from dataset import collate

    idxs = [i * max(len(dataset) // max(n, 1), 1) for i in range(n)]
    idxs = [i for i in idxs if i < len(dataset)][:n]
    batch = [dataset[i] for i in idxs]
    src, _, _, _ = collate(batch)
    ys = greedy_decode(model, src, max_len, device)

    print("  --- decode probe ---")
    for row, idx in zip(ys, idxs):
        hyp = sp.decode(strip_special(row.tolist()))
        ref = dataset.raw_tgt[idx]
        print(f"   [{dataset.langs[idx]}] hyp: {hyp[:110]}")
        print(f"        ref: {ref[:110]}")
    print("  --------------------")


def save_checkpoint(path, model, optimizer, scaler, step, epoch, best, cfg):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "step": step,
            "epoch": epoch,
            "best_dev_loss": best,
            "config": cfg,
        },
        path,
    )


def prune_checkpoints(save_dir, keep_last):
    ckpts = sorted(
        (f for f in os.listdir(save_dir) if f.startswith("step") and f.endswith(".pt")),
        key=lambda f: int(f[4:-3]),
    )
    for f in ckpts[:-keep_last]:
        os.remove(os.path.join(save_dir, f))


def main():
    ap = argparse.ArgumentParser(description="Train the multilingual transformer.")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--save-dir", default=None, help="overrides checkpoint.save_dir")
    ap.add_argument("--resume", default=None, help="path to a checkpoint, or 'auto'")
    ap.add_argument("--require-resume", action="store_true",
                    help="abort instead of starting from scratch if no checkpoint "
                         "is found. Use this on every restart after the first.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit-train", type=int, default=None,
                    help="truncate the training set (smoke tests)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg, mcfg, tcfg, ecfg, ccfg = (
        cfg["data"], cfg["model"], cfg["train"], cfg["eval"], cfg["checkpoint"]
    )
    save_dir = resolve(args.save_dir or ccfg["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(tcfg["seed"])
    device = pick_device(args.device)
    print(f"device: {device}")

    sp = load_spm(resolve(dcfg["spm_model"]))
    vocab_size = sp.get_piece_size()
    print(f"vocab: {vocab_size}")

    train_ds = ParallelDataset(resolve(dcfg["train"]), sp, dcfg["max_len"])
    dev_ds = ParallelDataset(resolve(dcfg["dev"]), sp, dcfg["max_len"])
    if args.limit_train:
        train_ds.src = train_ds.src[: args.limit_train]
        train_ds.tgt = train_ds.tgt[: args.limit_train]
        train_ds.langs = train_ds.langs[: args.limit_train]
        train_ds.raw_tgt = train_ds.raw_tgt[: args.limit_train]
    print(f"train pairs: {len(train_ds)}   dev pairs: {len(dev_ds)}")

    train_loader, sampler = make_loader(
        train_ds, tcfg["max_tokens"], shuffle=True,
        seed=tcfg["seed"], num_workers=tcfg["num_workers"],
    )
    dev_loader, _ = make_loader(
        dev_ds, tcfg["max_tokens"], shuffle=False, num_workers=0
    )

    model = Transformer(
        vocab_size=vocab_size,
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_enc_layers=mcfg["n_enc_layers"],
        n_dec_layers=mcfg["n_dec_layers"],
        d_ff=mcfg["d_ff"],
        dropout=mcfg["dropout"],
        max_len=dcfg["max_len"],
        tie_embeddings=mcfg["tie_embeddings"],
    ).to(device)
    print(f"parameters: {model.count_parameters() / 1e6:.1f}M")

    criterion = LabelSmoothingLoss(vocab_size, tcfg["label_smoothing"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # actual value comes from the Noam schedule each step
        betas=tuple(tcfg["betas"]),
        eps=tcfg["eps"],
        weight_decay=tcfg["weight_decay"],
    )
    use_amp = bool(tcfg["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    step, epoch, best = 0, 0, float("inf")
    since_improve = 0

    # Resuming is resolved loudly. Silently falling back to step 0 because a
    # path was wrong costs hours of GPU time and is invisible in the log until
    # you notice the step counter starting from 100 again.
    resume_path = args.resume
    if resume_path == "auto":
        cands = [f for f in os.listdir(save_dir) if f.startswith("step") and f.endswith(".pt")]
        if cands:
            resume_path = os.path.join(save_dir, max(cands, key=lambda f: int(f[4:-3])))
        else:
            listing = sorted(os.listdir(save_dir))
            print("!" * 70)
            print(f"!! --resume auto found NO step*.pt checkpoints in:\n!!   {save_dir}")
            print(f"!! that directory contains: {listing[:15] if listing else '(empty)'}")
            print("!! TRAINING WILL START FROM SCRATCH AT STEP 0.")
            print("!! If you expected to resume, stop now and check the path.")
            print("!" * 70)
            if args.require_resume:
                sys.exit("aborting: --require-resume was set but no checkpoint was found")
            resume_path = None
    elif resume_path and not os.path.exists(resume_path):
        # An explicit path that does not exist is always a mistake.
        sys.exit(f"--resume {resume_path!r} does not exist")

    if resume_path:
        blob = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        if scaler is not None and blob.get("scaler"):
            scaler.load_state_dict(blob["scaler"])
        step, epoch, best = blob["step"], blob["epoch"], blob["best_dev_loss"]
        print(f"RESUMED from {resume_path}\n  step {step}, epoch {epoch}, "
              f"best dev {best:.4f}")
    elif args.require_resume:
        sys.exit("aborting: --require-resume was set but no checkpoint was loaded")

    max_steps = args.max_steps or tcfg["max_steps"]
    d_model = mcfg["d_model"]
    warmup = tcfg["warmup_steps"]
    lr_scale = tcfg["lr_scale"]
    accum = tcfg["accum_steps"]
    history = []

    print(f"training to {max_steps} steps (accum={accum}, "
          f"~{tcfg['max_tokens'] * accum} tokens/update)")
    t0 = time.time()
    running, running_tok = 0.0, 0
    stop = False

    while not stop and step < max_steps:
        sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        micro = 0

        for src, tgt_in, tgt_out, _ in train_loader:
            src = src.to(device, non_blocking=True)
            tgt_in = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(src, tgt_in)
            loss_sum, n_tok = criterion(logits, tgt_out)
            loss = loss_sum / max(n_tok, 1) / accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss_sum.detach())
            running_tok += n_tok
            micro += 1

            if micro % accum != 0:
                continue

            step += 1
            lr = noam_lr(step, d_model, warmup, lr_scale)
            for g in optimizer.param_groups:
                g["lr"] = lr

            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["clip_norm"])
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % 100 == 0:
                tl = running / max(running_tok, 1)
                el = time.time() - t0
                print(f"step {step:6d} | train {tl:.4f} | ppl {math.exp(min(tl, 20)):8.2f} "
                      f"| lr {lr:.2e} | {el / 60:.1f}m")
                running, running_tok = 0.0, 0

            if step % ecfg["every_steps"] == 0 or step >= max_steps:
                dev_loss = evaluate(model, dev_loader, criterion, device)
                improved = dev_loss < best
                print(f"step {step:6d} | DEV {dev_loss:.4f} | ppl "
                      f"{math.exp(min(dev_loss, 20)):.2f}{'  *best*' if improved else ''}")
                probe(model, dev_ds, sp, device, ecfg["probe_sentences"], dcfg["max_len"])
                history.append({"step": step, "dev_loss": dev_loss})

                save_checkpoint(os.path.join(save_dir, f"step{step}.pt"),
                                model, optimizer, scaler, step, epoch, min(best, dev_loss), cfg)
                if improved:
                    best = dev_loss
                    since_improve = 0
                    save_checkpoint(os.path.join(save_dir, "best.pt"), model, optimizer,
                                    scaler, step, epoch, best, cfg)
                else:
                    since_improve += 1
                prune_checkpoints(save_dir, ccfg["keep_last"])

                with open(os.path.join(save_dir, "history.json"), "w") as fh:
                    json.dump(history, fh, indent=2)

                model.train()
                if since_improve >= ecfg["patience"]:
                    print(f"early stop: {since_improve} evals without improvement")
                    stop = True
                    break
                if step >= max_steps:
                    stop = True
                    break

        epoch += 1

    print(f"done. best dev loss {best:.4f} after {step} steps, "
          f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()

"""Dataset and token-based batching.

Two things here matter for training quality:

  * The target-language tag is injected as a token *id* at position 0 of the
    encoder input, not encoded as text. Encoding "<2hi> ..." as a string makes
    SentencePiece emit its dummy-whitespace piece first, wasting a slot.

  * Batches are built to a token budget with length bucketing rather than a
    fixed sentence count. With sentence lengths spanning 1..128, fixed-size
    batches either waste most of their compute on padding or blow up memory on
    the long tail; a token budget keeps the real work per step roughly constant.
"""

import os
import random
import sys

import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import BOS_ID, EOS_ID, LANG_TAG, PAD_ID, read_tsv  # noqa: E402


def split_tag(tagged_src):
    """'<2hi> some text' -> ('hi', 'some text')."""
    tag, _, rest = tagged_src.partition(" ")
    for lang, t in LANG_TAG.items():
        if tag == t:
            return lang, rest
    raise ValueError(f"source is missing a language tag: {tagged_src[:60]!r}")


class ParallelDataset(Dataset):
    """Encoded parallel corpus.

    Each item is (src_ids, tgt_ids, lang) where
        src_ids = [tag] + source pieces + [eos]
        tgt_ids = [bos] + target pieces + [eos]
    """

    def __init__(self, path, sp, max_len=128, with_target=True):
        self.max_len = max_len
        self.with_target = with_target
        self.sp = sp

        langs, raw_src, raw_tgt = [], [], []
        for tagged_src, tgt in read_tsv(path):
            lang, src = split_tag(tagged_src)
            langs.append(lang)
            raw_src.append(src)
            raw_tgt.append(tgt)

        tag_ids = {lang: sp.piece_to_id(tok) for lang, tok in LANG_TAG.items()}
        enc_src = sp.encode(raw_src, out_type=int)
        enc_tgt = sp.encode(raw_tgt, out_type=int) if with_target else None

        self.langs = langs
        self.src = []
        self.tgt = []
        for i, lang in enumerate(langs):
            s = [tag_ids[lang]] + enc_src[i][: max_len - 2] + [EOS_ID]
            self.src.append(s)
            if with_target:
                t = [BOS_ID] + enc_tgt[i][: max_len - 2] + [EOS_ID]
                self.tgt.append(t)

        self.raw_tgt = raw_tgt

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        tgt = self.tgt[idx] if self.with_target else [BOS_ID, EOS_ID]
        return self.src[idx], tgt, idx

    def size(self, idx):
        """Cost proxy used for bucketing: the longer of the two sides."""
        s = len(self.src[idx])
        t = len(self.tgt[idx]) if self.with_target else s
        return max(s, t)


def collate(batch):
    """Pad a list of (src_ids, tgt_ids, idx) into tensors."""
    srcs, tgts, idxs = zip(*batch)
    smax = max(len(s) for s in srcs)
    tmax = max(len(t) for t in tgts)

    src = torch.full((len(srcs), smax), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(tgts), tmax), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        tgt[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    # Teacher forcing: feed tgt[:-1], predict tgt[1:]
    return src, tgt[:, :-1].contiguous(), tgt[:, 1:].contiguous(), torch.tensor(idxs)


class TokenBatchSampler(torch.utils.data.Sampler):
    """Yield lists of indices whose padded token count stays under a budget.

    Indices are sorted by length (with a little noise so epochs differ), packed
    greedily, then the resulting batches are shuffled so the model does not see
    all the short sentences first.
    """

    def __init__(self, dataset, max_tokens=8000, shuffle=True, seed=1, noise=8):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.noise = noise
        self.epoch = 0
        self._batches = None

    def _build(self):
        rng = random.Random(self.seed + self.epoch)
        n = len(self.dataset)
        order = list(range(n))
        if self.shuffle:
            rng.shuffle(order)
            order.sort(key=lambda i: self.dataset.size(i) + rng.randint(0, self.noise))
        else:
            order.sort(key=self.dataset.size)

        batches, cur, cur_max = [], [], 0
        for i in order:
            L = self.dataset.size(i)
            new_max = max(cur_max, L)
            if cur and new_max * (len(cur) + 1) > self.max_tokens:
                batches.append(cur)
                cur, cur_max = [i], L
            else:
                cur.append(i)
                cur_max = new_max
        if cur:
            batches.append(cur)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch):
        self.epoch = epoch
        self._batches = None

    def __iter__(self):
        if self._batches is None:
            self._batches = self._build()
        return iter(self._batches)

    def __len__(self):
        if self._batches is None:
            self._batches = self._build()
        return len(self._batches)


def make_loader(dataset, max_tokens=8000, shuffle=True, seed=1, num_workers=2):
    sampler = TokenBatchSampler(dataset, max_tokens, shuffle=shuffle, seed=seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, sampler

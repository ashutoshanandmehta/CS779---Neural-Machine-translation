# English → {Hindi, Bengali} — Transformer from scratch

A single multilingual encoder–decoder translating English into Hindi and Bengali. The
Transformer is written from scratch in PyTorch (`src/model.py`): explicit multi-head
attention, sinusoidal positions, pre-norm residual blocks. No pretrained weights, no
`nn.Transformer`, no `nn.MultiheadAttention`, no HuggingFace.

## Design: one model, not a router

The target language is selected by a **tag token** (`<2hi>` / `<2bn>`) prepended to the
encoder input — Johnson et al. 2017, [arXiv:1611.04558](https://arxiv.org/abs/1611.04558).

A language-detection router was considered and rejected. The source side of this corpus is
always English, so a detector reading the input carries no information about whether Hindi
or Bengali is wanted; that is target-side metadata, and `val_data1.json` already supplies
it as a top-level key. Beyond that, with only ~140k pairs the two directions are better off
sharing an encoder: English gets twice the data, and Hindi and Bengali share enough
Sanskrit-derived (*tatsama*) vocabulary for the decoder to transfer as well. Separate models
would only win at a data scale where capacity dilution starts to bite.

Language identification is still used, but as a **data-cleaning tool** (`src/lid.py`) — a
character n-gram Naive Bayes classifier, 99.9% holdout accuracy, that flags rows whose
target is not in the language the file claims.

## Layout

```
src/common.py           paths, special ids, Unicode normalisation
src/lid.py              char n-gram language ID (data QA)
src/data_prep.py        clean / filter / de-leak / translation memory
src/tokenizer.py        joint SentencePiece over en+hi+bn
src/dataset.py          encoding + token-budget batching with length bucketing
src/model.py            the Transformer
src/train.py            Noam schedule, label smoothing, AMP, resumable checkpoints
src/decode.py           batched beam search, incremental cache, checkpoint averaging
src/score.py            local replica of the grader's BLEU / ROUGE-L / chrF++
src/gates.py            pre-training verification gates
src/make_submission.py  writes submission/answer.csv
configs/base.yaml       the real run;  configs/smoke.yaml  CPU plumbing test
colab_train.ipynb       the full executed run, with outputs
```

## Running it

```bash
python src/lid.py            # ~8s   char n-gram LID, writes dicts/lid_charngram.json
python src/data_prep.py      # ~30s  writes data/clean/, dicts/tm_*.json, report.json
python src/tokenizer.py      # ~60s  writes spm/joint16k.model
python src/gates.py          # ~2m   MUST be all-green before training
```

Then, on a GPU (see the notebook):

```bash
python src/train.py --config configs/base.yaml --save-dir outs/base --resume auto
python src/decode.py --save-dir outs/base --average-last 5 \
    --input data/clean/combined/dev.tsv --out outs/dev.hyp --beam 5 --verify-cache
python src/score.py --hyp outs/dev.hyp --ref data/clean/combined/dev.tsv
python src/decode.py --save-dir outs/base --average-last 5 \
    --val-json --out outs/val_hyps.json --beam 5
python src/make_submission.py --hyps outs/val_hyps.json
```

`--resume auto` continues from the newest checkpoint, so a session timeout costs at most
one eval interval.

## The gates

Every earlier attempt on this data trained to completion and produced garbage — one emitted
almost nothing but `<unk>` (a token-id mismatch), another collapsed its vocabulary to ~2.5k
pieces and drove the loss to 0.005 by learning only `PAD`. Both were silent for hours of GPU
time. `src/gates.py` makes those failures loud and immediate:

| Gate | Checks |
|---|---|
| G1 | `decode(encode(x)) == x` on real corpus lines |
| G2 | special ids agree across tokenizer, `common.py` and dataset |
| G3 | UNK rate on train, dev **and** the hidden validation sources |
| G4 | a small model overfits 64 pairs to ~0 loss and **greedy + beam-5 reproduce them exactly**; cached and uncached decoding agree |
| G5 | the decode probe emits non-degenerate text |
| G6 | scoring references against themselves gives 1.0; a shifted system gives ~0 |

Current status: **22/22 pass**.

## Data preparation

`data/clean/report.json` records every filter's count. Notable fixes:

* **670** Hindi targets carried a scraped `<digits>: ` line-number prefix.
* **4,528** Bengali targets used `׀` (U+05C0, Hebrew paseq) as a danda; **1,153** Hindi
  targets used ASCII `|`. Both normalised to `।`.
* U+FEFF and U+200B stripped; **ZWNJ (U+200C) and ZWJ (U+200D) deliberately preserved** —
  they are meaningful in Devanagari/Bengali conjuncts.
* Wrong-script targets dropped (303), length-ratio outliers (545), source-equals-target
  (17), exact duplicates (936, train only).
* **433 training rows removed for leaking into dev/test.** Left in, they inflate every
  number you report.

Tokenizer: joint SentencePiece unigram, 16k, `character_coverage=1.0` with byte fallback.
At the usual 0.9995 the tokenizer silently mapped `&`, `Q`, `Z`, `₹` and curly quotes to
`<unk>` — all of which occur in the validation sources. The corpus has only 410 distinct
characters, so full coverage costs 2.6% of the vocabulary and takes the UNK rate to zero.

**Translation memory.** ~5.1% (1,097 / 21,379) of validation sources appear verbatim in the
training data. `make_submission.py` emits the known reference for those.

## Evaluation

The grader (`evaluate.py`) sums three metrics and does **not** use sacrebleu — its BLEU is
`nltk.corpus_bleu` over `word_tokenize` output, uniform weights, no smoothing, on a 0–1
scale. That is far harsher than sacrebleu's 0–100, so `src/score.py` follows the grader's
semantics instead. It is pure stdlib; `--cross-check` compares against sacrebleu/nltk when
they are installed.

Caveat: `chrfscore.py` is not present anywhere in the project, so the chrF here follows
sacrebleu's chrF++ definition (char n-grams to 6, word n-grams to 2, β=2). It is a
reconstruction, not a verified match.

`val_data1.json` has no references, so all measurement happens on `data/clean/combined/dev.tsv`.

## Submission format

`submission/answer.csv` — despite the extension it is **tab-delimited with every field
quoted**, header `"ID"\t"Translation"`, Bengali block (147532–157367) before Hindi
(505511–517053), all 21,379 rows. Reconstructed from the notebook that produced an accepted
file plus the parsing in the grader's `rougescore.py`.

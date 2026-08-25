# English to Hindi and Bengali Neural Machine Translation

Ashutosh Anand · CS779

One multilingual Transformer translating English into Hindi and Bengali. Written from
scratch in PyTorch. No pretrained weights, no `nn.Transformer`, no `nn.MultiheadAttention`,
no HuggingFace.

## Result

Measured on 3,728 held out sentences.

| | BLEU | chrF++ |
|---|---|---|
| **Overall** | **27.5** | **50.1** |
| Hindi | 32.6 | 53.2 |
| Bengali | 18.6 | 45.9 |

Cross checked against sacrebleu on the overall set: BLEU 27.02 and chrF++ 50.53.

Hindi scores well above Bengali. Bengali has less data after cleaning, 64,359 pairs against
75,607, and richer morphology, so more of its word forms are unseen at test time. This is
the clearest limit on the current model.

On the course grader scale, which sums three metrics between 0 and 1: BLEU 0.275,
ROUGE-L 0.538, chrF 0.501, total 1.31.

## Data

140k sentence pairs, 139,966 after cleaning. Every filter count is written to
`data/clean/report.json`.

* 433 training rows leaked into the dev and test splits. Left in they inflate every
  number reported.
* 4,528 Bengali targets used the Hebrew paseq `׀` as a danda. 1,153 Hindi targets used an
  ASCII pipe. Both normalised to `।`.
* 670 Hindi targets carried a scraped line number prefix.
* 303 rows had a target in the wrong script. 545 were length ratio outliers. 936 were
  exact duplicates.
* U+FEFF and U+200B stripped. ZWNJ and ZWJ preserved, because they carry meaning in
  Devanagari and Bengali conjuncts.

Wrong script detection uses a character n-gram Naive Bayes classifier at 99.9% holdout
accuracy. It is a data QA tool and not part of the model.

**Tokenizer.** Joint SentencePiece unigram, 16k, shared across all three languages, with
`character_coverage` at 1.0 and byte fallback. At the usual 0.9995 the tokenizer silently
mapped `&`, `Q`, `Z`, `₹` and curly quotes to unknown, all of which occur in the validation
sources. The corpus has only 410 distinct characters, so full coverage costs 2.6% of the
vocabulary and takes the UNK rate to zero on every split.

## Model

52.3M parameters.

| | |
|---|---|
| Layers | 6 encoder, 6 decoder |
| d_model | 512 |
| Attention heads | 8, d_head 64 |
| Feed forward | 2048, ReLU |
| Dropout | 0.3 |
| Positions | Fixed sinusoidal |
| Normalisation | Before each sublayer, plus a final norm on each stack |
| Vocabulary | 16k joint SentencePiece |

Attention is written out in `src/model.py`. The query, key and value projections, the
scaled dot product, the softmax and the head merge are all explicit.

Two departures from Vaswani et al. 2017:

* Normalisation runs before each sublayer rather than after. Post norm at this depth needs
  a carefully tuned warmup to avoid diverging early. Pre norm trains stably across a much
  wider range of settings.
* Source embedding, target embedding and output projection are tied into one matrix. With
  a joint vocabulary this is what shares representations across the two directions, and it
  removes 16M parameters that 140k pairs cannot support.

## One model, not two

The target language is chosen by a tag token, `<2hi>` or `<2bn>`, prepended to the encoder
input. This follows Johnson et al. 2017.

A language detection router was considered and rejected. The source side is always English,
so a detector reading the input says nothing about which target is wanted. That is target
side metadata and the validation file already supplies it.

At 140k pairs the two directions are also better off sharing an encoder. English gets twice
the data, and Hindi and Bengali share enough Sanskrit derived vocabulary for the decoder to
transfer. Separate models would only win at a scale where capacity dilution starts to bite.

## Training

Adam with betas 0.9 and 0.98. Noam schedule with 4,000 warmup steps. Label smoothing 0.1.
Gradient clipping at 1.0. Mixed precision.

Batching is by token budget rather than sentence count, 8,000 padded tokens per micro batch
with 3 accumulation steps for an effective batch near 24k tokens. Sentence lengths span 1 to
128, so fixed size batches would waste most of the padding budget.

One T4 GPU. Early stopping fired at 38,000 steps after 8 evaluations without improvement.
Total 310 minutes. Best dev loss 3.185.

## Decoding

Batched beam search with an incremental key value cache. Batches are length sorted and
restored to the original order afterwards. Length normalisation follows Wu et al. 2016.
Repeated trigrams are blocked. The final model is the average of the last 5 checkpoints.

A sweep over 10 beam and length penalty settings moved the total by 0.013, which is inside
noise, so beam 5 was kept. Throughput is 21,379 sentences in under 10 minutes.

5.13% of validation sources appear verbatim in the training data. The submission emits the
known reference for those.

## Verification

`src/gates.py` runs 22 checks and must be fully green before training starts. All 22 pass.

| Gate | Checks |
|---|---|
| G1 | `decode(encode(x)) == x` on real corpus lines |
| G2 | Special ids agree across tokenizer, `common.py` and dataset |
| G3 | UNK rate on train, dev and the hidden validation sources |
| G4 | A small model overfits 64 pairs to near zero loss, and greedy and beam 5 both reproduce them exactly. Cached and uncached decoding agree |
| G5 | The decode probe emits non degenerate text |
| G6 | Scoring references against themselves gives 1.0. A shifted system gives near 0 |

Earlier attempts on this data trained to completion and produced garbage. One emitted almost
nothing but unknown tokens from a token id mismatch. Another collapsed its vocabulary and
drove the loss to 0.005 by learning only padding. Both were silent for hours of GPU time.
The gates make those failures immediate.

## Measurement

Two scales appear in this repo and they are not interchangeable. `evaluate.py` is the course
grader and sums BLEU, ROUGE-L and chrF between 0 and 1. sacrebleu is the standard and reports
between 0 and 100. Every headline number above is cross checked against sacrebleu.

BLEU and ROUGE-L are computed over word level tokens. Devanagari and Bengali matras are not
word characters under a plain `\w` regex, so a naive tokenizer splits whole words into single
characters and roughly doubles the reported BLEU. chrF is character based and is unaffected.
Tokenization is verified rather than assumed.

## Files

```
src/common.py           paths, special ids, Unicode normalisation
src/lid.py              char n-gram language ID for data QA
src/data_prep.py        clean, filter, remove leaked rows, translation memory
src/tokenizer.py        joint SentencePiece over en, hi and bn
src/dataset.py          encoding and token budget batching
src/model.py            the Transformer
src/train.py            Noam schedule, label smoothing, AMP, resumable checkpoints
src/decode.py           batched beam search, incremental cache, checkpoint averaging
src/score.py            BLEU, ROUGE-L and chrF++ with a sacrebleu cross check
src/gates.py            the 22 verification gates
src/make_submission.py  writes the submission file
configs/base.yaml       the training configuration
colab_train.ipynb       the full executed run, with outputs
artifacts/              dev hypotheses, scores, and the submission
```

## References

Vaswani et al. 2017, Attention Is All You Need, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

Johnson et al. 2017, Google's Multilingual Neural Machine Translation System,
[arXiv:1611.04558](https://arxiv.org/abs/1611.04558)

Wu et al. 2016, Google's Neural Machine Translation System,
[arXiv:1609.08144](https://arxiv.org/abs/1609.08144)

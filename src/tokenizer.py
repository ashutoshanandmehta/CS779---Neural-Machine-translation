"""Train and load the joint SentencePiece model.

One vocabulary spanning English, Hindi and Bengali. Joint rather than
per-language is what lets the embedding table be shared and tied to the output
projection, and it is the mechanism by which the two directions share
representations -- Hindi and Bengali have substantial common Sanskrit-derived
(tatsama) vocabulary, so the pieces genuinely transfer.

Special ids are pinned to the constants in common.py and asserted on load.
The previous attempt's all-<unk> output came from exactly this drifting.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sentencepiece as spm  # noqa: E402

from common import (  # noqa: E402
    BOS_ID,
    CLEAN_DIR,
    EOS_ID,
    PAD_ID,
    SPM_DIR,
    TAG_TOKENS,
    UNK_ID,
)

DEFAULT_MODEL_PREFIX = os.path.join(SPM_DIR, "joint")


def train(input_path, model_prefix, vocab_size, model_type="unigram",
          character_coverage=1.0):
    os.makedirs(os.path.dirname(model_prefix), exist_ok=True)
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        # Pinned ids -- must match common.py.
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        # Target-language tags must survive as atomic pieces.
        user_defined_symbols=TAG_TOKENS,
        # The corpus has only 410 distinct characters, so full coverage costs
        # ~2.6% of the vocabulary. At the default 0.9995 the tokenizer silently
        # dropped '&', 'Q', 'Z', '(rupee)' and curly quotes to <unk> -- all of
        # which appear in the validation sources.
        byte_fallback=True,
        normalization_rule_name="identity",  # we already normalised in data_prep
        input_sentence_size=2000000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
    )
    return f"{model_prefix}.model"


def load(model_path=None, vocab_size=None):
    """Load the SP model and assert the id contract holds."""
    model_path = model_path or f"{DEFAULT_MODEL_PREFIX}.model"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"{model_path} not found -- run `python src/tokenizer.py` first"
        )
    sp = spm.SentencePieceProcessor(model_file=model_path)

    assert sp.pad_id() == PAD_ID, f"pad_id {sp.pad_id()} != {PAD_ID}"
    assert sp.unk_id() == UNK_ID, f"unk_id {sp.unk_id()} != {UNK_ID}"
    assert sp.bos_id() == BOS_ID, f"bos_id {sp.bos_id()} != {BOS_ID}"
    assert sp.eos_id() == EOS_ID, f"eos_id {sp.eos_id()} != {EOS_ID}"
    if vocab_size is not None:
        assert sp.get_piece_size() == vocab_size, (
            f"vocab {sp.get_piece_size()} != expected {vocab_size}"
        )
    # The tags must exist as real, atomic pieces. They are injected into the
    # token sequence by id (see dataset.py) rather than encoded as text, which
    # avoids SentencePiece's dummy-whitespace prefix costing an extra token.
    for tag in TAG_TOKENS:
        tid = sp.piece_to_id(tag)
        assert tid != sp.unk_id(), f"tag {tag} is not in the vocabulary"
        assert sp.id_to_piece(tid) == tag, f"tag {tag} maps back to {sp.id_to_piece(tid)}"

    return sp


def main():
    ap = argparse.ArgumentParser(description="Train the joint SentencePiece model.")
    ap.add_argument("--input", default=os.path.join(CLEAN_DIR, "spm_input.txt"))
    ap.add_argument("--model-prefix", default=DEFAULT_MODEL_PREFIX)
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    args = ap.parse_args()

    prefix = f"{args.model_prefix}{args.vocab_size // 1000}k"
    print(f"training {args.model_type} vocab={args.vocab_size} on {args.input}")
    path = train(args.input, prefix, args.vocab_size, args.model_type)
    print(f"wrote {path}")

    sp = load(path, vocab_size=args.vocab_size)
    print(f"pieces={sp.get_piece_size()}  "
          f"pad={sp.pad_id()} unk={sp.unk_id()} bos={sp.bos_id()} eos={sp.eos_id()}")
    for tag in TAG_TOKENS:
        print(f"  {tag} -> id {sp.piece_to_id(tag)}")

    demo = "The remaining soybean meal is used mainly as animal feed."
    ids = [sp.piece_to_id("<2hi>")] + sp.encode(demo, out_type=int)
    print(f"\ndemo pieces: {[sp.id_to_piece(i) for i in ids][:12]} ...")
    print(f"demo decode: {sp.decode(ids)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import sentencepiece as spm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to training text, e.g. data/train.txt")
    ap.add_argument("--model_prefix", default="tokenizer/barebonegpt")
    ap.add_argument("--vocab_size", type=int, default=16000)
    ap.add_argument("--character_coverage", type=float, default=1.0)
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    model_prefix = Path(args.model_prefix)
    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=args.vocab_size,
        model_type="bpe",
        character_coverage=args.character_coverage,
        normalization_rule_name="identity",
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
    )

    print(f"Tokenizer model written to: {model_prefix}.model")
    print(f"Tokenizer vocab written to:  {model_prefix}.vocab")


if __name__ == "__main__":
    main()
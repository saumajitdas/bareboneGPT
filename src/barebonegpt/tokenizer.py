from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sentencepiece as spm


class TokenizerBase:
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError


@dataclass
class ByteTokenizer(TokenizerBase):
    @property
    def vocab_size(self) -> int:
        return 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))

    def decode(self, ids: list[int]) -> str:
        return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


class SentencePieceTokenizer(TokenizerBase):
    def __init__(self, model_path: str):
        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")

        self.model_path = str(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str) -> list[int]:
        return list(self.sp.encode(text, out_type=int))

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)


def build_tokenizer(tokenizer_cfg: dict[str, Any] | None) -> TokenizerBase:
    if tokenizer_cfg is None:
        return ByteTokenizer()

    tok_type = tokenizer_cfg.get("type", "byte").lower()

    if tok_type == "byte":
        return ByteTokenizer()

    if tok_type in {"spm", "sentencepiece", "bpe"}:
        model_path = tokenizer_cfg.get("model_path")
        if not model_path:
            raise ValueError(
                "SentencePiece tokenizer requires tokenizer.model_path in config."
            )
        return SentencePieceTokenizer(model_path=model_path)

    raise ValueError(f"Unsupported tokenizer type: {tok_type}")
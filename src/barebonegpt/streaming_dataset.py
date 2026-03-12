from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from .tokenizer import build_tokenizer


@dataclass
class StreamTokenizedDataset(IterableDataset):
    """
    Streams a large text file, tokenizes chunks on the fly, and yields (x, y) windows.

    Works for:
    - byte tokenizer
    - SentencePiece tokenizer

    Notes:
    - The file is partitioned by byte ranges across workers.
    - Each worker decodes bytes with utf-8 errors ignored.
    - Boundary tokenization loss is acceptable for large corpora.
    """

    path: str
    tokenizer_cfg: dict
    context_length: int
    chunk_bytes: int = 4 * 1024 * 1024
    stride: int = 8
    loop: bool = True

    def _iter_range(self, start: int, end: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        tokenizer = build_tokenizer(self.tokenizer_cfg)

        with open(self.path, "rb") as f:
            f.seek(start)
            remaining = end - start
            token_buffer: list[int] = []

            while True:
                if remaining <= 0:
                    if not self.loop:
                        break
                    f.seek(start)
                    remaining = end - start

                to_read = min(self.chunk_bytes, remaining)
                chunk = f.read(to_read)

                if not chunk:
                    if not self.loop:
                        break
                    f.seek(start)
                    remaining = end - start
                    continue

                remaining -= len(chunk)

                text = chunk.decode("utf-8", errors="ignore")
                if not text:
                    continue

                token_buffer.extend(tokenizer.encode(text))

                while len(token_buffer) >= self.context_length + 1:
                    window = token_buffer[: self.context_length + 1]
                    t = torch.tensor(window, dtype=torch.long)
                    x = t[:-1]
                    y = t[1:]
                    yield x, y
                    del token_buffer[: self.stride]

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        file_size = os.path.getsize(self.path)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            start, end = 0, file_size
        else:
            n = worker_info.num_workers
            wid = worker_info.id
            per = file_size // n

            start = wid * per
            end = file_size if wid == n - 1 else (wid + 1) * per

            # Backup slightly to reduce boundary loss.
            start = max(0, start - self.chunk_bytes)

        return self._iter_range(start, end)
"""Shared runtime helpers for model adapters and servers: file hashing and measured GPU time."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


class GpuTimer:
    """Measured device time for a block: CUDA events on GPU, ``None`` on CPU.

    ``ms`` is ``None`` when nothing was measured, so the pipeline's GPU budget is
    only ever enforced against real measurements (wall time is not a proxy; the
    elapsed-time cap covers wall time).
    """

    ms: float | None = None


@contextmanager
def cuda_timer(device: str) -> Iterator[GpuTimer]:
    timer = GpuTimer()
    if not str(device).startswith("cuda"):
        yield timer
        return
    import torch

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    try:
        yield timer
    finally:
        end.record()
        torch.cuda.synchronize()
        timer.ms = round(float(start.elapsed_time(end)), 3)

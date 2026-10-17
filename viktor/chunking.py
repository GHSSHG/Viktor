"""Dorado's variable-chunk-size chunking for transformer models.

utils::generate_variable_chunks_tx: full chunks every chunksize - overlap samples, then the
remainder as a shorter chunk; BasecallerNode repeat-pads each chunk (with its own start) to a
multiple of the granularity (768 samples for SUP).
"""

import numpy as np


def chunk_read(n, size, overlap):
    out, off = [], 0
    while off + size < n:
        out.append((off, off + size))
        off += size - overlap
    if off != n:
        out.append((off, n))
    return out


def repeat_pad(x, unit):
    while len(x) % unit:
        x = np.concatenate([x, x[:unit - len(x) % unit]])
    return x


def pack(chunks):
    """list of 1D arrays -> (concatenated signal, lens)."""
    return np.concatenate(chunks), np.array([len(c) for c in chunks], dtype=np.int64)

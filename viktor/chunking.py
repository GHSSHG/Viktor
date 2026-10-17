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


# ScalerNode.cpp expected open-pore levels (pA) by flow cell product code
_PORE_LEVEL = {"FLO-FLG114": 200.0, "FLO-FLG114HD": 200.0, "FLO-MIN004RA": 195.50, "FLO-PRO004RA": 194.97,
               "FLO-MIN114": 197.61, "FLO-MIN114HD": 197.61, "FLO-PRO114": 199.21, "FLO-PRO114HD": 199.21,
               "FLO-PRO114M": 199.21}
DNA_TRIM_START = 10  # ScalerNode: constant DNA trim for standardised pA scaling


def dorado_scale(raw, offset, scaling, mean, stdev, open_pore_level=float("nan"), flow_cell=""):
    """dorado ScalerNode, pA strategy with standardisation, DNA: int16 raw -> standardised fp16 signal,
    first DNA_TRIM_START samples removed."""
    shift = mean / scaling - offset
    scale = stdev / scaling
    expected = _PORE_LEVEL.get(flow_cell.upper().split("_")[0], 0.0)
    if not np.isnan(open_pore_level) and expected:
        shift += (open_pore_level - expected) / scaling
    x = ((raw.astype(np.float32) - np.float32(shift)) / np.float32(scale)).astype(np.float16)
    return x[DNA_TRIM_START:] if len(x) > DNA_TRIM_START else x


def scale_record(rec, mean, stdev):
    """dorado_scale for a pod5 ReadRecord."""
    return dorado_scale(rec.signal, rec.calibration.offset, rec.calibration.scale, mean, stdev,
                        getattr(rec, "open_pore_level", float("nan")), rec.run_info.flow_cell_product_code)

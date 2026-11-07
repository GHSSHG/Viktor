"""Rescore saved window calls (evaluate.py's *_calls.parquet) with errors / call length, the read-level identity formula.
Usage: python -m scripts.rescore_windows WINDOWS.parquet REF.fa CALLS.parquet OUT.json [snap-key ...]  (default: every column)"""
import json
import os
import sys

import polars as pl

from scripts.evaluate import scores
from viktor import load_dorado_encoder
from viktor.data import load_windows

MODEL = os.path.expanduser("~/tools/dorado_models/dna_r10.4.1_e8.2_400bps_sup@v5.2.0")
windows, ref, calls_path, out, *keys = sys.argv[1:]
rows = pl.read_parquet(windows).to_dicts()
enc = load_dorado_encoder(MODEL)
wins = load_windows(rows, enc.mean, enc.stdev, ref)
calls = pl.read_parquet(calls_path)
res = {k: scores(calls[k].to_list(), wins, rows, query=True) for k in (keys or calls.columns)}
json.dump(res, open(out, "w"), indent=1)

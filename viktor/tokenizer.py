"""Output tokenizers: single base (k=1) and fixed k-mer (k=4) with short tail tokens.

Vocabulary layout: PAD, BOS, EOS, then all full k-mers, then tails of length 1..k-1.
A tail token may only appear as the last token before EOS.
"""

from itertools import product

import torch

PAD, BOS, EOS = 0, 1, 2
N_SPECIAL = 3
BASES = "ACGT"


class KmerTokenizer:
    def __init__(self, k: int):
        self.k = k
        full = ["".join(p) for p in product(BASES, repeat=k)]
        tails = ["".join(p) for n in range(1, k) for p in product(BASES, repeat=n)]
        self.strings = ["<pad>", "<bos>", "<eos>"] + full + tails
        self.ids = {s: i for i, s in enumerate(self.strings) if i >= N_SPECIAL}
        self.vocab_size = len(self.strings)
        self.n_full = len(full)
        tail = torch.zeros(self.vocab_size, dtype=torch.bool)
        tail[N_SPECIAL + self.n_full:] = True
        self.tail_mask = tail
        self.n_bases = torch.tensor([0] * N_SPECIAL + [len(s) for s in full + tails])

    def encode(self, seq: str) -> list[int]:
        k = self.k
        return [self.ids[seq[i:i + k]] for i in range(0, len(seq), k)]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == EOS:
                break
            if i >= N_SPECIAL:
                out.append(self.strings[i])
        return "".join(out)

    def batch(self, seqs: list[str]):
        """Teacher-forcing tensors: inputs [BOS, t1..tn], targets [t1..tn, EOS], PAD-padded."""
        toks = [self.encode(s) for s in seqs]
        T = max(len(t) for t in toks) + 1
        inp = torch.full((len(seqs), T), PAD, dtype=torch.long)
        tgt = torch.full((len(seqs), T), PAD, dtype=torch.long)
        for b, t in enumerate(toks):
            inp[b, :len(t) + 1] = torch.tensor([BOS] + t)
            tgt[b, :len(t) + 1] = torch.tensor(t + [EOS])
        return inp, tgt, sum(len(s) for s in seqs)


def get_tokenizer(name: str) -> KmerTokenizer:
    return {"base": KmerTokenizer(1), "4mer": KmerTokenizer(4)}[name]

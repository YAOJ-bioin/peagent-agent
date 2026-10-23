"""Minimal PWM matching against a JASPAR-format motif file (e.g. JASPAR CORE plants).

Purpose: give the agent a *first-pass* name for a high-contribution segment
("looks like a TCP/TB1 site", "looks like an ARF AuxRE") without shipping MEME
or TOMTOM. It is deliberately simple: log-odds scan of the segment on both
strands, best hit per motif, ranked by relative score. The skill text tells the
agent to treat these as candidates to be confirmed with DAP-seq/ChIP evidence,
never as proof of binding.

Expected file: JASPAR "pfm" text format, downloadable from jaspar.elixir.no
(``JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.txt``).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_BASES = "ACGT"
_IDX = {b: i for i, b in enumerate(_BASES)}


@dataclass
class Motif:
    matrix_id: str
    name: str
    pfm: np.ndarray  # shape (4, L)

    @property
    def length(self) -> int:
        return int(self.pfm.shape[1])

    def log_odds(self, background: Optional[np.ndarray] = None, pseudocount: float = 0.5) -> np.ndarray:
        bg = background if background is not None else np.full(4, 0.25)
        counts = self.pfm + pseudocount * bg[:, None]
        probs = counts / counts.sum(axis=0, keepdims=True)
        return np.log2(probs / bg[:, None])

    def consensus(self) -> str:
        return "".join(_BASES[i] for i in self.pfm.argmax(axis=0))


@lru_cache(maxsize=2)
def load_jaspar(path: str) -> List[Motif]:
    motifs: List[Motif] = []
    current_id, current_name, rows = None, None, {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id and len(rows) == 4:
                    motifs.append(Motif(current_id, current_name or current_id, np.array([rows[b] for b in _BASES])))
                header = line[1:].split(None, 1)
                current_id = header[0]
                current_name = header[1] if len(header) > 1 else header[0]
                rows = {}
            else:
                m = re.match(r"^([ACGT])\s*\[?\s*([\d.\s]+)\]?$", line)
                if m:
                    rows[m.group(1)] = [float(x) for x in m.group(2).split()]
    if current_id and len(rows) == 4:
        motifs.append(Motif(current_id, current_name or current_id, np.array([rows[b] for b in _BASES])))
    return motifs


def _encode(seq: str) -> np.ndarray:
    arr = np.full((4, len(seq)), 0.25)
    for i, b in enumerate(seq.upper()):
        if b in _IDX:
            arr[:, i] = 0.0
            arr[_IDX[b], i] = 1.0
    return arr


def _rc(seq: str) -> str:
    return seq.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def scan_segment(segment: str, motifs: List[Motif], top_n: int = 5, min_relative_score: float = 0.80) -> List[Dict[str, object]]:
    """Best hit of every motif within ``segment`` (both strands), ranked by relative score."""
    hits: List[Dict[str, object]] = []
    seg = segment.upper()
    for motif in motifs:
        L = motif.length
        if L > len(seg):
            continue
        lo = motif.log_odds()
        max_score = float(lo.max(axis=0).sum())
        min_score = float(lo.min(axis=0).sum())
        best = (-math.inf, 0, "+")
        for strand, s in (("+", seg), ("-", _rc(seg))):
            enc = _encode(s)
            for start in range(0, len(s) - L + 1):
                score = float((lo * enc[:, start : start + L]).sum())
                if score > best[0]:
                    best = (score, start, strand)
        rel = (best[0] - min_score) / (max_score - min_score) if max_score > min_score else 0.0
        if rel >= min_relative_score:
            start = best[1] if best[2] == "+" else len(seg) - best[1] - L
            hits.append(
                {
                    "matrix_id": motif.matrix_id,
                    "name": motif.name,
                    "consensus": motif.consensus(),
                    "relative_score": round(rel, 3),
                    "log_odds": round(best[0], 2),
                    "strand": best[2],
                    "offset_in_segment": start + 1,
                    "matched_sequence": seg[start : start + L],
                }
            )
    hits.sort(key=lambda h: -float(h["relative_score"]))
    return hits[:top_n]

"""ACR atlas lookup: the peak set each model was trained on.

BED files are 0-based half-open (``chrom start end``), exactly as used in training. The model
window of an ACR is the 1,344-bp interval centred the way scBasset centred it during training:
``mid = (start + end) // 2``; ``seq_start0 = mid - 672``. Everything returned here is 1-based
inclusive, like the rest of the server.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from .genome import INPUT_LEN


def acr_relative_path(species: str) -> str:
    return "acr/%s_acr_peaks.bed" % species


@lru_cache(maxsize=3)
def load_acrs(species: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Return {chrom: (starts0, ends0)} sorted by start. Resolved like model weights
    (explicit $PEAGENT_ASSET_ROOT -> cache -> Hugging Face)."""
    from peagent_tool.assets import resolve_asset

    path = resolve_asset(acr_relative_path(species))
    per: Dict[str, List[Tuple[int, int]]] = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            c, s, e = line.split("\t")[:3]
            per.setdefault(c, []).append((int(s), int(e)))
    out = {}
    for c, iv in per.items():
        arr = np.array(sorted(iv), dtype=np.int64)
        out[c] = (arr[:, 0], arr[:, 1])
    return out


def model_window0(start0: int, end0: int) -> Tuple[int, int]:
    """0-based half-open model window of an ACR (training convention)."""
    mid = (start0 + end0) // 2
    s0 = mid - INPUT_LEN // 2
    return s0, s0 + INPUT_LEN


def _resolve_chrom(acrs: Dict, chrom: str) -> Optional[str]:
    for cand in (chrom, chrom.replace("chr", ""), "chr" + chrom.replace("chr", ""), chrom.capitalize(), chrom.lower()):
        if cand in acrs:
            return cand
    return None


def acr_record(chrom: str, s0: int, e0: int, query_mid1: Optional[int] = None) -> Dict[str, object]:
    ws0, we0 = model_window0(s0, e0)
    rec: Dict[str, object] = {
        "acr_id": "%s_%d_%d" % (chrom, s0, e0),
        "acr": "%s:%d-%d" % (chrom, s0 + 1, e0),
        "acr_center": (s0 + e0) // 2 + 1,
        "length_bp": e0 - s0,
        "model_window": "%s:%d-%d" % (chrom, ws0 + 1, we0),
    }
    if query_mid1 is not None:
        if s0 + 1 <= query_mid1 <= e0:
            rec["distance_bp"] = 0
        else:
            rec["distance_bp"] = int(min(abs(query_mid1 - (s0 + 1)), abs(query_mid1 - e0)))
        rec["offset_from_acr_center_bp"] = int(query_mid1 - ((s0 + e0) // 2 + 1))
    return rec


def find_acrs(species: str, chrom: str, start1: int, end1: int, max_hits: int = 25) -> Tuple[Optional[str], List[Dict[str, object]]]:
    """ACRs overlapping the 1-based inclusive interval, nearest to its midpoint first."""
    acrs = load_acrs(species)
    c = _resolve_chrom(acrs, chrom)
    if c is None:
        return None, []
    starts, ends = acrs[c]
    lo = int(np.searchsorted(ends, start1 - 1, side="right"))  # ends sorted approx with starts (non-overlapping peaks)
    lo = max(0, lo - 5)
    hi = int(np.searchsorted(starts, end1, side="left"))
    mid1 = (start1 + end1) // 2
    hits = [acr_record(c, int(starts[i]), int(ends[i]), mid1) for i in range(lo, hi) if ends[i] > start1 - 1 and starts[i] < end1]
    hits.sort(key=lambda r: (r["distance_bp"], abs(r["offset_from_acr_center_bp"])))
    return c, hits[:max_hits]


def acr_containing(species: str, chrom: str, pos1: int) -> Optional[Dict[str, object]]:
    c, hits = find_acrs(species, chrom, pos1, pos1, max_hits=1)
    return hits[0] if hits else None

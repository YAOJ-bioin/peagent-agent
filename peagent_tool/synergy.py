"""In-silico marginalization: does a pair of motifs act more than additively?

A motif pair is tested by implanting motif A alone, motif B alone, and both
together into a panel of background sequences, over a grid of spacings and
relative orientations. The readout is the pre-sigmoid pseudobulk logit of one
cell group,

    f_g(x) = z(x) . w_g + b_g

where ``z(x)`` is the 64-d bottleneck embedding, ``w_g`` the mean final-layer
weight vector over the cells of the group and ``b_g`` the mean bias. Synergy is

    synergy = dJoint - (dA + dB)

tested per arrangement with a one-sided paired Wilcoxon signed-rank test across
backgrounds, and corrected across all arrangements with Benjamini-Hochberg.

This mirrors the ``05_synergy`` step of the PEAgent analysis pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .attribution import GLOBAL_TARGET, ContributionScorer, _as_bottleneck_matrix
from .celltype_prediction import BOTTLENECK_SIZE
from .sequence import INPUT_LEN, onehot_from_normalized_sequence

COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A", "N": "N"}
HETERO_ORIENTATIONS = ("++", "+-", "-+", "--")
HOMO_ORIENTATIONS = ("++", "+-", "--")
DEFAULT_SPACINGS = tuple(list(range(0, 31)) + list(range(35, 205, 5)))


def reverse_complement(sequence: str) -> str:
    """Reverse complement a DNA string."""

    return "".join(COMPLEMENT[base] for base in reversed(str(sequence).upper()))


def implant(background: str, motif: str, start: int) -> str:
    """Replace ``len(motif)`` bases of ``background`` starting at 0-based ``start``."""

    if start < 0 or start + len(motif) > len(background):
        raise ValueError("Motif at %d does not fit in a %d bp background." % (start, len(background)))
    return background[:start] + motif + background[start + len(motif) :]


def shuffled_backgrounds(
    sequence: str,
    n: int = 32,
    seed: int = 2026,
    input_len: int = INPUT_LEN,
) -> List[str]:
    """Dinucleotide-preserving shuffles of one sequence, for use as backgrounds."""

    rng = np.random.default_rng(seed)
    letters = [base for base in str(sequence).upper() if base in "ACGT"]
    if len(letters) < input_len:
        raise ValueError("Need at least %d A/C/G/T bases to build backgrounds." % input_len)
    backgrounds = []
    for _ in range(int(n)):
        pairs = [letters[i : i + 2] for i in range(0, len(letters) - 1, 2)]
        rng.shuffle(pairs)
        shuffled = "".join("".join(pair) for pair in pairs)
        backgrounds.append(shuffled[:input_len])
    return backgrounds


def random_backgrounds(
    n: int = 32,
    gc_content: float = 0.42,
    seed: int = 2026,
    input_len: int = INPUT_LEN,
) -> List[str]:
    """Independent random backgrounds at a fixed GC content."""

    rng = np.random.default_rng(seed)
    gc = float(gc_content) / 2.0
    at = (1.0 - float(gc_content)) / 2.0
    probabilities = [at, gc, gc, at]
    return ["".join(rng.choice(list("ACGT"), size=int(input_len), p=probabilities)) for _ in range(int(n))]


@dataclass(frozen=True)
class SynergyResult:
    """Per-arrangement synergy table plus the best arrangement per pair."""

    arrangements: pd.DataFrame
    best: pd.DataFrame
    target: str
    n_backgrounds: int

    def significant(self, alpha: float = 0.01) -> pd.DataFrame:
        """Arrangements whose synergy survives BH correction at ``alpha``."""

        return self.arrangements[self.arrangements["padj"] < float(alpha)].copy()


class MotifSynergyTester:
    """Test motif pairs for non-additive effects on one PEAgent cell group."""

    def __init__(
        self,
        species: str,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        n_cells: Optional[int] = None,
        bottleneck_size: int = BOTTLENECK_SIZE,
        scorer: Optional[ContributionScorer] = None,
        batch_size: int = 64,
    ):
        self.scorer = scorer or ContributionScorer(
            species=species,
            model_path=model_path,
            metadata_path=metadata_path,
            n_cells=n_cells,
            bottleneck_size=bottleneck_size,
        )
        self.batch_size = int(batch_size)

    @cached_property
    def dense_bias(self) -> np.ndarray:
        return np.asarray(self.scorer.final_dense_layer.get_weights()[1], dtype=np.float32)

    def target_readout(self, target: str):
        """Return ``(w_g, b_g)`` for one target group, or the global average."""

        weights, meta = self.scorer.target_weights(target)
        bias = self.dense_bias
        if str(target).strip().lower() == GLOBAL_TARGET:
            return weights, float(bias.mean())
        row = self.scorer.group_metadata
        row = row[row["tissue_celltype"].astype(str) == str(target).strip()].iloc[0]
        indices = np.asarray(row["cell_indices"], dtype=np.int32)
        return weights, float(bias[indices].mean())

    def group_logit(self, sequences: Sequence[str], weights: np.ndarray, bias: float) -> np.ndarray:
        """Pre-sigmoid pseudobulk logit of one cell group for each sequence."""

        outputs = []
        for start in range(0, len(sequences), self.batch_size):
            batch = sequences[start : start + self.batch_size]
            x = np.stack([onehot_from_normalized_sequence(s, input_len=INPUT_LEN) for s in batch], axis=0)
            embedding = _as_bottleneck_matrix(
                self.scorer.bottleneck_model.predict_on_batch(x),
                expected_size=self.scorer.config.bottleneck_size,
            )
            outputs.append(embedding @ weights + bias)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def test_pair(
        self,
        motif_a: str,
        motif_b: str,
        target: str,
        backgrounds: Sequence[str],
        spacings: Iterable[int] = DEFAULT_SPACINGS,
        orientations: Optional[Sequence[str]] = None,
        centre: int = INPUT_LEN // 2,
        name_a: Optional[str] = None,
        name_b: Optional[str] = None,
    ) -> pd.DataFrame:
        """Return the per-arrangement synergy table for one motif pair (uncorrected p)."""

        from scipy.stats import wilcoxon

        motif_a = str(motif_a).strip().upper()
        motif_b = str(motif_b).strip().upper()
        name_a = name_a or motif_a
        name_b = name_b or motif_b
        backgrounds = [str(background).upper() for background in backgrounds]
        if orientations is None:
            orientations = HOMO_ORIENTATIONS if motif_a == motif_b else HETERO_ORIENTATIONS

        weights, bias = self.target_readout(target)
        baseline = self.group_logit(backgrounds, weights, bias)
        # Motifs are placed by their CENTRE (paper Methods): start = anchor - len(motif) // 2.
        delta_a_forward = self.group_logit(
            [implant(background, motif_a, centre - len(motif_a) // 2) for background in backgrounds], weights, bias
        ) - baseline

        records = []
        for orientation in orientations:
            oriented_a = motif_a if orientation[0] == "+" else reverse_complement(motif_a)
            oriented_b = motif_b if orientation[1] == "+" else reverse_complement(motif_b)
            delta_a = (
                delta_a_forward
                if orientation[0] == "+"
                else self.group_logit(
                    [implant(background, oriented_a, centre - len(oriented_a) // 2) for background in backgrounds], weights, bias
                )
                - baseline
            )
            delta_b = (
                self.group_logit(
                    [implant(background, oriented_b, centre - len(oriented_b) // 2) for background in backgrounds], weights, bias
                )
                - baseline
            )
            for spacing in spacings:
                spacing = int(spacing)
                edge_gap = spacing - (len(motif_a) + len(motif_b)) / 2.0
                if edge_gap < 0:  # the two motifs would overlap
                    continue
                anchor_a = centre - spacing // 2  # centre-to-centre spacing
                start_a = anchor_a - len(oriented_a) // 2
                start_b = anchor_a + spacing - len(oriented_b) // 2
                joint = [
                    implant(implant(background, oriented_a, start_a), oriented_b, start_b)
                    for background in backgrounds
                ]
                delta_joint = self.group_logit(joint, weights, bias) - baseline
                synergy = delta_joint - (delta_a + delta_b)
                p_value = float(wilcoxon(synergy, alternative="greater")[1]) if np.any(synergy) else 1.0
                records.append(
                    {
                        "motif_a": name_a,
                        "motif_b": name_b,
                        "target": target,
                        "orientation": orientation,
                        "spacing": spacing,
                        "edge_gap": edge_gap,
                        "mean_delta_a": float(delta_a.mean()),
                        "mean_delta_b": float(delta_b.mean()),
                        "mean_delta_joint": float(delta_joint.mean()),
                        "mean_synergy": float(synergy.mean()),
                        "p": p_value,
                    }
                )
        return pd.DataFrame(records)

    def test_pairs(
        self,
        pairs: Iterable[Dict[str, str]],
        target: str,
        backgrounds: Sequence[str],
        spacings: Iterable[int] = DEFAULT_SPACINGS,
        alpha: float = 0.01,
        **kwargs,
    ) -> SynergyResult:
        """Test several motif pairs and apply one BH correction across all arrangements."""

        frames = [
            self.test_pair(
                pair["motif_a"],
                pair["motif_b"],
                target=target,
                backgrounds=backgrounds,
                spacings=spacings,
                name_a=pair.get("name_a"),
                name_b=pair.get("name_b"),
                **kwargs,
            )
            for pair in pairs
        ]
        arrangements = pd.concat(frames, ignore_index=True)
        arrangements["padj"] = benjamini_hochberg(arrangements["p"].to_numpy())
        best = (
            arrangements.sort_values("mean_delta_joint")
            .groupby(["motif_a", "motif_b"], as_index=False)
            .tail(1)
            .sort_values("mean_synergy", ascending=False)
            .reset_index(drop=True)
        )
        return SynergyResult(
            arrangements=arrangements,
            best=best,
            target=str(target),
            n_backgrounds=len(backgrounds),
        )


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""

    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted

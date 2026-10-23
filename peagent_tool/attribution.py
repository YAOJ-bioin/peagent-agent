"""Contribution (attribution) scores for PEAgent models via bottleneck in-silico mutagenesis.

Every single-cell output of a PEAgent model reads out the same 64-d bottleneck
embedding, so in-silico mutagenesis is run **once** per sequence at the
bottleneck and then projected through the final dense layer onto any cell group.
One ISM pass therefore yields contribution scores for the global track and for
all ``tissue_celltype`` groups at no extra model cost.

Two normalizations are provided, matching the PEAgent analysis pipeline:

``normalize="none"`` (raw)
    ``centered_ism . w_g`` projected onto the reference base. This is the
    convention of the *global* PEAgent track.

``normalize="tissue"``
    The same score minus the mean of the reference-projected scores over **all
    valid ``tissue_celltype`` groups of the same tissue**. This is the
    convention of the cell-type PEAgent tracks, i.e. the scores shown as
    *Attribution score* in the PEAgent JBrowse browser, and the input used for
    TF-MoDISco motif discovery in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .celltype_prediction import (
    BOTTLENECK_SIZE,
    _get_make_model,
    _load_group_metadata,
    resolve_species_config,
)
from .sequence import INPUT_LEN, SequenceRecord, onehot_from_normalized_sequence, parse_sequences

BASES = ("A", "C", "G", "T")
BASE_TO_INDEX = {base: idx for idx, base in enumerate(BASES)}
DEFAULT_POSITION_BATCH_SIZE = 64
GLOBAL_TARGET = "global"
NORMALIZATIONS = ("none", "tissue")
DNAInput = Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]]


@dataclass(frozen=True)
class AttributionResult:
    """Contribution scores for one sequence and one target."""

    sequence_id: str
    species: str
    species_display_name: str
    target: str
    tissue: Optional[str]
    celltype: Optional[str]
    n_cells: int
    normalize: str
    tissue_reference: Tuple[str, ...]
    original_length: int
    normalized_length: int
    normalization_action: str
    normalized_sequence: str
    hypothetical: np.ndarray
    attribution: np.ndarray
    raw_attribution: np.ndarray
    model_path: str
    metadata_path: str

    @property
    def profile(self) -> np.ndarray:
        """Per-position contribution score of the reference base, shape ``(L,)``."""

        return self.attribution.sum(axis=1)

    def attribution_dataframe(self) -> pd.DataFrame:
        """Reference-projected contribution scores as an A/C/G/T matrix (logomaker input)."""

        return pd.DataFrame(self.attribution, columns=list(BASES))

    def hypothetical_dataframe(self) -> pd.DataFrame:
        """Hypothetical (all four bases) contribution scores as an A/C/G/T matrix."""

        return pd.DataFrame(self.hypothetical, columns=list(BASES))

    def to_dataframe(self, position_offset: int = 0) -> pd.DataFrame:
        """Tidy per-position table with the reference base and both score versions."""

        positions = np.arange(1, self.normalized_length + 1, dtype=np.int64) + int(position_offset)
        return pd.DataFrame(
            {
                "position": positions,
                "base": list(self.normalized_sequence),
                "contribution": self.profile,
                "contribution_raw": self.raw_attribution.sum(axis=1),
            }
        )

    def describe(self) -> Dict[str, object]:
        """Compact provenance summary, useful for logging and notebook display."""

        return {
            "sequence_id": self.sequence_id,
            "species": self.species,
            "target": self.target,
            "tissue": self.tissue,
            "celltype": self.celltype,
            "n_cells": self.n_cells,
            "normalize": self.normalize,
            "n_tissue_reference_groups": len(self.tissue_reference),
            "normalized_length": self.normalized_length,
            "normalization_action": self.normalization_action,
        }


def _strict_reference_onehot(seq: str) -> np.ndarray:
    """One-hot the reference sequence; ``N`` positions stay all-zero."""

    arr = np.zeros((len(seq), 4), dtype=np.float32)
    for pos, base in enumerate(seq.upper()):
        idx = BASE_TO_INDEX.get(base)
        if idx is not None:
            arr[pos, idx] = 1.0
    return arr


def _center_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != 4:
        raise ValueError("scores must have shape (sequence_length, 4).")
    return scores - scores.mean(axis=1, keepdims=True)


def _normalize_mode(normalize: Union[str, bool, None]) -> str:
    if normalize is None or normalize is False:
        return "none"
    if normalize is True:
        return "tissue"
    value = str(normalize).strip().lower()
    aliases = {"raw": "none", "global": "none", "tissue_mean": "tissue", "celltype": "tissue"}
    value = aliases.get(value, value)
    if value not in NORMALIZATIONS:
        raise ValueError("normalize must be one of %s, got %r" % (", ".join(NORMALIZATIONS), normalize))
    return value


def _find_final_dense_layer(model, tf, n_cells: int):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Dense):
            weights = layer.get_weights()
            if len(weights) == 2 and weights[0].shape[1] == int(n_cells):
                return layer
    raise ValueError("Could not find final Dense layer with %d outputs." % int(n_cells))


def _window_to_slice(sequence_length: int, plot_start: Optional[int], plot_end: Optional[int]) -> Tuple[int, int]:
    start = 1 if plot_start is None else int(plot_start)
    end = int(sequence_length) if plot_end is None else int(plot_end)
    if start < 1:
        raise ValueError("plot_start must be >= 1.")
    if end < start:
        raise ValueError("plot_end must be >= plot_start.")
    if end > int(sequence_length):
        raise ValueError("plot_end must be <= normalized sequence length %d." % int(sequence_length))
    return start - 1, end


def _as_bottleneck_matrix(value, expected_size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("Bottleneck output must include batch and feature dimensions.")
    arr = arr.reshape(int(arr.shape[0]), -1)
    if arr.shape[1] != int(expected_size):
        raise ValueError("Expected bottleneck size %d, got %d." % (int(expected_size), int(arr.shape[1])))
    return arr


def _mutation_batch(reference_onehot: np.ndarray, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int32)
    batch = np.repeat(reference_onehot[None, :, :], repeats=len(positions) * 4, axis=0)
    eye = np.eye(4, dtype=np.float32)
    for offset, pos in enumerate(positions):
        batch[offset * 4 : offset * 4 + 4, int(pos), :] = eye
    return batch


def _single_sequence_record(sequences: DNAInput) -> SequenceRecord:
    records = parse_sequences(sequences, input_len=INPUT_LEN)
    if len(records) != 1:
        raise ValueError("Attribution expects exactly one sequence per call, got %d." % len(records))
    return records[0]


class ContributionScorer:
    """Compute raw and tissue-normalized contribution scores for one PEAgent model."""

    def __init__(
        self,
        species: str,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        n_cells: Optional[int] = None,
        bottleneck_size: int = BOTTLENECK_SIZE,
        model=None,
        bottleneck_model=None,
        dense_kernel: Optional[np.ndarray] = None,
    ):
        self.config = resolve_species_config(
            species=species,
            model_path=model_path,
            metadata_path=metadata_path,
            n_cells=n_cells,
            bottleneck_size=bottleneck_size,
            resolve_model=model is None and bottleneck_model is None,
        )
        self._provided_model = model
        self._provided_bottleneck_model = bottleneck_model
        self._provided_dense_kernel = dense_kernel

    # ------------------------------------------------------------------ assets

    @cached_property
    def group_metadata(self) -> pd.DataFrame:
        return _load_group_metadata(self.config.metadata_path, self.config.n_cells)

    @cached_property
    def valid_celltypes(self) -> List[str]:
        return self.group_metadata["tissue_celltype"].astype(str).tolist()

    @cached_property
    def tissues(self) -> List[str]:
        return sorted(self.group_metadata["tissue"].astype(str).unique().tolist())

    def celltypes_in_tissue(self, tissue: str) -> List[str]:
        """Valid ``tissue_celltype`` groups belonging to one tissue."""

        mask = self.group_metadata["tissue"].astype(str) == str(tissue)
        return self.group_metadata.loc[mask, "tissue_celltype"].astype(str).tolist()

    @cached_property
    def model(self):
        if self._provided_model is not None:
            return self._provided_model
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError("Missing model file: %s" % model_path)
        make_model = _get_make_model()
        model = make_model(
            self.config.bottleneck_size,
            int(self.config.n_cells),
            seq_len=INPUT_LEN,
            show_summary=False,
        )
        model.load_weights(str(model_path))
        return model

    @cached_property
    def final_dense_layer(self):
        import tensorflow as tf

        return _find_final_dense_layer(self.model, tf, self.config.n_cells)

    @cached_property
    def dense_kernel(self) -> np.ndarray:
        if self._provided_dense_kernel is not None:
            kernel = np.asarray(self._provided_dense_kernel, dtype=np.float32)
        else:
            kernel = np.asarray(self.final_dense_layer.get_weights()[0], dtype=np.float32)
        if kernel.ndim != 2:
            raise ValueError("Final Dense kernel must be 2-dimensional.")
        if kernel.shape[1] != int(self.config.n_cells):
            raise ValueError(
                "Final Dense kernel has %d cells, but config expects %d."
                % (int(kernel.shape[1]), int(self.config.n_cells))
            )
        return kernel

    @cached_property
    def bottleneck_model(self):
        if self._provided_bottleneck_model is not None:
            return self._provided_bottleneck_model
        import tensorflow as tf

        return tf.keras.Model(inputs=self.model.input, outputs=self.final_dense_layer.input)

    # ------------------------------------------------------------------ targets

    def target_weights(self, target: str) -> Tuple[np.ndarray, Dict[str, object]]:
        """Return the bottleneck weight vector and metadata for one target."""

        label = str(target).strip()
        kernel = self.dense_kernel
        if label.lower() == GLOBAL_TARGET:
            return kernel.mean(axis=1).astype(np.float32, copy=False), {
                "target": GLOBAL_TARGET,
                "tissue": None,
                "celltype": None,
                "n_cells": int(kernel.shape[1]),
            }

        matches = self.group_metadata[self.group_metadata["tissue_celltype"].astype(str) == label]
        if matches.empty:
            preview = ", ".join(self.valid_celltypes[:10])
            raise ValueError(
                "Unknown target %r. Use target='global' or one valid tissue_celltype. "
                "First valid tissue_celltypes: %s" % (target, preview)
            )
        row = matches.iloc[0]
        indices = np.asarray(row["cell_indices"], dtype=np.int32)
        if indices.size == 0:
            raise ValueError("Target %r has no cell indices." % label)
        return kernel[:, indices].mean(axis=1).astype(np.float32, copy=False), {
            "target": label,
            "tissue": str(row["tissue"]),
            "celltype": str(row["celltype"]),
            "n_cells": int(row["n_cells"]),
        }

    # --------------------------------------------------------------------- ISM

    def bottleneck_ism(
        self,
        record: SequenceRecord,
        position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    ) -> np.ndarray:
        """Return the ``(L, 4, bottleneck)`` change-in-embedding tensor for one sequence.

        The tensor is model-independent of the target, so it can be projected onto
        any number of cell groups afterwards.
        """

        position_batch_size = int(position_batch_size)
        if position_batch_size <= 0:
            raise ValueError("position_batch_size must be positive.")

        model_onehot = onehot_from_normalized_sequence(record.normalized_sequence, input_len=INPUT_LEN)
        reference = _as_bottleneck_matrix(
            self.bottleneck_model.predict_on_batch(model_onehot[None, :, :]),
            expected_size=self.config.bottleneck_size,
        )[0]

        ism = np.empty((INPUT_LEN, 4, int(self.config.bottleneck_size)), dtype=np.float32)
        for start in range(0, INPUT_LEN, position_batch_size):
            stop = min(start + position_batch_size, INPUT_LEN)
            positions = np.arange(start, stop, dtype=np.int32)
            mutants = _mutation_batch(model_onehot, positions)
            mutant_embedding = _as_bottleneck_matrix(
                self.bottleneck_model.predict_on_batch(mutants),
                expected_size=self.config.bottleneck_size,
            )
            ism[start:stop] = (mutant_embedding - reference[None, :]).reshape(len(positions), 4, -1)
        return ism

    # ------------------------------------------------------------- computation

    def compute(
        self,
        sequences: DNAInput,
        target: str = GLOBAL_TARGET,
        normalize: Union[str, bool, None] = "none",
        position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    ) -> AttributionResult:
        """Compute contribution scores for one sequence and one target."""

        results = self.compute_many(
            sequences,
            targets=[target],
            normalize=normalize,
            position_batch_size=position_batch_size,
        )
        return results[str(target).strip()]

    def compute_many(
        self,
        sequences: DNAInput,
        targets: Sequence[str] = (GLOBAL_TARGET,),
        normalize: Union[str, bool, None] = "none",
        position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    ) -> Dict[str, AttributionResult]:
        """Compute contribution scores for one sequence and several targets at once.

        The expensive bottleneck ISM pass is shared across all targets, so asking
        for 20 cell types costs essentially the same as asking for one.
        """

        mode = _normalize_mode(normalize)
        targets = [str(t).strip() for t in targets]
        if not targets:
            raise ValueError("At least one target is required.")

        record = _single_sequence_record(sequences)
        reference_onehot = _strict_reference_onehot(record.normalized_sequence)
        ism = self.bottleneck_ism(record, position_batch_size=position_batch_size)

        # Every group whose reference-projected track is needed: the requested
        # targets plus, for tissue normalization, all groups of each tissue.
        weights: Dict[str, np.ndarray] = {}
        meta: Dict[str, Dict[str, object]] = {}
        tissue_reference: Dict[str, Tuple[str, ...]] = {}

        for target in targets:
            weights[target], meta[target] = self.target_weights(target)
            if mode == "tissue":
                tissue = meta[target]["tissue"]
                if tissue is None:
                    raise ValueError(
                        "normalize='tissue' needs a tissue_celltype target; "
                        "target='global' has no tissue. Use normalize='none' for the global track."
                    )
                siblings = tuple(self.celltypes_in_tissue(str(tissue)))
                tissue_reference[target] = siblings
                for sibling in siblings:
                    if sibling not in weights:
                        weights[sibling], meta[sibling] = self.target_weights(sibling)

        # One tensordot for all needed groups: (L, 4, B) x (B, G) -> (L, 4, G)
        names = list(weights)
        weight_matrix = np.stack([weights[name] for name in names], axis=1)
        projected = np.tensordot(ism, weight_matrix, axes=([2], [0]))

        reference_tracks: Dict[str, np.ndarray] = {}
        hypothetical_tracks: Dict[str, np.ndarray] = {}
        for index, name in enumerate(names):
            centered = _center_scores(projected[:, :, index])
            hypothetical_tracks[name] = centered
            reference_tracks[name] = centered * reference_onehot

        results: Dict[str, AttributionResult] = {}
        for target in targets:
            raw = reference_tracks[target]
            siblings = tissue_reference.get(target, ())
            if mode == "tissue":
                tissue_mean = np.mean([reference_tracks[s] for s in siblings], axis=0)
                attribution = raw - tissue_mean
                hypothetical_mean = np.mean([hypothetical_tracks[s] for s in siblings], axis=0)
                hypothetical = hypothetical_tracks[target] - hypothetical_mean
            else:
                attribution = raw
                hypothetical = hypothetical_tracks[target]

            info = meta[target]
            results[target] = AttributionResult(
                sequence_id=record.sequence_id,
                species=self.config.id,
                species_display_name=self.config.display_name,
                target=str(info["target"]),
                tissue=info["tissue"],
                celltype=info["celltype"],
                n_cells=int(info["n_cells"]),
                normalize=mode,
                tissue_reference=siblings,
                original_length=record.original_length,
                normalized_length=record.normalized_length,
                normalization_action=record.normalization_action,
                normalized_sequence=record.normalized_sequence,
                hypothetical=hypothetical.astype(np.float32, copy=False),
                attribution=attribution.astype(np.float32, copy=False),
                raw_attribution=raw.astype(np.float32, copy=False),
                model_path=str(self.config.model_path),
                metadata_path=str(self.config.metadata_path),
            )
        return results


def compute_attribution(
    sequences: DNAInput,
    species: str,
    target: str = GLOBAL_TARGET,
    normalize: Union[str, bool, None] = "none",
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
) -> AttributionResult:
    """Compute contribution scores for one sequence and one target."""

    scorer = ContributionScorer(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
    )
    return scorer.compute(
        sequences,
        target=target,
        normalize=normalize,
        position_batch_size=position_batch_size,
    )


# ------------------------------------------------------------------- plotting


def _configure_plot_style(matplotlib) -> str:
    from matplotlib import font_manager

    candidates = ["Arial", "Arial Unicode MS", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in candidates if font in available), "DejaVu Sans")
    matplotlib.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "sans-serif",
            "font.sans-serif": [selected],
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )
    return selected


def _default_title(result: AttributionResult) -> str:
    if result.target == GLOBAL_TARGET:
        return "%s | global" % result.species_display_name
    suffix = " | tissue-normalized" if result.normalize == "tissue" else ""
    return "%s | %s (n=%d)%s" % (result.species_display_name, result.target, result.n_cells, suffix)


def _prefixed_path(out_prefix: Union[str, Path], suffix: str) -> Path:
    prefix = Path(out_prefix)
    return prefix.parent / ("%s%s" % (prefix.name, suffix))


def _tick_positions(x_values: np.ndarray) -> np.ndarray:
    if len(x_values) <= 12:
        return x_values
    step = int(np.ceil(len(x_values) / 8.0))
    ticks = list(x_values[::step])
    if ticks[-1] != x_values[-1]:
        ticks.append(x_values[-1])
    return np.asarray(ticks, dtype=x_values.dtype)


def _style_axis(ax) -> None:
    ax.grid(False)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(1.2)


def _draw_logo(ax, result: AttributionResult, start_idx: int, end_idx: int, position_offset: int):
    import logomaker

    scores = result.attribution[start_idx:end_idx, :]
    x_values = np.arange(start_idx + 1, end_idx + 1, dtype=np.int64) + int(position_offset)
    frame = pd.DataFrame(scores, columns=list(BASES), index=x_values)
    logomaker.Logo(
        frame,
        ax=ax,
        color_scheme={"A": "#1A9850", "C": "#1F4AFF", "G": "#FF9D00", "T": "#D7191C"},
        width=0.9,
    )
    ax.set_xlim(float(x_values[0]) - 0.5, float(x_values[-1]) + 0.5)
    ticks = _tick_positions(x_values)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    _style_axis(ax)
    return x_values


def _save(fig, out_prefix: Union[str, Path]) -> Dict[str, Union[Path, str]]:
    import matplotlib.pyplot as plt

    pdf_path = _prefixed_path(out_prefix, ".pdf")
    png_path = _prefixed_path(out_prefix, ".png")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": pdf_path, "png": png_path}


def plot_attribution(
    result: AttributionResult,
    out_prefix: Optional[Union[str, Path]] = None,
    plot_start: Optional[int] = None,
    plot_end: Optional[int] = None,
    position_offset: int = 0,
    title: Optional[str] = None,
    ax=None,
):
    """Draw a sequence-logo contribution map for one result.

    With ``out_prefix`` the figure is written as matching ``.pdf`` and ``.png``
    files and the paths are returned; with ``ax`` the logo is drawn onto an
    existing axis and the axis is returned.
    """

    import matplotlib

    font = _configure_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    start_idx, end_idx = _window_to_slice(result.normalized_length, plot_start, plot_end)

    if ax is not None:
        _draw_logo(ax, result, start_idx, end_idx, position_offset)
        ax.set_title(title if title is not None else _default_title(result), fontweight="bold", pad=10)
        return ax

    n_bases = end_idx - start_idx
    fig_width = min(max(6.0, n_bases * 0.13), 24.0)
    fig, axis = plt.subplots(figsize=(fig_width, 3.0))
    _draw_logo(axis, result, start_idx, end_idx, position_offset)
    axis.set_title(title if title is not None else _default_title(result), fontweight="bold", pad=10)
    axis.set_xlabel("Position")
    axis.set_ylabel("Contribution score")
    if out_prefix is None:
        return fig
    paths = _save(fig, out_prefix)
    paths["font"] = font
    return paths


def plot_attribution_panels(
    results: Sequence[AttributionResult],
    out_prefix: Optional[Union[str, Path]] = None,
    plot_start: Optional[int] = None,
    plot_end: Optional[int] = None,
    position_offset: int = 0,
    labels: Optional[Sequence[str]] = None,
    share_y: bool = True,
    panel_height: float = 1.15,
    title: Optional[str] = None,
):
    """Stack several contribution maps over the same window, one panel per target."""

    import matplotlib

    _configure_plot_style(matplotlib)
    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("At least one result is required.")
    lengths = {result.normalized_length for result in results}
    if len(lengths) != 1:
        raise ValueError("All results must share the same normalized sequence length.")

    start_idx, end_idx = _window_to_slice(results[0].normalized_length, plot_start, plot_end)
    n_bases = end_idx - start_idx
    fig_width = min(max(6.0, n_bases * 0.13), 24.0)
    fig, axes = plt.subplots(
        len(results),
        1,
        figsize=(fig_width, panel_height * len(results) + 1.0),
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    if share_y:
        limit = max(float(np.abs(r.attribution[start_idx:end_idx]).max()) for r in results)
        limit = limit * 1.15 if limit > 0 else 1.0

    for index, (ax, result) in enumerate(zip(axes, results)):
        _draw_logo(ax, result, start_idx, end_idx, position_offset)
        if share_y:
            ax.set_ylim(-limit, limit)
        label = labels[index] if labels is not None else result.target
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11, labelpad=10)
        if index < len(results) - 1:
            ax.tick_params(axis="x", length=0)

    axes[-1].set_xlabel("Position")
    if title:
        axes[0].set_title(title, fontweight="bold", pad=10)
    fig.align_ylabels(axes)

    if out_prefix is None:
        return fig
    return _save(fig, out_prefix)


def compute_and_plot_attribution(
    sequences: DNAInput,
    species: str,
    target: str = GLOBAL_TARGET,
    normalize: Union[str, bool, None] = "none",
    out_prefix: Union[str, Path] = "contribution_map",
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    position_batch_size: int = DEFAULT_POSITION_BATCH_SIZE,
    plot_start: Optional[int] = None,
    plot_end: Optional[int] = None,
    position_offset: int = 0,
    title: Optional[str] = None,
) -> Tuple[AttributionResult, Dict[str, Union[Path, str]]]:
    """Compute and plot one contribution map in a single call."""

    result = compute_attribution(
        sequences=sequences,
        species=species,
        target=target,
        normalize=normalize,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
        position_batch_size=position_batch_size,
    )
    paths = plot_attribution(
        result,
        out_prefix=out_prefix,
        plot_start=plot_start,
        plot_end=plot_end,
        position_offset=position_offset,
        title=title,
    )
    return result, paths

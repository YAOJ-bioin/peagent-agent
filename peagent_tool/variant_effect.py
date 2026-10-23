"""Variant effect scoring for PEAgent scBasset models."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .celltype_prediction import (
    BOTTLENECK_SIZE,
    DEFAULT_BATCH_SIZE,
    _get_make_model,
    _load_group_metadata,
    resolve_species_config,
)
from .sequence import INPUT_LEN, SequenceRecord, onehot_from_normalized_sequence, parse_sequences

BASE_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}
SNP_OFFSET = INPUT_LEN // 2
DNAInput = Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]]


@dataclass(frozen=True)
class VariantRecord:
    """A single biallelic SNV using 1-based genomic coordinates."""

    chrom: str
    pos: int
    ref: str
    alt: str
    variant_id: Optional[str] = None

    def __post_init__(self) -> None:
        chrom = str(self.chrom)
        pos = int(self.pos)
        ref = str(self.ref).upper()
        alt = str(self.alt).upper()
        if pos < 1:
            raise ValueError("Variant pos must be a positive 1-based coordinate.")
        if len(ref) != 1 or len(alt) != 1 or ref not in BASE_TO_INDEX or alt not in BASE_TO_INDEX:
            raise ValueError("VariantRecord only supports biallelic A/C/G/T SNVs.")
        if ref == alt:
            raise ValueError("Variant ref and alt must differ.")
        object.__setattr__(self, "chrom", chrom)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "alt", alt)
        if self.variant_id is None:
            object.__setattr__(self, "variant_id", "%s_%d_%s_%s" % (chrom, pos, ref, alt))
        else:
            object.__setattr__(self, "variant_id", str(self.variant_id))


@dataclass(frozen=True)
class _SequencePair:
    ref_onehot: np.ndarray
    alt_onehot: np.ndarray
    meta: Mapping[str, object]


@dataclass(frozen=True)
class _TargetSpec:
    target: str
    tissue: Optional[str]
    celltype: Optional[str]
    n_cells: int
    indices: Optional[np.ndarray]


def _prob_to_logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x)).astype(np.float32)


def _normalize_model_output(pred: np.ndarray, n_cells: int) -> np.ndarray:
    arr = np.asarray(pred, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[1] == 1:
        arr = arr[:, 0, :]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2:
        raise ValueError("Model predictions must be 2-dimensional after normalization.")
    if arr.shape[1] != int(n_cells):
        raise ValueError("Model returned %d cells, expected %d." % (int(arr.shape[1]), int(n_cells)))
    return arr


def _open_fasta(path_or_fasta):
    if hasattr(path_or_fasta, "fetch") and hasattr(path_or_fasta, "get_reference_length"):
        return path_or_fasta, False
    try:
        import pysam
    except ImportError as exc:
        raise ImportError("FASTA-based variant scoring requires pysam. Install pysam (included in peagent-agent).") from exc
    return pysam.FastaFile(str(path_or_fasta)), True


def _variant_columns_from_dataframe(df: pd.DataFrame) -> tuple[str, str, str, str, Optional[str]]:
    aliases = {
        "chrom": ["chrom", "chr", "CHROM"],
        "pos": ["pos", "position", "POS"],
        "ref": ["ref", "REF"],
        "alt": ["alt", "ALT"],
        "variant_id": ["variant_id", "id", "ID", "snvID"],
    }
    found = {}
    for key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in df.columns:
                found[key] = candidate
                break
    missing = [key for key in ("chrom", "pos", "ref", "alt") if key not in found]
    if missing:
        raise ValueError("Variant table is missing columns: %s" % ", ".join(missing))
    return found["chrom"], found["pos"], found["ref"], found["alt"], found.get("variant_id")


def parse_variant_records(variants) -> list[VariantRecord]:
    """Coerce a DataFrame or iterable into VariantRecord objects."""

    if isinstance(variants, pd.DataFrame):
        chrom_col, pos_col, ref_col, alt_col, id_col = _variant_columns_from_dataframe(variants)
        out = []
        for row in variants.itertuples(index=False):
            data = row._asdict()
            out.append(
                VariantRecord(
                    chrom=data[chrom_col],
                    pos=int(data[pos_col]),
                    ref=data[ref_col],
                    alt=data[alt_col],
                    variant_id=data[id_col] if id_col is not None else None,
                )
            )
        return out

    out = []
    for item in variants:
        if isinstance(item, VariantRecord):
            out.append(item)
        elif isinstance(item, Mapping):
            variant_id = item.get("variant_id", item.get("id", item.get("ID", item.get("snvID"))))
            out.append(
                VariantRecord(
                    chrom=item.get("chrom", item.get("chr", item.get("CHROM"))),
                    pos=int(item.get("pos", item.get("position", item.get("POS")))),
                    ref=item.get("ref", item.get("REF")),
                    alt=item.get("alt", item.get("ALT")),
                    variant_id=variant_id,
                )
            )
        else:
            values = tuple(item)
            if len(values) not in {4, 5}:
                raise TypeError("Variant iterable items must have 4 or 5 values.")
            out.append(VariantRecord(*values))
    if not out:
        raise ValueError("No variants provided.")
    return out


class VariantEffectScorer:
    """Score REF and ALT allele effects for PEAgent model targets."""

    def __init__(
        self,
        species: str,
        model_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        genome_fasta: Optional[Union[str, Path]] = None,
        n_cells: Optional[int] = None,
        bottleneck_size: int = BOTTLENECK_SIZE,
        model=None,
    ):
        self.config = resolve_species_config(
            species=species,
            model_path=model_path,
            metadata_path=metadata_path,
            n_cells=n_cells,
            bottleneck_size=bottleneck_size,
            resolve_model=model is None,
        )
        self.genome_fasta = genome_fasta
        self._provided_model = model

    @cached_property
    def group_metadata(self) -> pd.DataFrame:
        return _load_group_metadata(self.config.metadata_path, self.config.n_cells)

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

    def _resolve_targets(self, targets=None) -> list[_TargetSpec]:
        if targets is None:
            requested = ["all"]
        elif isinstance(targets, str):
            requested = [targets]
        else:
            requested = list(targets)
        if not requested:
            raise ValueError("At least one target is required.")

        specs: list[_TargetSpec] = []
        group_df = self.group_metadata
        for target in requested:
            label = str(target).strip()
            if label.lower() == "all":
                for row in group_df.itertuples(index=False):
                    specs.append(
                        _TargetSpec(
                            target=str(row.tissue_celltype),
                            tissue=str(row.tissue),
                            celltype=str(row.celltype),
                            n_cells=int(row.n_cells),
                            indices=np.asarray(row.cell_indices, dtype=np.int32),
                        )
                    )
                continue
            if label.lower() == "global":
                specs.append(
                    _TargetSpec(
                        target="global",
                        tissue=None,
                        celltype=None,
                        n_cells=int(self.config.n_cells),
                        indices=None,
                    )
                )
                continue
            matches = group_df[group_df["tissue_celltype"].astype(str) == label]
            if matches.empty:
                preview = ", ".join(group_df["tissue_celltype"].astype(str).head(10).tolist())
                raise ValueError(
                    "Unknown variant target %r. Use 'all', 'global', or one valid tissue_celltype. "
                    "First valid tissue_celltypes: %s" % (target, preview)
                )
            row = matches.iloc[0]
            specs.append(
                _TargetSpec(
                    target=str(row["tissue_celltype"]),
                    tissue=str(row["tissue"]),
                    celltype=str(row["celltype"]),
                    n_cells=int(row["n_cells"]),
                    indices=np.asarray(row["cell_indices"], dtype=np.int32),
                )
            )
        return specs

    def _fetch_variant_pair(self, fasta, variant: VariantRecord) -> _SequencePair:
        chrom_len = int(fasta.get_reference_length(variant.chrom))
        seq_start = int(variant.pos) - 1 - SNP_OFFSET
        fetch_start = max(0, seq_start)
        fetch_end = min(chrom_len, seq_start + INPUT_LEN)
        seq = fasta.fetch(variant.chrom, fetch_start, fetch_end).upper()
        left_pad = fetch_start - seq_start
        right_pad = INPUT_LEN - left_pad - len(seq)
        if left_pad < 0 or right_pad < 0:
            raise ValueError("Bad padding for %s: left=%d right=%d" % (variant.variant_id, left_pad, right_pad))
        padded = ("N" * left_pad) + seq + ("N" * right_pad)
        if len(padded) != INPUT_LEN:
            raise ValueError("Sequence length mismatch for %s: %d" % (variant.variant_id, len(padded)))
        fasta_ref = padded[SNP_OFFSET].upper()
        if fasta_ref != variant.ref:
            raise ValueError(
                "FASTA/REF mismatch at %s:%d: fasta=%s table_ref=%s variant=%s"
                % (variant.chrom, variant.pos, fasta_ref, variant.ref, variant.variant_id)
            )

        ref_onehot = onehot_from_normalized_sequence(padded, input_len=INPUT_LEN)
        alt_onehot = ref_onehot.copy()
        alt_onehot[SNP_OFFSET, :] = 0.0
        alt_onehot[SNP_OFFSET, BASE_TO_INDEX[variant.alt]] = 1.0
        meta = {
            "variant_id": variant.variant_id,
            "chrom": variant.chrom,
            "pos": int(variant.pos),
            "ref": variant.ref,
            "alt": variant.alt,
            "seq_start_0based": int(seq_start),
            "seq_end_0based_exclusive": int(seq_start + INPUT_LEN),
            "left_pad": int(left_pad),
            "right_pad": int(right_pad),
            "fasta_ref_base": fasta_ref,
        }
        return _SequencePair(ref_onehot=ref_onehot, alt_onehot=alt_onehot, meta=meta)

    @staticmethod
    def _pair_ref_alt_records(ref_sequences: DNAInput, alt_sequences: DNAInput) -> list[_SequencePair]:
        ref_records = parse_sequences(ref_sequences, input_len=INPUT_LEN)
        alt_records = parse_sequences(alt_sequences, input_len=INPUT_LEN)
        if len(ref_records) != len(alt_records):
            raise ValueError("ref_sequences and alt_sequences must contain the same number of records.")
        pairs = []
        for ref_record, alt_record in zip(ref_records, alt_records):
            if ref_record.sequence_id != alt_record.sequence_id:
                raise ValueError(
                    "REF/ALT sequence IDs must match in order: %s != %s"
                    % (ref_record.sequence_id, alt_record.sequence_id)
                )
            meta = VariantEffectScorer._metadata_from_direct_pair(ref_record, alt_record)
            pairs.append(
                _SequencePair(
                    ref_onehot=onehot_from_normalized_sequence(ref_record.normalized_sequence, input_len=INPUT_LEN),
                    alt_onehot=onehot_from_normalized_sequence(alt_record.normalized_sequence, input_len=INPUT_LEN),
                    meta=meta,
                )
            )
        return pairs

    @staticmethod
    def _metadata_from_direct_pair(ref_record: SequenceRecord, alt_record: SequenceRecord) -> Mapping[str, object]:
        ref_seq = ref_record.normalized_sequence
        alt_seq = alt_record.normalized_sequence
        diff = [i for i, (ref_base, alt_base) in enumerate(zip(ref_seq, alt_seq)) if ref_base != alt_base]
        if len(diff) == 1:
            idx = diff[0]
            pos = idx + 1
            ref = ref_seq[idx]
            alt = alt_seq[idx]
        else:
            pos = np.nan
            ref = ""
            alt = ""
        return {
            "variant_id": ref_record.sequence_id,
            "chrom": "",
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "seq_start_0based": np.nan,
            "seq_end_0based_exclusive": np.nan,
            "left_pad": np.nan,
            "right_pad": np.nan,
            "fasta_ref_base": "",
        }

    def score_variants(
        self,
        variants,
        genome_fasta: Optional[Union[str, Path]] = None,
        targets=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> pd.DataFrame:
        """Score genomic SNVs from a FASTA reference."""

        records = parse_variant_records(variants)
        fasta_input = genome_fasta if genome_fasta is not None else self.genome_fasta
        if fasta_input is None:
            raise ValueError("genome_fasta is required for score_variants.")
        fasta, should_close = _open_fasta(fasta_input)
        try:
            pairs = [self._fetch_variant_pair(fasta, variant) for variant in records]
        finally:
            if should_close:
                fasta.close()
        return self._score_pairs(pairs, targets=targets, batch_size=batch_size)

    def score_ref_alt_sequences(
        self,
        ref_sequences: DNAInput,
        alt_sequences: DNAInput,
        targets=None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> pd.DataFrame:
        """Score already-constructed REF and ALT sequence pairs."""

        pairs = self._pair_ref_alt_records(ref_sequences, alt_sequences)
        return self._score_pairs(pairs, targets=targets, batch_size=batch_size)

    def _score_pairs(self, pairs: Sequence[_SequencePair], targets=None, batch_size: int = DEFAULT_BATCH_SIZE):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        target_specs = self._resolve_targets(targets)
        rows = []
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            x_ref = np.stack([pair.ref_onehot for pair in batch_pairs], axis=0).astype(np.float32, copy=False)
            x_alt = np.stack([pair.alt_onehot for pair in batch_pairs], axis=0).astype(np.float32, copy=False)
            pred_ref = _normalize_model_output(self.model.predict_on_batch(x_ref), self.config.n_cells)
            pred_alt = _normalize_model_output(self.model.predict_on_batch(x_alt), self.config.n_cells)
            logit_ref = _prob_to_logit(pred_ref)
            logit_alt = _prob_to_logit(pred_alt)
            for i, pair in enumerate(batch_pairs):
                rows.extend(
                    self._rows_for_pair(
                        pair.meta,
                        pred_ref[i],
                        pred_alt[i],
                        logit_ref[i],
                        logit_alt[i],
                        target_specs,
                    )
                )
        return pd.DataFrame(rows)

    def _rows_for_pair(
        self,
        meta: Mapping[str, object],
        pred_ref: np.ndarray,
        pred_alt: np.ndarray,
        logit_ref: np.ndarray,
        logit_alt: np.ndarray,
        target_specs: Sequence[_TargetSpec],
    ) -> list[Mapping[str, object]]:
        rows = []
        for target in target_specs:
            idx = slice(None) if target.indices is None else target.indices
            ref_probability = float(pred_ref[idx].mean())
            alt_probability = float(pred_alt[idx].mean())
            ref_logit = float(logit_ref[idx].mean())
            alt_logit = float(logit_alt[idx].mean())
            rows.append(
                {
                    "variant_id": meta["variant_id"],
                    "chrom": meta["chrom"],
                    "pos": meta["pos"],
                    "ref": meta["ref"],
                    "alt": meta["alt"],
                    "target": target.target,
                    "tissue": target.tissue,
                    "celltype": target.celltype,
                    "n_cells": int(target.n_cells),
                    "ref_probability": ref_probability,
                    "alt_probability": alt_probability,
                    "delta_probability": alt_probability - ref_probability,
                    "ref_logit": ref_logit,
                    "alt_logit": alt_logit,
                    "delta_logit": alt_logit - ref_logit,
                    "abs_delta_logit": abs(alt_logit - ref_logit),
                    "seq_start_0based": meta["seq_start_0based"],
                    "seq_end_0based_exclusive": meta["seq_end_0based_exclusive"],
                    "left_pad": meta["left_pad"],
                    "right_pad": meta["right_pad"],
                    "fasta_ref_base": meta["fasta_ref_base"],
                    "species": self.config.id,
                    "model_path": str(self.config.model_path),
                    "metadata_path": str(self.config.metadata_path),
                }
            )
        return rows


def score_variants(
    variants,
    species: str,
    genome_fasta: Optional[Union[str, Path]] = None,
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    targets=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Convenience wrapper for FASTA-based SNV effect scoring."""

    scorer = VariantEffectScorer(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        genome_fasta=genome_fasta,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
    )
    return scorer.score_variants(variants, targets=targets, batch_size=batch_size)


def score_ref_alt_sequences(
    ref_sequences: DNAInput,
    alt_sequences: DNAInput,
    species: str,
    model_path: Optional[Union[str, Path]] = None,
    metadata_path: Optional[Union[str, Path]] = None,
    n_cells: Optional[int] = None,
    bottleneck_size: int = BOTTLENECK_SIZE,
    targets=None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> pd.DataFrame:
    """Convenience wrapper for direct REF/ALT sequence effect scoring."""

    scorer = VariantEffectScorer(
        species=species,
        model_path=model_path,
        metadata_path=metadata_path,
        n_cells=n_cells,
        bottleneck_size=bottleneck_size,
    )
    return scorer.score_ref_alt_sequences(
        ref_sequences=ref_sequences,
        alt_sequences=alt_sequences,
        targets=targets,
        batch_size=batch_size,
    )


__all__ = [
    "SNP_OFFSET",
    "VariantEffectScorer",
    "VariantRecord",
    "parse_variant_records",
    "score_ref_alt_sequences",
    "score_variants",
]

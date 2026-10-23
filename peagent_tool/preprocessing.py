"""Lightweight preprocessing helpers for PEAgent package workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union

import numpy as np

from .celltype_prediction import _build_group_weight_matrix, _load_group_metadata
from .sequence import (
    INPUT_LEN,
    SequenceRecord,
    clean_sequence,
    normalize_sequence,
    onehot_from_normalized_sequence,
    parse_sequences,
)

DNAInput = Union[str, Mapping[str, str], Sequence[str], Iterable[Tuple[str, str]]]


def normalize_sequences(sequences: DNAInput, input_len: int = INPUT_LEN) -> list[SequenceRecord]:
    """Parse and center-normalize DNA input to the PEAgent model length."""

    return parse_sequences(sequences, input_len=input_len)


def encode_sequences(sequences: DNAInput, input_len: int = INPUT_LEN) -> np.ndarray:
    """Return a batch of one-hot encoded, normalized PEAgent inputs."""

    records = normalize_sequences(sequences, input_len=input_len)
    return np.stack(
        [
            onehot_from_normalized_sequence(record.normalized_sequence, input_len=input_len)
            for record in records
        ],
        axis=0,
    )


def load_group_metadata(path: Union[str, Path], n_cells: int):
    """Load and validate PEAgent tissue-celltype group metadata."""

    return _load_group_metadata(path, n_cells=n_cells)


def build_group_weight_matrix(group_metadata, n_cells: int) -> np.ndarray:
    """Build cell-to-group averaging weights from group metadata."""

    return _build_group_weight_matrix(group_metadata, n_cells=n_cells)


__all__ = [
    "DNAInput",
    "INPUT_LEN",
    "SequenceRecord",
    "build_group_weight_matrix",
    "clean_sequence",
    "encode_sequences",
    "load_group_metadata",
    "normalize_sequence",
    "normalize_sequences",
    "onehot_from_normalized_sequence",
    "parse_sequences",
]

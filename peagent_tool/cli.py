"""Console entry points for peagent_tool."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from peagent_tool.assets import describe_assets
from peagent_tool.attribution import compute_and_plot_attribution
from peagent_tool.celltype_prediction import list_valid_celltypes, predict_celltypes
from peagent_tool.variant_effect import score_ref_alt_sequences, score_variants


def _write_table(df, out: Optional[Path]) -> None:
    if out is None:
        print(df.to_csv(sep="\t", index=False), end="")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)


def _add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--species", "-s", required=True, help="Species: soybean, maize, or rice.")
    parser.add_argument("--metadata-path", type=Path, default=None, help="Override the default group metadata CSV.")
    parser.add_argument("--n-cells", type=int, default=None, help="Override the species output cell count.")


def _target_arg(args) -> Optional[Sequence[str]]:
    return args.target if args.target else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="peagent-tool", description="PEAgent sequence prediction utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser(
        "predict-celltypes",
        help="Predict valid tissue-celltype scores for DNA sequence input.",
    )
    _add_common_model_args(predict)
    sequence_group = predict.add_mutually_exclusive_group(required=True)
    sequence_group.add_argument("--sequence", help="Raw DNA sequence or FASTA text.")
    sequence_group.add_argument("--fasta", type=Path, help="Path to a FASTA file.")
    predict.add_argument("--model-path", type=Path, default=None, help="Override the default species model path.")
    predict.add_argument("--mode", choices=("exact", "fast"), default="exact", help="Prediction mode.")
    predict.add_argument("--top-k", type=int, default=None, help="Return only the top K tissue-celltypes per sequence.")
    predict.add_argument("--batch-size", type=int, default=32, help="Prediction batch size.")
    predict.add_argument("--out", "-o", type=Path, default=None, help="Optional TSV output path.")

    ism = subparsers.add_parser(
        "attribution",
        help="Compute and plot one contribution (attribution) map for global or one tissue-celltype target.",
    )
    _add_common_model_args(ism)
    ism_sequence_group = ism.add_mutually_exclusive_group(required=True)
    ism_sequence_group.add_argument("--sequence", help="Raw DNA sequence or FASTA text.")
    ism_sequence_group.add_argument("--fasta", type=Path, help="Path to a FASTA file.")
    ism.add_argument("--model-path", type=Path, default=None, help="Override the default species model path.")
    ism.add_argument(
        "--target",
        required=True,
        help="Target: global or an exact valid tissue_celltype from list-celltypes.",
    )
    ism.add_argument(
        "--normalize",
        choices=("none", "tissue"),
        default="none",
        help=(
            "none: raw contribution scores (global-track convention). "
            "tissue: subtract the mean over all valid tissue_celltypes of the same tissue, "
            "matching the PEAgent JBrowse cell-type attribution tracks."
        ),
    )
    ism.add_argument("--out-prefix", required=True, type=Path, help="Output prefix for matching PDF and PNG files.")
    ism.add_argument(
        "--position-batch-size",
        type=int,
        default=64,
        help="Number of sequence positions mutated per model batch.",
    )
    ism.add_argument(
        "--plot-start",
        type=int,
        default=None,
        help="1-based normalized sequence start position to plot.",
    )
    ism.add_argument(
        "--plot-end",
        type=int,
        default=None,
        help="1-based normalized sequence end position to plot.",
    )
    ism.add_argument(
        "--position-offset",
        type=int,
        default=0,
        help="Add this offset to plotted normalized sequence positions.",
    )
    ism.add_argument("--title", default=None, help="Optional custom plot title.")

    assets_cmd = subparsers.add_parser(
        "assets",
        help="Show where model weights and cell-group metadata are resolved from.",
    )
    assets_cmd.add_argument("--species", "-s", required=True, help="Species: soybean, maize, or rice.")

    list_cmd = subparsers.add_parser("list-celltypes", help="List valid tissue-celltype outputs for a species.")
    _add_common_model_args(list_cmd)
    list_cmd.add_argument("--out", "-o", type=Path, default=None, help="Optional TSV output path.")

    score_variants_cmd = subparsers.add_parser(
        "score-variants",
        help="Score biallelic SNV REF/ALT effects from a variant TSV and genome FASTA.",
    )
    _add_common_model_args(score_variants_cmd)
    score_variants_cmd.add_argument("--variants", required=True, type=Path, help="TSV/CSV with chrom, pos, ref, alt.")
    score_variants_cmd.add_argument("--fasta", required=True, type=Path, help="Reference genome FASTA.")
    score_variants_cmd.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the default species model path.",
    )
    score_variants_cmd.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target to score: all, global, or exact tissue_celltype. Repeat for multiple targets.",
    )
    score_variants_cmd.add_argument("--batch-size", type=int, default=32, help="Prediction batch size.")
    score_variants_cmd.add_argument("--out", "-o", type=Path, default=None, help="Optional TSV output path.")

    score_ref_alt_cmd = subparsers.add_parser(
        "score-ref-alt",
        help="Score REF/ALT effects from paired sequence inputs.",
    )
    _add_common_model_args(score_ref_alt_cmd)
    ref_group = score_ref_alt_cmd.add_mutually_exclusive_group(required=True)
    ref_group.add_argument("--ref-sequence", help="Raw REF DNA sequence or FASTA text.")
    ref_group.add_argument("--ref-fasta", type=Path, help="Path to REF FASTA.")
    alt_group = score_ref_alt_cmd.add_mutually_exclusive_group(required=True)
    alt_group.add_argument("--alt-sequence", help="Raw ALT DNA sequence or FASTA text.")
    alt_group.add_argument("--alt-fasta", type=Path, help="Path to ALT FASTA.")
    score_ref_alt_cmd.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Override the default species model path.",
    )
    score_ref_alt_cmd.add_argument(
        "--target",
        action="append",
        default=None,
        help="Target to score: all, global, or exact tissue_celltype. Repeat for multiple targets.",
    )
    score_ref_alt_cmd.add_argument("--batch-size", type=int, default=32, help="Prediction batch size.")
    score_ref_alt_cmd.add_argument("--out", "-o", type=Path, default=None, help="Optional TSV output path.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "predict-celltypes":
        sequence_input = args.sequence if args.sequence is not None else args.fasta.read_text()
        df = predict_celltypes(
            sequence_input,
            species=args.species,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            n_cells=args.n_cells,
            mode=args.mode,
            top_k=args.top_k,
            batch_size=args.batch_size,
        )
        _write_table(df, args.out)
        return 0

    if args.command == "attribution":
        sequence_input = args.sequence if args.sequence is not None else args.fasta.read_text()
        result, paths = compute_and_plot_attribution(
            sequence_input,
            species=args.species,
            target=args.target,
            normalize=args.normalize,
            out_prefix=args.out_prefix,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            n_cells=args.n_cells,
            position_batch_size=args.position_batch_size,
            plot_start=args.plot_start,
            plot_end=args.plot_end,
            position_offset=args.position_offset,
            title=args.title,
        )
        print("sequence_id\t%s" % result.sequence_id)
        print("species\t%s" % result.species)
        print("target\t%s" % result.target)
        print("normalize\t%s" % result.normalize)
        print("tissue_reference_groups\t%d" % len(result.tissue_reference))
        print("pdf\t%s" % paths["pdf"])
        print("png\t%s" % paths["png"])
        return 0

    if args.command == "assets":
        for key, value in describe_assets(args.species).items():
            print("%s\t%s" % (key, value))
        return 0

    if args.command == "list-celltypes":
        df = list_valid_celltypes(
            species=args.species,
            metadata_path=args.metadata_path,
            n_cells=args.n_cells,
        )
        _write_table(df, args.out)
        return 0

    if args.command == "score-variants":
        variants = pd.read_csv(args.variants, sep=None, engine="python")
        df = score_variants(
            variants,
            species=args.species,
            genome_fasta=args.fasta,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            n_cells=args.n_cells,
            targets=_target_arg(args),
            batch_size=args.batch_size,
        )
        _write_table(df, args.out)
        return 0

    if args.command == "score-ref-alt":
        ref_input = args.ref_sequence if args.ref_sequence is not None else args.ref_fasta.read_text()
        alt_input = args.alt_sequence if args.alt_sequence is not None else args.alt_fasta.read_text()
        df = score_ref_alt_sequences(
            ref_input,
            alt_input,
            species=args.species,
            model_path=args.model_path,
            metadata_path=args.metadata_path,
            n_cells=args.n_cells,
            targets=_target_arg(args),
            batch_size=args.batch_size,
        )
        _write_table(df, args.out)
        return 0

    parser.error("Unknown command: %s" % args.command)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

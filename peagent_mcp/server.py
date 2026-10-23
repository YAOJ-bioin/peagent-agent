"""PEAgent MCP server — exposes the PEAgent models, atlases and analyses as tools.

Run (stdio, for Claude Code / Codex / Claude Agent SDK)::

    python -m peagent_mcp.server

or ``fastmcp run peagent_mcp/server.py:mcp``. For a remote server on a GPU node::

    python -m peagent_mcp.server --http --host 0.0.0.0 --port 8765

Design rules (see README):
* every tool returns a small JSON summary + file paths + a ``run_id`` receipt;
  large tables stay on disk and are previewed with ``peagent_read_table``;
* TensorFlow and the model are imported/loaded lazily inside the tools, so the
  server starts instantly and ``peagent_status`` works without weights;
* coordinates are 1-based inclusive everywhere; the species' assembly is
  always echoed back so the agent cannot silently mix genomes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from pydantic import Field

from .config import ASSEMBLY, SPECIES, load_config
from .genome import INPUT_LEN, centred_window, fetch_sequence, find_gene, parse_region, reverse_complement
from .receipts import load_manifest, new_run

CFG = load_config()

mcp = FastMCP(
    name="PEAgent",
    instructions=(
        "Tools around the PEAgent sequence-to-single-cell-accessibility models for maize (B73 NAM-5.0), "
        "rice (MSU7) and soybean (Wm82.a4.v1). Load the `peagent-agent` skill first. Typical flow: peagent_gene_locus -> "
        "peagent_find_acrs (use each ACR's model_window as the region) -> peagent_predict_celltypes -> peagent_attribution (for the top cell types) -> peagent_motif_match "
        "on high-contribution segments -> peagent_score_variants / peagent_score_ref_alt for edits. Every "
        "result carries a run_id; cite it. Predictions are model outputs (evidence tier T4), not experiments."
    ),
)

# ----------------------------------------------------------------------------- helpers


def _species(species: str) -> str:
    from peagent_tool.assets import normalize_species

    return normalize_species(species)


def _model_info(species: str) -> Dict[str, Optional[str]]:
    from peagent_tool import describe_assets

    rep = describe_assets(species)
    return {"model_path": rep.get("model_path"), "metadata_path": rep.get("metadata_path")}


def _resolve_sequence(species: str, sequence: Optional[str], region: Optional[str]) -> Dict[str, Any]:
    """Return a dict with the 1,344-bp model window and where it came from."""
    if bool(sequence) == bool(region):
        raise ValueError("Give exactly one of `sequence` or `region`.")
    if region:
        paths = CFG.paths(species)
        if not paths.genome:
            raise FileNotFoundError(
                "No genome FASTA configured for %s (set PEAGENT_GENOME_%s or the YAML config)." % (species, species.upper())
            )
        chrom, start, end = parse_region(region)
        w_start, w_end = centred_window(start, end, INPUT_LEN)
        seq = fetch_sequence(paths.genome, chrom, w_start, w_end)
        return {
            "sequence": seq,
            "source": "genome",
            "assembly": paths.assembly,
            "chrom": chrom,
            "requested_start": start,
            "requested_end": end,
            "window_start": w_start,
            "window_end": w_end,
            "position_offset": w_start - 1,
            "window_label": "%s:%d-%d" % (chrom, w_start, w_end),
        }
    seq = "".join(sequence.split()).upper()
    if seq.startswith(">"):
        seq = "".join(line for line in sequence.splitlines() if not line.startswith(">")).upper()
    if len(seq) != INPUT_LEN:
        note = "sequence is %d bp; peagent_tool will centre-%s it to %d bp" % (
            len(seq),
            "crop" if len(seq) > INPUT_LEN else "pad",
            INPUT_LEN,
        )
    else:
        note = "sequence is exactly %d bp" % INPUT_LEN
    return {"sequence": seq, "source": "user", "assembly": None, "position_offset": 0, "note": note, "window_label": "user_sequence"}


def _segments(profile: np.ndarray, sequence: str, top_frac: float, min_len: int, max_gap: int, offset: int, max_segments: int) -> List[Dict[str, Any]]:
    """Contiguous runs of high |contribution| positions, merged across small gaps."""
    prof = np.asarray(profile, dtype=float)
    thr = float(np.quantile(np.abs(prof), 1.0 - top_frac)) if top_frac < 1 else 0.0
    above = np.abs(prof) >= thr
    segs: List[Dict[str, Any]] = []
    i, n = 0, len(prof)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        last_true = i
        while j < n and (above[j] or j - last_true <= max_gap):
            if above[j]:
                last_true = j
            j += 1
        s, e = i, last_true  # 0-based inclusive
        if e - s + 1 >= min_len:
            block = prof[s : e + 1]
            direction = "activating" if block.sum() > 0 else "repressive"
            segs.append(
                {
                    "start": s + 1,
                    "end": e + 1,
                    "genomic_start": s + 1 + offset if offset else None,
                    "genomic_end": e + 1 + offset if offset else None,
                    "length_bp": e - s + 1,
                    "sequence": sequence[s : e + 1],
                    "sum_contribution": round(float(block.sum()), 4),
                    "peak_contribution": round(float(block[np.abs(block).argmax()]), 4),
                    "direction": direction,
                }
            )
        i = j
    segs.sort(key=lambda d: -abs(float(d["sum_contribution"])))
    return segs[:max_segments]


def _df_records(df: pd.DataFrame, n: int) -> List[Dict[str, Any]]:
    return json.loads(df.head(n).to_json(orient="records"))


# ----------------------------------------------------------------------------- tools


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_status() -> Dict[str, Any]:
    """Report which species, model weights, genomes, annotations and motif databases this server can use.

    Call this first in a session. Loads nothing heavy."""
    from peagent_tool import __version__, describe_assets

    out: Dict[str, Any] = {"peagent_tool_version": __version__, "run_dir": str(CFG.run_dir), "species": {}}
    out["motif_db"] = str(CFG.motif_db) if CFG.motif_db and CFG.motif_db.exists() else None
    for sp in SPECIES:
        rep = describe_assets(sp)
        p = CFG.paths(sp)
        out["species"][sp] = {
            "assembly": p.assembly or ASSEMBLY[sp],
            "model_available": bool(rep.get("model_available")),
            "model_path": rep.get("model_path"),
            "metadata_available": bool(rep.get("metadata_available")),
            "genome_fasta": str(p.genome) if p.genome and p.genome.exists() else None,
            "gff3": str(p.gff) if p.gff and p.gff.exists() else None,
        }
    out["input_length_bp"] = INPUT_LEN
    out["coordinate_convention"] = "1-based inclusive; variant pos is the 1-based position of the REF base"
    return out


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_list_celltypes(
    species: Annotated[str, Field(description="maize, rice or soybean (Latin names accepted)")],
    tissue: Annotated[Optional[str], Field(description="Only groups of this tissue (case-insensitive substring)")] = None,
    pattern: Annotated[Optional[str], Field(description="Substring filter on the full tissue_celltype name")] = None,
) -> Dict[str, Any]:
    """List the valid `tissue_celltype` groups of a species' atlas (the exact strings other tools accept as targets)."""
    from peagent_tool import list_valid_celltypes

    sp = _species(species)
    df = list_valid_celltypes(sp)
    if tissue:
        df = df[df["tissue"].str.contains(tissue, case=False, na=False)]
    if pattern:
        df = df[df["tissue_celltype"].str.contains(pattern, case=False, na=False)]
    return {
        "species": sp,
        "assembly": CFG.paths(sp).assembly,
        "n_groups": int(len(df)),
        "tissues": sorted(df["tissue"].unique().tolist()),
        "groups": _df_records(df[["tissue_celltype", "tissue", "celltype", "n_cells"]], 200),
    }


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_gene_locus(
    species: str,
    gene: Annotated[str, Field(description="Gene ID (e.g. Zm00001eb054440), symbol (e.g. tb1) or alias")],
    upstream_bp: Annotated[int, Field(description="How far upstream of the TSS to report as the promoter window")] = 2000,
) -> Dict[str, Any]:
    """Resolve a gene to coordinates on the model's assembly from the configured GFF3, and propose analysis windows.

    Returns the gene body, TSS, a promoter window (upstream_bp upstream of TSS) and the 1,344-bp model window
    centred on the TSS. Symbols are matched against Name/Alias attributes, so classical names (tb1) may need the
    gene ID if the GFF lacks aliases."""
    sp = _species(species)
    gff = CFG.paths(sp).gff
    if not gff or not gff.exists():
        raise FileNotFoundError("No GFF3 configured for %s (PEAGENT_GFF_%s)." % (sp, sp.upper()))
    hits = find_gene(gff, gene)
    if not hits:
        return {"species": sp, "query": gene, "hits": [], "note": "no gene matched; try the gene ID or check the assembly"}
    results = []
    for rec in hits:
        d = rec.as_dict()
        tss = int(d["tss"])
        if rec.strand == "+":
            prom = (max(1, tss - upstream_bp), tss)
        else:
            prom = (tss, tss + upstream_bp)
        ws, we = centred_window(tss, tss, INPUT_LEN)
        d["promoter_window"] = "%s:%d-%d" % (rec.chrom, prom[0], prom[1])
        d["model_window_on_tss"] = "%s:%d-%d" % (rec.chrom, ws, we)
        results.append(d)
    return {"species": sp, "assembly": CFG.paths(sp).assembly, "query": gene, "hits": results}


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_find_acrs(
    species: str,
    region: Annotated[Optional[str], Field(description="chrom:start-end (1-based) to search")] = None,
    gene: Annotated[Optional[str], Field(description="Gene ID/symbol; searches the gene body plus flank_bp on both sides")] = None,
    flank_bp: Annotated[int, Field(description="Extra bp added on both sides of the region/gene")] = 0,
    max_hits: int = 25,
) -> Dict[str, Any]:
    """Find atlas ACRs (the peaks the model was trained on) in a region or around a gene.

    Each hit gives the ACR coordinates, its distance to the query and its `model_window`: the exact
    1,344-bp window the model saw in training. Pass `model_window` as `region` to predict/attribution
    to analyse an ACR the way the PEAgent paper did. An empty result means no atlas ACR there,
    which is itself evidence (T3) that the region is not accessible in the atlas."""
    from .acr import find_acrs

    sp = _species(species)
    if bool(region) == bool(gene):
        raise ValueError("Give exactly one of `region` or `gene`.")
    query: Dict[str, Any] = {}
    if gene:
        gff = CFG.paths(sp).gff
        if not gff or not gff.exists():
            raise FileNotFoundError("No GFF3 configured for %s (PEAGENT_GFF_%s)." % (sp, sp.upper()))
        hits = find_gene(gff, gene)
        if not hits:
            return {"species": sp, "query": gene, "acrs": [], "note": "gene not found in the GFF3"}
        g = hits[0]
        chrom, start, end = g.chrom, g.start, g.end
        query = {"gene_id": g.gene_id, "strand": g.strand, "tss": g.start if g.strand == "+" else g.end}
    else:
        chrom, start, end = parse_region(region)
    s, e = max(1, start - flank_bp), end + flank_bp
    c, acrs = find_acrs(sp, chrom, s, e, max_hits=max_hits)
    if "tss" in query:
        for a in acrs:
            centre = int(a["acr_center"])
            signed = centre - query["tss"]
            a["acr_center_relative_to_tss_bp"] = signed if query["strand"] == "+" else -signed
    return {
        "species": sp,
        "assembly": CFG.paths(sp).assembly,
        "searched": "%s:%d-%d" % (c or chrom, s, e),
        **query,
        "n_acrs": len(acrs),
        "acrs": acrs,
        "evidence_tier": "T3 atlas ACR (observed accessibility in the scATAC atlas)",
    }


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_fetch_sequence(
    species: str,
    region: Annotated[str, Field(description="chrom:start-end, 1-based inclusive, on the model's assembly")],
    exact: Annotated[bool, Field(description="If true return the region as given; if false return the 1,344-bp window centred on it")] = False,
    reverse_complement_output: bool = False,
) -> Dict[str, Any]:
    """Fetch genomic sequence from the configured reference FASTA of the species."""
    sp = _species(species)
    paths = CFG.paths(sp)
    if not paths.genome:
        raise FileNotFoundError("No genome FASTA configured for %s." % sp)
    chrom, start, end = parse_region(region)
    if not exact:
        start, end = centred_window(start, end, INPUT_LEN)
    seq = fetch_sequence(paths.genome, chrom, start, end)
    if reverse_complement_output:
        seq = reverse_complement(seq)
    gc = (seq.count("G") + seq.count("C")) / max(1, len(seq) - seq.count("N"))
    return {
        "species": sp,
        "assembly": paths.assembly,
        "region": "%s:%d-%d" % (chrom, start, end),
        "length_bp": len(seq),
        "gc": round(gc, 3),
        "n_count": seq.count("N"),
        "sequence": seq,
    }


@mcp.tool
def peagent_predict_celltypes(
    species: str,
    sequence: Annotated[Optional[str], Field(description="Raw DNA (any length; centre-cropped/padded to 1,344 bp)")] = None,
    region: Annotated[Optional[str], Field(description="chrom:start-end instead of a sequence; the 1,344-bp window centred on it is used")] = None,
    top_k: Annotated[int, Field(description="How many top cell-type groups to return (full table is saved to disk)")] = 10,
    mode: Annotated[str, Field(description="'exact' (default) or 'fast'")] = "exact",
) -> Dict[str, Any]:
    """Predict accessibility of one sequence across every tissue_celltype group of the species' single-cell atlas.

    Returns the ranked top-k groups, distribution statistics across all groups (to judge specificity), the
    run_id and the path of the full TSV. Model probabilities are evidence tier T4 (prediction)."""
    from peagent_tool import predict_celltypes

    sp = _species(species)
    src = _resolve_sequence(sp, sequence, region)
    run = new_run(CFG.run_dir, "peagent_predict_celltypes", {"species": sp, "region": region, "top_k": top_k, "mode": mode, "window": src.get("window_label")})
    run.set_model(sp, **_model_info(sp))
    df = predict_celltypes({src["window_label"]: src["sequence"]}, species=sp, mode=mode)
    df = df.sort_values("prediction", ascending=False).reset_index(drop=True)
    tsv = run.file("predictions.tsv")
    df.to_csv(tsv, sep="\t", index=False)
    run.register_output(tsv, "table", "all tissue_celltype predictions")
    fa = run.file("input.fa")
    fa.write_text(">%s\n%s\n" % (src["window_label"], src["sequence"]))
    run.register_output(fa, "fasta", "model input window")
    preds = df["prediction"].to_numpy(dtype=float)
    per_tissue = df.groupby("tissue")["prediction"].max().sort_values(ascending=False)
    summary = {
        "species": sp,
        "assembly": src.get("assembly") or CFG.paths(sp).assembly,
        "window": src.get("window_label"),
        "window_start": src.get("window_start"),
        "window_end": src.get("window_end"),
        "top": _df_records(df[["rank", "tissue_celltype", "tissue", "celltype", "prediction", "n_cells"]], top_k),
        "n_groups": int(len(df)),
        "max_prediction": round(float(preds.max()), 4),
        "median_prediction": round(float(np.median(preds)), 4),
        "n_groups_above_0.5": int((preds > 0.5).sum()),
        "specificity_ratio_top1_over_median": round(float(preds.max() / max(np.median(preds), 1e-6)), 2),
        "per_tissue_max": {k: round(float(v), 4) for k, v in per_tissue.head(12).items()},
        "files": {"predictions_tsv": str(tsv), "input_fasta": str(fa)},
        "evidence_tier": "T4 model prediction",
    }
    manifest = run.finish(summary)
    summary["run_id"] = manifest["run_id"]
    return summary


@mcp.tool
def peagent_attribution(
    species: str,
    sequence: Optional[str] = None,
    region: Annotated[Optional[str], Field(description="chrom:start-end; the 1,344-bp window centred on it is analysed")] = None,
    targets: Annotated[List[str], Field(description="['global'] or exact tissue_celltype names; all targets share one ISM pass")] = ["global"],
    normalize: Annotated[str, Field(description="'none' (raw) or 'tissue' (subtract tissue mean; browser convention; not allowed for 'global')")] = "none",
    segment_top_frac: Annotated[float, Field(description="Fraction of positions counted as high-contribution when calling segments")] = 0.03,
    min_segment_len: int = 5,
    max_segments: int = 8,
    plot: bool = True,
    variant: Annotated[Optional[Dict[str, Any]], Field(description="Optional SNV {pos (1-based), ref, alt} inside the region's window: "
                                                       "attribution is computed for BOTH alleles and the variant position's contribution and rank are reported "
                                                       "(how the paper's caQTL attribution panels were made)")] = None,
) -> Dict[str, Any]:
    """Per-base contribution scores (bottleneck in-silico mutagenesis) for one sequence and one or more cell-type targets.

    Returns, per target: the highest-contribution segments (positions, sequence, sign) — these are the pieces
    to run through peagent_motif_match — plus logo plots and a per-position TSV. One ISM pass serves all
    targets, so ask for several cell types at once. With `variant`, REF and ALT are both analysed and the
    variant position is ranked by |contribution| in each allele (a SNV that creates a motif is high in ALT,
    one that destroys a motif is high in REF). Evidence tier T4."""
    from peagent_tool import ContributionScorer, plot_attribution

    sp = _species(species)
    if normalize == "tissue" and any(t.strip().lower() == "global" for t in targets):
        raise ValueError("normalize='tissue' cannot be used with the 'global' target; drop it or use normalize='none'.")
    src = _resolve_sequence(sp, sequence, region)
    run = new_run(CFG.run_dir, "peagent_attribution", {"species": sp, "region": region, "targets": targets, "normalize": normalize, "window": src.get("window_label"), "variant": variant})
    run.set_model(sp, **_model_info(sp))
    scorer = ContributionScorer(sp)
    results = scorer.compute_many({src["window_label"]: src["sequence"]}, targets=targets, normalize=normalize)
    offset = int(src.get("position_offset") or 0)
    norm_seq = next(iter(results.values())).to_dataframe()  # includes the normalized sequence positions
    seq_norm = "".join(norm_seq["base"].tolist()) if "base" in norm_seq.columns else src["sequence"][:INPUT_LEN]
    alt_results = {}
    var_off = None
    if variant:
        if not region:
            raise ValueError("`variant` needs `region` (genomic coordinates) to locate the SNV in the window.")
        var_off = int(variant["pos"]) - (offset + 1)
        if not 0 <= var_off < len(seq_norm):
            raise ValueError("variant pos %s is outside the window %s" % (variant["pos"], src.get("window_label")))
        if seq_norm[var_off] != str(variant["ref"]).upper():
            raise ValueError("REF mismatch: window base at %s is %s, variant says %s" % (variant["pos"], seq_norm[var_off], variant["ref"]))
        alt_seq = seq_norm[:var_off] + str(variant["alt"]).upper() + seq_norm[var_off + 1:]
        alt_results = scorer.compute_many({src["window_label"] + "_ALT": alt_seq}, targets=targets, normalize=normalize)
    per_target: Dict[str, Any] = {}
    frames = []
    for target, res in results.items():
        df = res.to_dataframe(position_offset=offset)
        df.insert(0, "target", target)
        frames.append(df)
        segs = _segments(res.profile, seq_norm, segment_top_frac, min_segment_len, max_gap=2, offset=offset, max_segments=max_segments)
        entry: Dict[str, Any] = {
            "segments": segs,
            "total_positive_contribution": round(float(np.clip(res.profile, 0, None).sum()), 4),
            "total_negative_contribution": round(float(np.clip(res.profile, None, 0).sum()), 4),
        }
        if plot:
            safe = target.replace("/", "_").replace(" ", "_")
            out_prefix = run.file("attribution_%s" % safe)
            try:
                paths = plot_attribution(res, out_prefix=out_prefix, position_offset=offset, title="%s | %s | %s" % (sp, src.get("window_label"), target))
                for p in (paths if isinstance(paths, (list, tuple)) else [paths]):
                    run.register_output(Path(p), "figure", "contribution logo for %s" % target)
                entry["plot_png"] = str(out_prefix) + ".png"
            except Exception as exc:  # plotting must not kill the analysis
                run.notes.append("plot failed for %s: %s" % (target, exc))
        if variant:
            alt = alt_results[target]
            adf = alt.to_dataframe(position_offset=offset)
            adf.insert(0, "target", target)
            df.insert(1, "allele", "REF")
            adf.insert(1, "allele", "ALT")
            frames.append(adf)

            def _rank(profile):
                ab = np.abs(np.asarray(profile, dtype=float))
                r = int((ab > ab[var_off]).sum()) + 1
                return {"contribution": round(float(profile[var_off]), 5), "abs_rank": r, "of": int(len(ab)),
                        "percentile": round(100.0 * (1 - r / len(ab)), 2)}

            entry["variant_position"] = {"pos": int(variant["pos"]), "window_index_1based": var_off + 1,
                                         "REF": _rank(res.profile), "ALT": _rank(alt.profile),
                                         "context_REF": seq_norm[max(0, var_off - 8):var_off] + seq_norm[var_off].lower() + seq_norm[var_off + 1:var_off + 9],
                                         "context_ALT": seq_norm[max(0, var_off - 8):var_off] + str(variant["alt"]).lower() + seq_norm[var_off + 1:var_off + 9]}
            entry["segments_ALT"] = _segments(alt.profile, alt_seq, segment_top_frac, min_segment_len, max_gap=2, offset=offset, max_segments=max_segments)
            if plot:
                try:
                    pfx = run.file("attribution_%s_ALT" % target.replace("/", "_").replace(" ", "_"))
                    plot_attribution(alt, out_prefix=pfx, position_offset=offset, title="%s | %s ALT | %s" % (sp, src.get("window_label"), target))
                    entry["plot_png_ALT"] = str(pfx) + ".png"
                except Exception as exc:  # noqa: BLE001
                    run.notes.append("ALT plot failed for %s: %s" % (target, exc))
        per_target[target] = entry
    table = pd.concat(frames, ignore_index=True)
    tsv = run.file("contributions.tsv")
    table.to_csv(tsv, sep="\t", index=False)
    run.register_output(tsv, "table", "per-position contribution scores for all targets")
    summary = {
        "species": sp,
        "assembly": src.get("assembly") or CFG.paths(sp).assembly,
        "window": src.get("window_label"),
        "position_offset": offset,
        "normalize": normalize,
        "targets": per_target,
        "files": {"contributions_tsv": str(tsv)},
        "how_to_read": "segments are 1-based positions in the 1,344-bp window (genomic_* when a region was given); "
        "positive = bases that increase predicted accessibility of the target; run peagent_motif_match on segment sequences",
        "evidence_tier": "T4 model attribution",
    }
    manifest = run.finish(summary)
    summary["run_id"] = manifest["run_id"]
    return summary


@mcp.tool
def peagent_score_variants(
    species: str,
    variants: Annotated[List[Dict[str, Any]], Field(description="List of {chrom, pos (1-based), ref, alt, variant_id?} on the model's assembly")],
    targets: Annotated[List[str], Field(description="['global'], ['all'] or exact tissue_celltype names")] = ["global"],
    top_k_celltypes: Annotated[int, Field(description="Per variant, how many most-affected groups to report when targets include 'all'")] = 8,
    window: Annotated[str, Field(description="'variant' (default, PEAgent paper caQTL protocol): 1,344-bp window centred on the variant; "
                                 "'acr': the training window of the atlas ACR containing the variant (falls back to variant-centred)")] = "variant",
) -> Dict[str, Any]:
    """Score biallelic SNVs (REF vs ALT) against the reference genome.

    By default the window is centred on the variant, as in the PEAgent paper's caQTL validation (Fig. 5A).
    window='acr' instead uses the atlas ACR's training window, useful to compare with ACR-level attribution.
    The FASTA base is checked against the declared REF; mismatches are reported, not silently flipped.
    Returns delta_logit / delta_probability per variant (and per cell type if requested). Evidence tier T4."""
    from peagent_tool import score_ref_alt_sequences, score_variants

    from .acr import acr_containing

    sp = _species(species)
    paths = CFG.paths(sp)
    if not paths.genome:
        raise FileNotFoundError("No genome FASTA configured for %s; use peagent_score_ref_alt with explicit sequences instead." % sp)
    if window not in ("acr", "variant"):
        raise ValueError("window must be 'acr' or 'variant'")
    run = new_run(CFG.run_dir, "peagent_score_variants", {"species": sp, "n_variants": len(variants), "targets": targets, "window": window})
    run.set_model(sp, **_model_info(sp))
    vdf = pd.DataFrame(variants)
    for col in ("chrom", "pos", "ref", "alt"):
        if col not in vdf.columns:
            raise ValueError("variants need columns chrom,pos,ref,alt (+ optional variant_id); missing %s" % col)
    if "variant_id" not in vdf.columns:
        vdf["variant_id"] = ["%s_%s_%s_%s" % tuple(r) for r in vdf[["chrom", "pos", "ref", "alt"]].astype(str).itertuples(index=False)]
    tgt = targets if targets != ["all"] else "all"
    vdf["pos"] = vdf["pos"].astype(int)
    window_info: Dict[str, Dict[str, Any]] = {}
    acr_ref, acr_alt, centred = {}, {}, []
    for r in vdf.itertuples(index=False):
        acr = acr_containing(sp, str(r.chrom), int(r.pos)) if window == "acr" else None
        if acr is None or len(str(r.ref)) != 1 or len(str(r.alt)) != 1:
            fb = fetch_sequence(paths.genome, str(r.chrom), int(r.pos), int(r.pos))
            ok = fb == str(r.ref).upper()
            hit = acr if acr is not None else acr_containing(sp, str(r.chrom), int(r.pos))
            window_info[r.variant_id] = {"window_mode": "variant_centred", "in_atlas_acr": hit["acr"] if hit else None,
                                         "fasta_ref_base": fb, "ref_matches_fasta": ok}
            if ok:
                centred.append(r.variant_id)  # mismatches are reported, not scored
            continue
        wchrom, wrange = str(acr["model_window"]).split(":")
        ws1, we1 = (int(x) for x in wrange.split("-"))
        seq = fetch_sequence(paths.genome, wchrom, ws1, we1)
        off = int(r.pos) - ws1
        fasta_ref = seq[off]
        info = {"window_mode": "acr_centred", "acr": acr["acr"], "model_window": acr["model_window"],
                "offset_from_acr_center_bp": acr["offset_from_acr_center_bp"], "fasta_ref_base": fasta_ref,
                "ref_matches_fasta": fasta_ref == str(r.ref).upper()}
        window_info[r.variant_id] = info
        if not info["ref_matches_fasta"]:
            continue  # reported, not scored
        acr_ref[r.variant_id] = seq
        acr_alt[r.variant_id] = seq[:off] + str(r.alt).upper() + seq[off + 1:]
    frames = []
    if acr_ref:
        d = score_ref_alt_sequences(acr_ref, acr_alt, species=sp, targets=tgt)
        d["variant_id"] = d["variant_id"] if "variant_id" in d.columns else d.iloc[:, 0]
        frames.append(d)
    if centred:
        frames.append(score_variants(vdf[vdf["variant_id"].isin(centred)], species=sp, genome_fasta=paths.genome, targets=tgt))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["variant_id"])
    meta = vdf.set_index("variant_id")[["chrom", "pos", "ref", "alt"]]
    for col in ("chrom", "pos", "ref", "alt"):
        df[col] = df["variant_id"].map(meta[col])
    df["window_mode"] = df["variant_id"].map(lambda v: window_info.get(v, {}).get("window_mode"))
    tsv = run.file("variant_effects.tsv")
    df.to_csv(tsv, sep="\t", index=False)
    run.register_output(tsv, "table", "REF/ALT effects")
    group_col = next((c for c in ("tissue_celltype", "target", "group") if c in df.columns), None)
    per_variant = []
    for vid, sub in df.groupby("variant_id", sort=False):
        sub = sub.copy()
        if "delta_logit" in sub.columns:
            sub["abs_delta_logit"] = sub["delta_logit"].abs()
            sub = sub.sort_values("abs_delta_logit", ascending=False)
        cols = [c for c in (group_col, "ref_probability", "alt_probability", "delta_probability", "delta_logit") if c and c in sub.columns]
        entry: Dict[str, Any] = {"variant_id": vid, "n_targets_scored": int(len(sub)), **window_info.get(vid, {})}
        if group_col:
            glob = sub[sub[group_col].astype(str).str.lower() == "global"]
            if len(glob):
                entry["global"] = _df_records(glob[cols], 1)[0]
            entry["most_affected"] = _df_records(sub[sub[group_col].astype(str).str.lower() != "global"][cols], top_k_celltypes)
        else:
            entry["effects"] = _df_records(sub[cols], top_k_celltypes)
        per_variant.append(entry)
    scored = set(df["variant_id"].astype(str))
    for vid, info in window_info.items():
        if str(vid) not in scored:
            per_variant.append({"variant_id": vid, "scored": False, **info,
                                "note": "REF does not match the reference FASTA; not scored (check assembly / strand / allele polarity)"})
    summary = {
        "species": sp,
        "assembly": paths.assembly,
        "targets": targets,
        "window": window,
        "variants": per_variant,
        "files": {"variant_effects_tsv": str(tsv)},
        "how_to_read": "delta = ALT - REF; positive means ALT predicted more accessible; |delta_logit| ranks effect size; "
        "compare against caQTL/MPRA evidence before calling a variant functional",
        "evidence_tier": "T4 model prediction",
    }
    manifest = run.finish(summary)
    summary["run_id"] = manifest["run_id"]
    return summary


@mcp.tool
def peagent_score_ref_alt(
    species: str,
    ref_sequence: str,
    alt_sequence: str,
    targets: List[str] = ["global"],
    top_k_celltypes: int = 8,
) -> Dict[str, Any]:
    """Score a designed edit: REF and ALT sequences you constructed (same length preferred), e.g. a motif deletion or insertion."""
    from peagent_tool import score_ref_alt_sequences

    sp = _species(species)
    run = new_run(CFG.run_dir, "peagent_score_ref_alt", {"species": sp, "targets": targets, "ref_len": len(ref_sequence), "alt_len": len(alt_sequence)})
    run.set_model(sp, **_model_info(sp))
    tgt = targets if targets != ["all"] else "all"
    df = score_ref_alt_sequences({"edit": ref_sequence}, {"edit": alt_sequence}, species=sp, targets=tgt)
    tsv = run.file("ref_alt_effects.tsv")
    df.to_csv(tsv, sep="\t", index=False)
    run.register_output(tsv, "table", "designed REF/ALT effects")
    if "delta_logit" in df.columns:
        df = df.reindex(df["delta_logit"].abs().sort_values(ascending=False).index)
    cols = [c for c in ("tissue_celltype", "target", "ref_probability", "alt_probability", "delta_probability", "delta_logit") if c in df.columns]
    summary = {"species": sp, "effects": _df_records(df[cols], top_k_celltypes), "files": {"tsv": str(tsv)}, "evidence_tier": "T4 model prediction"}
    manifest = run.finish(summary)
    summary["run_id"] = manifest["run_id"]
    return summary


PAPER_SPACINGS = list(range(0, 31)) + list(range(35, 201, 5))  # centre-to-centre grid of the PEAgent paper


def _pooled_readout_tester(sp: str, groups: List[str]):
    """MotifSynergyTester whose readout averages weights/bias over the union of cells of several groups
    (the paper's pooled 'soybean endosperm' readout = mean of per-cell pre-sigmoid logits)."""
    from peagent_tool.synergy import MotifSynergyTester

    class _Pooled(MotifSynergyTester):
        def target_readout(self, target: str):
            if not str(target).startswith("pool:"):
                return super().target_readout(target)
            meta = self.scorer.group_metadata
            idx = np.unique(np.concatenate([np.asarray(r, dtype=np.int64) for r in meta[meta["tissue_celltype"].isin(groups)]["cell_indices"]]))
            kernel = self.scorer.dense_kernel
            return kernel[:, idx].mean(axis=1).astype(np.float32), float(self.dense_bias[idx].mean())

    return _Pooled(sp)


def _load_backgrounds(sp: str, kind: str, n: int, gc_content: float, seed: int) -> List[str]:
    from peagent_tool.synergy import random_backgrounds

    if kind == "random":
        return random_backgrounds(n, gc_content=gc_content, seed=seed)
    from peagent_tool.assets import resolve_asset

    path = resolve_asset("backgrounds/%s_synergy_backgrounds.fa" % sp)
    seqs = [l.strip() for l in open(path) if l.strip() and not l.startswith(">")]
    return seqs[:n]


@mcp.tool
def peagent_motif_synergy(
    species: str,
    motif_a: Annotated[str, Field(description="Motif A sequence. Prefer the paper's lexicon consensus (peagent_search_lexicon -> query_consensus), e.g. CATTAATT")],
    motif_b: str,
    target: Annotated[str, Field(description="'global', one exact tissue_celltype, or 'pool:<substring>' to pool every group whose tissue_celltype contains the substring (e.g. 'pool:Endosperm' = all endosperm cells, as in the paper)")] = "global",
    spacings: Annotated[Optional[List[int]], Field(description="Centre-to-centre spacings (bp). Default = the paper grid: 0-30 by 1, 35-200 by 5")] = None,
    orientations: Optional[List[str]] = None,
    backgrounds: Annotated[str, Field(description="'paper' (default): the paper's 100 GC-matched non-accessible genomic windows of this species; 'random': i.i.d. sequence at gc_content")] = "paper",
    n_backgrounds: int = 100,
    gc_content: float = 0.42,
    seed: int = 2026,
    alpha: float = 0.01,
) -> Dict[str, Any]:
    """In-silico marginalisation test of whether two motifs act synergistically (joint effect > sum of singles) in a cell type,
    over a spacing x orientation grid with a one-sided paired Wilcoxon test and BH correction. Defaults reproduce the
    PEAgent paper's protocol (paper backgrounds, paper spacing grid); compare with peagent_paper_table('S3'/'S4')."""
    sp = _species(species)
    groups: List[str] = []
    if target.startswith("pool:"):
        from peagent_tool import list_valid_celltypes

        allg = list_valid_celltypes(sp)
        groups = allg[allg["tissue_celltype"].str.contains(target[5:], case=False, regex=False)]["tissue_celltype"].tolist()
        if not groups:
            raise ValueError("No tissue_celltype contains %r" % target[5:])
        n_cells_pool = int(allg[allg["tissue_celltype"].isin(groups)]["n_cells"].sum())
    grid = spacings or PAPER_SPACINGS
    run = new_run(CFG.run_dir, "peagent_motif_synergy", {"species": sp, "motif_a": motif_a, "motif_b": motif_b, "target": target,
                                                           "pooled_groups": groups, "spacings": grid, "orientations": orientations,
                                                           "backgrounds": backgrounds, "n_backgrounds": n_backgrounds, "alpha": alpha, "seed": seed})
    run.set_model(sp, **_model_info(sp))
    tester = _pooled_readout_tester(sp, groups)
    bgs = _load_backgrounds(sp, backgrounds, n_backgrounds, gc_content, seed)
    kwargs = {"orientations": orientations} if orientations else {}
    result = tester.test_pairs([{"motif_a": motif_a, "motif_b": motif_b}], target=target, backgrounds=bgs, spacings=grid, alpha=alpha, **kwargs)
    tsv = run.file("synergy_arrangements.tsv")
    result.arrangements.to_csv(tsv, sep="\t", index=False)
    run.register_output(tsv, "table", "all arrangements")
    sig = result.significant(alpha)
    arr = result.arrangements
    top = arr.sort_values("mean_synergy", ascending=False)
    summary = {
        "species": sp,
        "target": target,
        "pooled_groups": groups or None,
        "pooled_n_cells": n_cells_pool if groups else None,
        "backgrounds": "%s (n=%d)" % (backgrounds, len(bgs)),
        "n_arrangements": int(len(arr)),
        "skipped_spacings_note": "spacings where the two motifs would overlap (centre distance < mean motif length) are skipped",
        "single_effects": {"mean_delta_a": round(float(arr["mean_delta_a"].iloc[0]), 4), "mean_delta_b": round(float(arr["mean_delta_b"].iloc[0]), 4)} if len(arr) else None,
        "top_by_synergy": _df_records(top[["orientation", "spacing", "edge_gap", "mean_delta_joint", "mean_synergy", "p", "padj"]], 5),
        "n_significant": int(len(sig)),
        "significant": _df_records(sig.sort_values("mean_synergy", ascending=False)[["orientation", "spacing", "edge_gap", "mean_synergy", "padj"]], 10),
        "files": {"tsv": str(tsv)},
        "how_to_read": "synergy = dJ - (dA + dB) on the pre-sigmoid logit, per background; significant = padj < alpha. Orientation '-+' = A reverse, B forward.",
        "evidence_tier": "T4 model prediction",
    }
    manifest = run.finish(summary)
    summary["run_id"] = manifest["run_id"]
    return summary


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_search_lexicon(
    query: Annotated[Optional[str], Field(description="Substring of label, TF family, best-match TF or consensus (e.g. 'TCP', 'HD-ZIP', 'GGGCC')")] = None,
    species: Annotated[Optional[str], Field(description="Only patterns found in this species")] = None,
    tissue_celltype: Annotated[Optional[str], Field(description="Only patterns found in groups whose name contains this (e.g. 'Endosperm')")] = None,
    pattern_type: Annotated[Optional[str], Field(description="base, base_with_flank, homocomposite, heterocomposite, unresolved")] = None,
    max_rows: int = 20,
) -> Dict[str, Any]:
    """Search the PEAgent paper's lexicon of 243 cell-type-resolved contribution meta-patterns (Table S2).

    Returns each pattern's type, TF family, best match, consensus (use it as a motif for synergy or edits)
    and the cell types it was discovered in. Evidence tier T4 (model-derived patterns) with T5 TF annotation."""
    from .paper import load_table

    lex = load_table("S2")
    ct = load_table("lexicon_celltypes")
    hits = lex
    if query:
        q = query.lower()
        cols = ["label", "tf_family", "tf_subfamily", "best_match_tf", "query_consensus", "zone1_tf", "zone2_tf"]
        mask = np.zeros(len(hits), dtype=bool)
        for c in cols:
            mask |= hits[c].astype(str).str.lower().str.contains(q, regex=False).to_numpy()
        hits = hits[mask]
    if pattern_type:
        hits = hits[hits["pattern_type"].astype(str).str.lower() == pattern_type.lower()]
    sub = ct
    if species:
        sub = sub[sub["species"].str.lower() == _species(species)]
    if tissue_celltype:
        sub = sub[sub["tissue_celltype"].str.contains(tissue_celltype, case=False, regex=False)]
    if species or tissue_celltype:
        hits = hits[hits["unique_id"].isin(sub["unique_id"])]
    hits = hits.sort_values("total_seqlets", ascending=False)
    rows = []
    for r in hits.head(max_rows).itertuples(index=False):
        where = sub[sub["unique_id"] == r.unique_id].sort_values("num_seqlets", ascending=False)
        rows.append({
            "unique_id": int(r.unique_id), "label": r.label, "pattern_type": r.pattern_type, "tf_family": r.tf_family,
            "best_match_tf": r.best_match_tf, "annotation_quality": r.annotation_quality, "consensus": r.query_consensus,
            "species": r.species_list, "n_celltypes_total": int(r.n_celltypes), "total_seqlets": int(r.total_seqlets),
            "found_in_top": [{"species": w.species, "tissue_celltype": w.tissue_celltype, "seqlets": int(w.num_seqlets)} for w in where.head(6).itertuples(index=False)],
        })
    return {"n_matches": int(len(hits)), "patterns": rows, "source": "PEAgent paper Table S2 (lexicon) + provenance",
            "evidence_tier": "T4 model-derived pattern; TF family is T5 (TomTom similarity)"}


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_paper_table(
    table: Annotated[str, Field(description="S1 performance, S2 lexicon, S3 synergy (best cell type), S4 endosperm synergy, S5 rice ACR conservation, S6 cell-type ontology, lexicon_celltypes; or 'list'")],
    filters: Annotated[Optional[Dict[str, str]], Field(description="{column: substring} or numeric '>x', '<=x', '=x'")] = None,
    columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
    max_rows: int = 30,
    describe_columns: bool = False,
) -> Dict[str, Any]:
    """Query the PEAgent paper's published results so answers can be compared with (or grounded in) the paper.

    Use it to check whether the paper already reports a result (e.g. a motif pair's synergy, a cell type's
    AUROC) before re-computing, and to cite 'PEAgent paper Table Sx' as the source."""
    from .paper import PAPER_TABLES, column_descriptions, filter_table, load_table

    if table == "list":
        return {"tables": {k: v["about"] for k, v in PAPER_TABLES.items()}}
    df = filter_table(load_table(table), filters or {})
    if sort_by:
        df = df.sort_values(sort_by, ascending=not descending)
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    out: Dict[str, Any] = {"table": table, "about": PAPER_TABLES[table]["about"], "n_rows": int(len(df)),
                           "columns": df.columns.tolist(), "rows": _df_records(df, max_rows),
                           "source": "PEAgent paper (Yao et al., under review), Table %s" % table}
    if describe_columns:
        out["column_descriptions"] = column_descriptions(table)
    return out


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_motif_match(
    segment: Annotated[str, Field(description="Short DNA segment (6-60 bp), e.g. a high-contribution segment from peagent_attribution")],
    top_n: int = 5,
    min_relative_score: float = 0.80,
) -> Dict[str, Any]:
    """Match a short segment against the configured JASPAR plant PWM database (first-pass TF family candidates, not proof of binding)."""
    from .motifs import load_jaspar, scan_segment

    if not CFG.motif_db or not CFG.motif_db.exists():
        return {"hits": [], "note": "No motif database configured (PEAGENT_MOTIF_DB); download JASPAR CORE plants pfm text to enable."}
    motifs = load_jaspar(str(CFG.motif_db))
    hits = scan_segment(segment, motifs, top_n=top_n, min_relative_score=min_relative_score)
    return {
        "segment": segment.upper(),
        "n_motifs_scanned": len(motifs),
        "hits": hits,
        "evidence_tier": "T5 sequence similarity to a known motif; confirm with DAP-seq/ChIP (T2) before claiming the TF binds",
    }


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_get_run(run_id: str) -> Dict[str, Any]:
    """Return the provenance manifest (parameters, model hash, output files with sha256, versions) of a previous run_id."""
    return load_manifest(CFG.run_dir, run_id)


@mcp.tool(annotations={"readOnlyHint": True, "idempotentHint": True})
def peagent_read_table(
    path: Annotated[str, Field(description="A TSV path returned by another tool (must be inside the run directory)")],
    max_rows: int = 40,
    columns: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    descending: bool = True,
) -> Dict[str, Any]:
    """Preview rows of a result table without flooding the context."""
    p = Path(path).resolve()
    if CFG.run_dir.resolve() not in p.parents:
        raise ValueError("Only files inside the run directory can be read with this tool.")
    df = pd.read_csv(p, sep="\t")
    if columns:
        df = df[[c for c in columns if c in df.columns]]
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)
    return {"path": str(p), "n_rows_total": int(len(df)), "columns": df.columns.tolist(), "rows": _df_records(df, max_rows)}


# ----------------------------------------------------------------------------- entry point


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PEAgent MCP server")
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP instead of stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.http:
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main(sys.argv[1:])

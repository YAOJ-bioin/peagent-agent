#!/usr/bin/env python
"""Smoke test for the PEAgent MCP server without model weights or TensorFlow.

Builds a throw-away asset root (fake group metadata, a tiny genome FASTA + GFF3 and a two-motif
JASPAR file), points the server at it, and exercises every tool that does not need the model:
status, list_celltypes, gene_locus, fetch_sequence, motif_match, read_table/get_run guards.
Also checks that the MCP tool list is what the skills expect.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_TOOLS = {
    "peagent_status",
    "peagent_list_celltypes",
    "peagent_gene_locus",
    "peagent_find_acrs",
    "peagent_fetch_sequence",
    "peagent_predict_celltypes",
    "peagent_attribution",
    "peagent_score_variants",
    "peagent_score_ref_alt",
    "peagent_motif_synergy",
    "peagent_motif_match",
    "peagent_search_lexicon",
    "peagent_paper_table",
    "peagent_get_run",
    "peagent_read_table",
}


def make_fake_assets(tmp: Path) -> dict:
    asset_root = tmp / "assets"
    (asset_root / "metadata").mkdir(parents=True)
    (asset_root / "models" / "maize").mkdir(parents=True)
    # group metadata: tissue_celltype,tissue,celltype,n_cells,group_idx,cell_indices
    rows = []
    idx = 0
    for tissue, cts in {"crownRoot": ["trichoblast", "cortex", "endodermis"], "leaf": ["mesophyll", "bundleSheath"]}.items():
        for ct in cts:
            cells = list(range(idx * 10, idx * 10 + 10))
            rows.append("%s_%s,%s,%s,%d,%d,\"%s\"" % (tissue, ct, tissue, ct, len(cells), idx, json.dumps(cells)))
            idx += 1
    for sp in ("maize", "rice", "soybean"):
        (asset_root / "metadata" / ("%s_group_metadata.csv" % sp)).write_text(
            "tissue_celltype,tissue,celltype,n_cells,group_idx,cell_indices\n" + "\n".join(rows) + "\n"
        )
    # genome: one 20 kb contig with a planted TCP motif; bgzip + faidx via pysam
    random.seed(1)
    seq = "".join(random.choice("ACGT") for _ in range(20000))
    seq = seq[:10000] + "GGACCA" + seq[10006:]
    fa_plain = tmp / "genome.fa"
    fa_plain.write_text(">chr1\n" + "\n".join(seq[i : i + 80] for i in range(0, len(seq), 80)) + "\n")
    import pysam

    fa_gz = tmp / "genome.fa.gz"
    pysam.tabix_compress(str(fa_plain), str(fa_gz), force=True)
    pysam.faidx(str(fa_gz))
    gff = tmp / "genes.gff3.gz"
    with gzip.open(gff, "wt") as fh:
        fh.write("##gff-version 3\n")
        fh.write("chr1\ttest\tgene\t12001\t15000\t.\t+\t.\tID=gene:Zm00001ebTEST01;Name=tb1;biotype=protein_coding\n")
        fh.write("chr1\ttest\tgene\t3000\t5000\t.\t-\t.\tID=Zm00001ebTEST02;Name=zmm28\n")
    jaspar = tmp / "motifs.jaspar"
    jaspar.write_text(
        ">MA_TEST.1 TCP\nA [ 0 0 20 0 0 20 ]\nC [ 0 0 0 20 20 0 ]\nG [ 20 20 0 0 0 0 ]\nT [ 0 0 0 0 0 0 ]\n"
        ">MA_TEST.2 Gbox\nA [ 0 20 0 0 0 0 ]\nC [ 20 0 20 0 0 0 ]\nG [ 0 0 0 20 0 20 ]\nT [ 0 0 0 0 20 0 ]\n"
    )
    cfg = tmp / "peagent_agent.yaml"
    cfg.write_text(
        "run_dir: %s\nmotif_db: %s\nspecies:\n  maize:\n    genome: %s\n    gff: %s\n    assembly: FAKE-v0\n"
        % (tmp / "runs", jaspar, fa_gz, gff)
    )
    return {"PEAGENT_ASSET_ROOT": str(asset_root), "PEAGENT_AGENT_CONFIG": str(cfg)}


async def exercise(env: dict) -> None:
    os.environ.update(env)
    sys.path.insert(0, str(ROOT))
    from fastmcp import Client

    from peagent_mcp import server

    async with Client(server.mcp) as client:
        tools = {t.name for t in await client.list_tools()}
        missing = EXPECTED_TOOLS - tools
        assert not missing, "missing tools: %s" % missing
        print("tools OK:", len(tools))

        st = (await client.call_tool("peagent_status", {})).data
        assert st["species"]["maize"]["genome_fasta"], st
        print("status OK: maize genome", st["species"]["maize"]["genome_fasta"])

        ct = (await client.call_tool("peagent_list_celltypes", {"species": "Zea mays", "tissue": "crownRoot"})).data
        assert ct["n_groups"] == 3, ct
        print("list_celltypes OK:", [g["tissue_celltype"] for g in ct["groups"]])

        gl = (await client.call_tool("peagent_gene_locus", {"species": "maize", "gene": "tb1"})).data
        assert gl["hits"][0]["gene_id"] == "Zm00001ebTEST01" and gl["hits"][0]["tss"] == 12001, gl
        print("gene_locus OK:", gl["hits"][0]["model_window_on_tss"])

        fs = (await client.call_tool("peagent_fetch_sequence", {"species": "maize", "region": "chr1:10001-10006", "exact": True})).data
        assert fs["sequence"] == "GGACCA", fs
        fw = (await client.call_tool("peagent_fetch_sequence", {"species": "maize", "region": "chr1:10001-10006"})).data
        assert fw["length_bp"] == 1344, fw
        print("fetch_sequence OK:", fs["sequence"], fw["region"])

        mm = (await client.call_tool("peagent_motif_match", {"segment": "TTGGACCATT"})).data
        assert mm["hits"] and mm["hits"][0]["name"] == "TCP", mm
        print("motif_match OK:", mm["hits"][0]["name"], mm["hits"][0]["relative_score"])

        # guards
        try:
            await client.call_tool("peagent_read_table", {"path": "/etc/passwd"})
            raise AssertionError("read_table should refuse paths outside run_dir")
        except Exception as exc:  # noqa: BLE001
            assert "run directory" in str(exc) or "Only files" in str(exc), exc
        print("read_table guard OK")

        # model-dependent tools should fail cleanly without weights, not crash the server
        try:
            await client.call_tool("peagent_predict_celltypes", {"species": "maize", "sequence": "ACGT" * 336})
            print("predict ran (weights present?)")
        except Exception as exc:  # noqa: BLE001
            print("predict without weights fails cleanly:", str(exc)[:120])


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        env = make_fake_assets(Path(d))
        asyncio.run(exercise(env))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()

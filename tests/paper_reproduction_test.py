#!/usr/bin/env python
"""Does the MCP synergy tool reproduce the paper's Table S4 numbers? (soybean endosperm, HD-ZIP x TCP)"""
import asyncio, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastmcp import Client  # noqa: E402
from peagent_mcp import server  # noqa: E402


async def main():
    async with Client(server.mcp) as c:
        lx = (await c.call_tool("peagent_search_lexicon", {"query": "HD-ZIP", "tissue_celltype": "Endosperm", "max_rows": 3})).data
        print(json.dumps(lx["patterns"][:2], indent=1)[:1500])
        paper = (await c.call_tool("peagent_paper_table", {"table": "S4", "filters": {"pair_id": "1_15"}})).data["rows"]
        paper = [r for r in paper if r["pair_id"] == "1_15"][0]
        print("PAPER S4 1_15:", {k: paper[k] for k in ("motifA_label", "motifB_label", "opt_orientation", "opt_spacing_center_bp", "synergy_effect", "padj_BH")})
        a = paper["motifA_label"].split(":")[1]; b = paper["motifB_label"].split(":")[1]
        r = (await c.call_tool("peagent_motif_synergy", {"species": "soybean", "motif_a": a, "motif_b": b, "target": "pool:Endosperm"})).data
        print("TOOL:", json.dumps({k: r[k] for k in ("pooled_groups", "pooled_n_cells", "backgrounds", "n_arrangements", "top_by_synergy", "n_significant", "run_id")}, indent=1))
        import pandas as pd
        arr = pd.read_csv(r["files"]["tsv"], sep="\t")
        row = arr[(arr.orientation == paper["opt_orientation"]) & (arr.spacing == paper["opt_spacing_center_bp"])].iloc[0]
        print("TOOL at paper's optimum %s/%s: synergy=%.6f vs paper %.6f ; joint=%.6f vs paper %.6f" % (
            row.orientation, row.spacing, row.mean_synergy, paper["synergy_effect"], row.mean_delta_joint, paper["joint_dJ_opt"]))


asyncio.run(main())

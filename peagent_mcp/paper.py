"""The PEAgent paper's own results as queryable tables (Supplementary Tables S1-S6 + lexicon provenance).

Resolved like model weights: $PEAGENT_ASSET_ROOT/paper/<file> -> cache -> Hugging Face.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict

import pandas as pd

PAPER_TABLES: Dict[str, Dict[str, str]] = {
    "S1": {"file": "Table_S1_celltype_prediction_performance.tsv", "about": "Held-out AUROC and PCC of every tissue_celltype group (Fig. 1)"},
    "S2": {"file": "Table_S2_metapattern_lexicon.tsv", "about": "The 243 cell-type-resolved contribution meta-patterns: type, TF family, best match, consensus, species (Fig. 2)"},
    "S3": {"file": "Table_S3_motif_pair_synergy.tsv", "about": "In-silico motif-pair synergy in each pair's best cell type, optimal spacing/orientation (Fig. 3)"},
    "S4": {"file": "Table_S4_endosperm_motif_pair_synergy.tsv", "about": "Soybean endosperm synergy case study, 36 pairs, pooled 8,511 endosperm cells (Fig. 3)"},
    "S5": {"file": "Table_S5_rice_ACR_conservation_classification.tsv", "about": "945 rice ACRs: grass-conserved vs rice-specific, PEAgent vs phyloP/phastCons (Fig. 4)"},
    "S6": {"file": "Table_S6_celltype_anatomy_stage_classification.tsv", "about": "Anatomy / developmental-stage ontology of the 419 tissue_celltype labels"},
    "lexicon_celltypes": {"file": "lexicon_celltypes.tsv", "about": "Which species_tissue_celltype each S2 meta-pattern was found in, with seqlet counts"},
}


@lru_cache(maxsize=16)
def load_table(name: str) -> pd.DataFrame:
    from peagent_tool.assets import resolve_asset

    if name not in PAPER_TABLES:
        raise ValueError("Unknown table %r; choose from %s" % (name, ", ".join(PAPER_TABLES)))
    return pd.read_csv(resolve_asset("paper/" + PAPER_TABLES[name]["file"]), sep="\t")


@lru_cache(maxsize=8)
def column_descriptions(name: str) -> Dict[str, str]:
    from peagent_tool.assets import resolve_asset

    if not name.startswith("S"):
        return {}
    f = PAPER_TABLES[name]["file"].split("_")[:2]
    try:
        df = pd.read_csv(resolve_asset("paper/%s_%s_column_descriptions.tsv" % tuple(f)), sep="\t")
        return dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str)))
    except Exception:  # noqa: BLE001
        return {}


def filter_table(df: pd.DataFrame, filters: Dict[str, str]) -> pd.DataFrame:
    """Case-insensitive substring match per column; numeric columns accept '>x', '<x', '>=x', '<=x', '=x'."""
    out = df
    for col, cond in (filters or {}).items():
        if col not in out.columns:
            raise ValueError("No column %r. Columns: %s" % (col, ", ".join(out.columns)))
        cond = str(cond).strip()
        if pd.api.types.is_numeric_dtype(out[col]) and cond[:1] in "<>=":
            for op in (">=", "<=", ">", "<", "="):
                if cond.startswith(op):
                    v = float(cond[len(op):])
                    s = out[col]
                    out = out[{">=": s >= v, "<=": s <= v, ">": s > v, "<": s < v, "=": s == v}[op]]
                    break
        else:
            out = out[out[col].astype(str).str.contains(cond, case=False, na=False, regex=False)]
    return out

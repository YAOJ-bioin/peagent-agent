---
name: peagent-celltype-specificity
description: Determine in which tissues and cell types a sequence or region is predicted accessible, how specific that pattern is, and what distinguishes the top cell type from its tissue, using the PEAgent atlases (maize 94, rice 128, soybean 98 tissue_celltype groups). Use for "is this element root-specific / mesophyll-specific", comparing two sequences' specificity, or choosing cell types for a reporter assay.
---

# Cell-type specificity

## Procedure
1. `peagent_list_celltypes(species, tissue=...)` to know the group vocabulary; quote exact names.
2. `peagent_predict_celltypes(..., top_k=15)`; read `per_tissue_max`, `n_groups_above_0.5`,
   `specificity_ratio_top1_over_median`.
3. Classify (state the rule you applied):
   - **specific**: top-1 ≥ 0.5, ≤ 5 groups ≥ 0.5, ratio ≥ 3, and the ≥ 0.5 groups share a
     tissue or a lineage (e.g. all epidermal);
   - **tissue-restricted**: ≥ 0.5 groups confined to one tissue but many cell types;
   - **broad**: ≥ 0.5 in ≥ 3 tissues;
   - **not accessible**: max < 0.3.
4. For specific/tissue-restricted elements, `peagent_attribution` with the top group and a
   sibling group of the same tissue, `normalize="tissue"`, to find the bases that make the top
   cell type different; `peagent_motif_match` the top segments.
5. If two sequences are compared, run both with identical settings and report the
   per-group differences for the groups where either is ≥ 0.3 (use `peagent_read_table` on
   both TSVs rather than eyeballing the top-k lists).
6. Report: the classification, the supporting numbers, the distinguishing segments/motifs, and
   the caveat that predictions reflect the training atlas (tissues/stages sampled) — absence of
   a tissue in the atlas is not evidence of inaccessibility there.

## Pitfalls
- The three atlases have different tissue sets; do not compare raw probabilities across species.
- `fast` mode is for screening many sequences; use `exact` for anything you will report.

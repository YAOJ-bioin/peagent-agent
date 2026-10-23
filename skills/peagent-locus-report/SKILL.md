---
name: peagent-locus-report
description: Build an evidence-tiered regulatory report for a gene or genomic region in maize, rice or soybean using the PEAgent tools - cell-type accessibility prediction, per-base attribution, motif candidates, optional variant overlay. Use when asked which cell types a region/promoter is active in, what sequence features drive it, or to explain a regulatory mechanism at a locus (e.g. "what drives tb1's upstream enhancer?").
---

# Locus report

## Inputs you need
- `species` (maize | rice | soybean) and a locus: gene ID/symbol **or** `chrom:start-end`.
- Optional: a cell type / tissue of interest, known variants, a hypothesis to test.

## Procedure
1. `peagent_status` → confirm model, genome and GFF for the species. If GFF is missing and the
   user gave a gene symbol, ask for coordinates or the gene ID.
2. `peagent_gene_locus` → coordinates, TSS, `model_window_on_tss`. For enhancer questions
   (e.g. "the ~60 kb upstream control region"), build the region explicitly from the TSS and
   the stated distance and **say which strand you assumed**.
3. `peagent_find_acrs(gene=..., flank_bp=5000)` (or `region=`): list the atlas ACRs near the
   locus. These are the observed open regions (T3) and their `model_window` is the exact window
   the model was trained on, so prefer ACR windows over arbitrary TSS-centred windows. If there is
   no ACR, say so: predictions on a never-open window are extrapolation.
4. `peagent_predict_celltypes(region=<model_window>, top_k=10)` for each candidate window (promoter, and
   any distal element). Record `run_id`, top groups, `specificity_ratio_top1_over_median`,
   `n_groups_above_0.5`. Interpret: ratio > 3 and few groups > 0.5 = cell-type-specific;
   many groups > 0.5 = broadly accessible; max < 0.3 = not predicted accessible (do not
   over-interpret attribution on such windows).
5. `peagent_attribution(region=..., targets=[top 2–3 tissue_celltype, plus one contrasting
   low-ranked group], normalize="tissue")` — one call, several targets. Also run
   `targets=["global"], normalize="none"` if the question is about accessibility in general.
6. For the top 3–5 segments per target: `peagent_motif_match(segment)`. Report only hits with
   relative_score ≥ 0.85 as "candidate <family> site"; below that say "no confident motif match".
7. Optional overlays: known SNVs → `peagent_score_variants`; a designed deletion of a segment →
   `peagent_score_ref_alt` (ALT = window with the segment replaced by shuffled or N bases of
   equal length; keep length constant).
8. Literature: if web access is available, search for the locus' known regulators/perturbation
   evidence (T1/T2) and cite DOIs; otherwise state that literature was not consulted.
9. Draft the report in the report format of the `peagent-agent` skill; send to the `verifier` subagent if > 3 claims.

## Interpretation guardrails
- Attribution is relative to the model's prediction for *this* window; a strong segment in a
  window predicted inaccessible everywhere means little.
- Tissue-normalized scores highlight what makes a cell type *different from its tissue*; raw
  (`none`) scores show what makes the window accessible at all. Say which you used.
- Motif similarity (T5) is not binding (T2). Phrase as "segment resembles an X-family motif".
- Negative (repressive) segments are real model outputs but harder to interpret; mention them
  only when strong and consistent across targets.

## Output checklist
- [ ] species + assembly named; window coordinates 1-based.
- [ ] top cell types with probabilities and the specificity statistics.
- [ ] segments with positions, sequences, direction, motif candidates (or "none").
- [ ] every number traceable to a `run_id`.
- [ ] tiers assigned; caveats; next experiments (e.g. DAP-seq of candidate TF, reporter assay).

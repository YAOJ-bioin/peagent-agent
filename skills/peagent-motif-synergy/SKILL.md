---
name: peagent-motif-synergy
description: Test whether two TF motifs cooperate (synergy beyond additivity) in a cell type by in-silico marginalisation with PEAgent over spacing and orientation, with Wilcoxon + BH statistics, using the PEAgent paper's protocol. Use for questions about motif pairs, spacing/orientation grammar, composite elements, or whether a partner TF explains cell-type specificity.
---

# Motif synergy

## 1. Check what the paper already found
`peagent_paper_table("S3", filters={"motifA_family": "<fam>"})` (best cell type per pair) and, for
soybean endosperm, `peagent_paper_table("S4")`. If the pair is there, report the paper's optimum
(orientation, centre spacing, synergy, padj) and say you are re-testing it, not discovering it.

## 2. Choose the motifs
Use the paper's own motifs: `peagent_search_lexicon(query="<TF family>", species=...,
tissue_celltype=...)` and take `consensus` of the base pattern found in the cell type of
interest. Only if the lexicon has nothing, fall back to `peagent_motif_match`/JASPAR consensus,
and state which sequence you used and why.

## 3. Choose the readout
- One group: an exact `tissue_celltype` (`peagent_list_celltypes`).
- A cell type spread over several groups/stages (e.g. all endosperm): `target="pool:<substring>"`
  pools every matching group's cells, the paper's readout. Report `pooled_groups` and
  `pooled_n_cells`.

## 4. Run
`peagent_motif_synergy(species, motif_a, motif_b, target)` with defaults = paper protocol:
100 GC-matched non-accessible genomic backgrounds of that species, centre-to-centre spacing
0-30 bp by 1 and 35-200 bp by 5, all orientations, one-sided paired Wilcoxon, BH. Takes about
1 min on CPU. Do not shrink the grid or the backgrounds for a reportable result.

## 5. Read
- synergy = dJ - (dA + dB) on the logit. Report the best arrangement (orientation, spacing,
  edge gap), its synergy, padj, and the number of significant arrangements at alpha = 0.01.
- Check `single_effects`: if a motif alone barely moves the logit, a synergy call is fragile.
- Syntax: a sharp optimum at short spacing (< 20 bp) with orientation dependence = hard syntax
  (composite-like); broad, orientation-insensitive gains at 20-150 bp = soft syntax.

## 6. Scope and tier
- Answer for the readout the user asked about. When re-testing a paper result, use the paper's
  readout and report agreement with the paper's table; do not widen the claim.
- A contrasting readout (another cell type, or `global`) is optional: run it only when the user
  asks whether the grammar is cell-type specific, and then say the paper did not claim specificity.
- All results are T4. T2 follow-ups: EMSA/DAP-seq with both factors; reporter with spacing and
  orientation variants.

## Pitfalls
- Orientation '-+' means motif A reverse-complemented, motif B forward.
- Motifs are placed by their centre (paper Methods). padj is BH over this call's arrangements;
  the paper corrected over all pairs at once, so padj values differ while synergy values match.
- Quote padj, never raw p.

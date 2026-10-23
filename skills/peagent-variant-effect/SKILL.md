---
name: peagent-variant-effect
description: Score and interpret the predicted effect of SNVs or designed sequence edits on cell-type-resolved chromatin accessibility with PEAgent (REF vs ALT in a 1,344-bp window). Use for caQTL/GWAS variants, allele comparisons, or "what happens if we mutate this motif" questions in maize, rice or soybean.
---

# Variant / edit effect

## Inputs
- species; variants as chrom, pos (1-based), ref, alt **on the model assembly** (maize B73
  NAM-5.0, rice MSU7, soybean Wm82.a4.v1); or REF/ALT sequences for designed edits.
- Optional: cell types of interest; the trait/phenotype context.

## Procedure
1. `peagent_status` (genome configured?). If the variants are on another assembly, stop and
   say so — do not guess coordinates.
2. `peagent_find_acrs(region=<variant>)`: is the variant inside an atlas ACR? The paper analysed
   SNVs located within ACRs; a variant outside every ACR sits in sequence the atlas never saw
   open, so say so before interpreting any score.
3. `peagent_score_variants(variants, targets=["all"])` for <= 50 variants (batch larger sets).
   The default window is centred on the variant, the paper's caQTL protocol (Fig. 5A).
   Variants with `scored: false` failed the REF check: report them, never flip alleles.
   Rank by `|delta_logit|`; report the most affected cell types with signs.
   What the paper's caQTL validation established (Fig. 5B; state it, do not overstate it): across 13 maize
   seedling cell types, fine-mapped caQTL SNVs had larger |delta_logit| than same-ACR non-causal
   controls in 92.3% of cell types, and the predicted direction was more often right for
   large-effect than small-effect variants. So a large |delta_logit| prioritises a variant; it
   does not prove it causal, and small effects carry little directional information.
4. Mechanism: `peagent_attribution(region=<variant>, targets=[most affected cell types],
   variant={"pos":..,"ref":..,"alt":..})` analyses BOTH alleles in one call. Read
   `variant_position`: a variant that CREATES a site ranks high in ALT, one that DESTROYS a
   site ranks high in REF (the paper's Fig. 5C/F logic). Then `peagent_motif_match` on the
   ALT context (creation) or REF context (disruption). A motif change at a high-attribution
   position is a mechanistic hypothesis (T4 + T5), not a result.
5. If the question concerns a causal claim (e.g. "is this the causal caQTL/GWAS variant?"),
   look for T1/T2 evidence (fine-mapping, allelic imbalance, reporter assays, DAP-seq of the
   implicated TF) in the literature; otherwise say the model can only prioritise.
6. Report per the report format of the `peagent-agent` skill; always include the REF check and the assembly.

## Designed edits (`peagent_score_ref_alt`)
- Keep REF and ALT the same length (replace, do not delete) unless the question is about an
  indel; state the length change if any.
- Test the edit in the cell types where REF is accessible; an edit cannot be interpreted in a
  window predicted inaccessible.
- For motif knock-outs, mutate the core bases of the matched motif (from `peagent_motif_match`),
  and include a scrambled-control edit elsewhere in the window as a negative control.

## Pitfalls
- 0/1-based confusion: `pos` is the 1-based position of the REF base.
- Multi-allelic or indel variants are not supported by `peagent_score_variants`; build sequences.
- Strand: the model is strand-aware only through the genome FASTA orientation; do not reverse
  complement unless you are deliberately testing orientation.

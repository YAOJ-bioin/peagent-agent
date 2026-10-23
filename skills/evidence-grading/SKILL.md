---
name: evidence-grading
description: Assign evidence tiers (T1 genetic perturbation, T2 biochemical/functional assay, T3 observational genomics, T4 model prediction, T5 inference/homology) to every claim in a plant gene-regulation answer and write the evidence table. Use whenever a final report is being assembled or reviewed, and when judging whether a statement is over-claimed.
---

# Evidence grading

## Tier definitions (same as the `peagent-agent` skill)
- **T1** genetic/transgenic perturbation changes expression or phenotype (mutant, CRISPR edit,
  transgenic reporter driving expression in planta, fine-mapped QTL with a functional test).
- **T2** biochemical or functional assay at the element (DAP-seq/ChIP-seq occupancy, EMSA,
  STARR-seq/MPRA activity, caQTL with allelic imbalance).
- **T3** observational genomics association (ACR present in the atlas, expression
  correlation, sequence conservation, co-accessibility).
- **T4** model prediction (PEAgent probability, attribution, in-silico variant or synergy).
- **T5** inference or homology (motif similarity, ortholog transfer, reasoning by analogy).

## Procedure
1. Split the draft into atomic claims (one subject, one predicate).
2. For each claim find the *strongest* source actually in hand; assign its tier. If the only
   source is the model, the tier is T4 even if the literature "probably" agrees.
3. Rewrite over-claims: "X binds the enhancer" (needs T2) → "the segment resembles an X-family
   motif (T5) and contributes to the predicted accessibility (T4)".
4. Flag claims with no source as **unsupported** — they must be removed or marked as hypothesis.
5. Produce the table: claim | tier | source (`run_id` / DOI / PMID) | note.
6. Add the "what would raise the tier" line for every T4/T5 claim that matters to the answer.

## Common misgradings
- Treating a published *prediction* (another model's output in a paper) as T2/T3.
- Treating atlas accessibility (T3) as proof the element is functional (needs T1/T2).
- Treating a caQTL association (T3) as causality (needs allelic-imbalance/fine-mapping T2, or
  editing T1).

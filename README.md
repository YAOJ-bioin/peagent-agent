# PEAgent

**Sequence-to-single-cell chromatin accessibility for plant genomes, as an AI agent.**

PEAgent models read 1,344 bp of DNA and predict accessibility in every cell type of
single-cell ATAC atlases of maize, rice and soybean, and which bases drive it
(Yao *et al.*, under review; portal: [peagent.org](https://peagent.org)).

This repository is the single entry point. It contains:

- **An agent for Claude Code.** Ask in plain language. The agent plans the analysis, runs the
  models, checks every number against the run records and answers with an evidence-tiered
  report (T1 genetics … T4 model prediction, T5 motif similarity).
- **The `peagent_tool` Python library and `peagent-tool` command line** for scripted and
  high-throughput use.

| Species | Assembly | Nuclei | Cell-type groups |
|---|---|---:|---:|
| Maize (*Zea mays*) | B73 NAM-5.0 | 50,639 | 94 |
| Rice (*Oryza sativa*) | Nipponbare MSU7 | 104,029 | 128 |
| Soybean (*Glycine max*) | Wm82.a4.v1 | 200,732 | 98 |

## Use it as an agent

Requirements: [Claude Code](https://claude.com/claude-code) 2.1.275+, [uv](https://docs.astral.sh/uv/), ~2 GB disk. CPU is enough.

```bash
# 1. Model access. Weights are pre-publication and gated: request access at
#    https://huggingface.co/YAOJ-bioin/peagent , then
export HF_TOKEN=hf_xxx

# 2. Optional, for questions about genes, regions or variants: genome + gene annotation on the
#    model assembly (bgzipped or plain FASTA with .fai; GFF3)
export PEAGENT_GENOME_MAIZE=/path/Zm-B73-REFERENCE-NAM-5.0.fa.gz
export PEAGENT_GFF_MAIZE=/path/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.gz
#    likewise _RICE (MSU7 osa1_r7) and _SOYBEAN (Wm82.a4.v1)

# 3. Install the plugin (inside Claude Code)
/plugin install peagent --marketplace YAOJ-bioin/peagent-agent
```

The first session installs the Python environment (about 1-5 minutes, once) and downloads
the weights of the species you use. Then just ask, for example:

- "In maize, what does the SNV chr8:176,677,455 T>C do to accessibility, and in which cell types?"
- "大豆胚乳中 HD-ZIP 和 TCP motif 是否协同？最佳间距和方向是什么？"
- "Which atlas ACRs lie near LOC_Os01g01010 in rice, and in which cell types are they predicted open?"
- "Paste a sequence: where is it accessible, and which bases drive it?"

Two complete example sessions, with the agent's reports and how they compare with the paper,
are in [`examples/`](examples/).

### What the agent can do

| Capability | Tool | Paper |
|---|---|---|
| Cell-type accessibility of any sequence, gene window or ACR | `peagent_predict_celltypes` | Fig. 1 |
| Per-base contribution maps, several cell types in one pass; REF and ALT of a variant | `peagent_attribution` | Fig. 2, 5 |
| Atlas ACRs near a gene or in a region, with the model's training window | `peagent_find_acrs` | Fig. 1 |
| SNV effects (REF/ALT check) and designed edits | `peagent_score_variants`, `peagent_score_ref_alt` | Fig. 5 |
| Motif-pair synergy over spacing x orientation, paper protocol | `peagent_motif_synergy` | Fig. 3 |
| The 243-pattern regulatory lexicon and where each pattern is used | `peagent_search_lexicon` | Fig. 2 |
| The paper's result tables S1-S6, to compare against | `peagent_paper_table` | all |
| Gene lookup, sequence fetch, JASPAR motif match, run records | `peagent_gene_locus`, `peagent_fetch_sequence`, `peagent_motif_match`, `peagent_get_run`, `peagent_read_table` | |

Procedures and interpretation rules live in [`skills/`](skills/); sub-agents (`analyst`,
`literature`, `verifier`) in [`agents/`](agents/). Every tool call writes a `run_id` record
(parameters, model sha256, output hashes) to `~/peagent_runs`.

The tools are a standard MCP server, so other MCP clients (Codex CLI, Gemini CLI, Cursor,
Claude Desktop) can use them too:

```bash
uv run --directory /path/to/peagent-agent peagent-mcp                      # stdio
uv run --directory /path/to/peagent-agent peagent-mcp --http --port 8765    # e.g. on a GPU node
```

## Use it as a Python library

```bash
pip install "git+https://github.com/YAOJ-bioin/peagent-agent"     # Python 3.10 or 3.11
```

```python
from peagent_tool import predict_celltypes, compute_attribution, score_variants

predict_celltypes(sequence, species="maize", top_k=5)
res = compute_attribution(sequence, species="maize", target="leaf_guard_mother_cell", normalize="tissue")
score_variants(variants_df, species="maize", genome_fasta="Zm-B73-REFERENCE-NAM-5.0.fa.gz", targets="all")
```

```bash
peagent-tool list-celltypes    --species maize
peagent-tool predict-celltypes --species maize --fasta seqs.fa --top-k 10
peagent-tool attribution       --species maize --fasta acr.fa --target crownRoot_trichoblast --normalize tissue --out-prefix contrib
peagent-tool score-variants    --species maize --variants snvs.tsv --fasta genome.fa.gz
```

## Models and data

Weights, cell-group metadata, the atlas ACR sets, synergy backgrounds and the paper tables are
resolved in this order: `$PEAGENT_ASSET_ROOT` (a local copy) → `~/.cache/peagent` →
the gated Hugging Face repository (`$PEAGENT_HF_REPO`, default `YAOJ-bioin/peagent`, needs `HF_TOKEN`).

| Variable | Meaning |
|---|---|
| `HF_TOKEN`, `PEAGENT_HF_REPO` | Access to the gated model repository |
| `PEAGENT_ASSET_ROOT` | Local copy of the assets instead of downloading |
| `PEAGENT_GENOME_<SPECIES>`, `PEAGENT_GFF_<SPECIES>` | Reference FASTA and GFF3 on the model assembly |
| `PEAGENT_MOTIF_DB` | JASPAR-format PFMs (default: bundled JASPAR 2026 CORE plants non-redundant, CC BY 4.0) |
| `PEAGENT_RUN_DIR` | Results and run records (default `~/peagent_runs`) |
| `PEAGENT_AGENT_CONFIG` | One YAML file with all of the above ([example](peagent_agent.example.yaml)) |

## Citation

Yao J., Li J., Zhang X., Li X., Marand A.P., Pickering E., Schmitz R.J. Sequence-based modeling of
plant epigenomes reveals cell-type-specific cis-regulatory grammar. Under review (2026).

MIT License. The model-construction code adapted from scBasset is BSD-3 ([notices](THIRD_PARTY_NOTICES.md)).

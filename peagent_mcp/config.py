"""Configuration for the PEAgent MCP server.

Resolution order for every setting: explicit YAML config file
(``$PEAGENT_AGENT_CONFIG``), then environment variables, then defaults.

Example ``peagent_agent.yaml``::

    run_dir: /scratch/jy16611/peagent-agent/runs
    motif_db: /scratch/jy16611/resources/JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.txt
    species:
      maize:
        genome: /scratch/jy16611/genomes/Zm-B73-REFERENCE-NAM-5.0.fa.gz
        gff: /scratch/jy16611/genomes/Zm-B73-REFERENCE-NAM-5.0_Zm00001eb.1.gff3.gz
        assembly: B73 NAM-5.0
      rice:
        genome: /scratch/jy16611/genomes/IRGSP-1.0_MSU7.fa.gz
        gff: ...
      soybean:
        genome: ...

Environment-variable equivalents: ``PEAGENT_RUN_DIR``, ``PEAGENT_MOTIF_DB``,
``PEAGENT_GENOME_MAIZE`` / ``PEAGENT_GFF_MAIZE`` (and ``_RICE`` / ``_SOYBEAN``).
Model weights are resolved by ``peagent_tool`` itself (``$PEAGENT_ASSET_ROOT``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

SPECIES = ("maize", "rice", "soybean")

BUNDLED_MOTIF_DB = Path(__file__).resolve().parent / "data" / "JASPAR2026_CORE_plants_nonredundant.jaspar"

ASSEMBLY = {
    "maize": "Zm-B73-REFERENCE-NAM-5.0",
    "rice": "Nipponbare MSU7 (osa1 r7)",
    "soybean": "Wm82.a4.v1",
}


@dataclass
class SpeciesPaths:
    genome: Optional[Path] = None
    gff: Optional[Path] = None
    assembly: str = ""


@dataclass
class AgentConfig:
    # absolute default: `uv run --directory` (plugin launch) changes cwd to the plugin cache
    run_dir: Path = field(default_factory=lambda: Path("~/peagent_runs").expanduser())
    motif_db: Optional[Path] = None
    species: Dict[str, SpeciesPaths] = field(default_factory=dict)

    def paths(self, species: str) -> SpeciesPaths:
        return self.species.get(species, SpeciesPaths(assembly=ASSEMBLY.get(species, "")))


def _env_path(name: str) -> Optional[Path]:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def load_config() -> AgentConfig:
    cfg = AgentConfig()
    # 1. YAML file
    yaml_path = _env_path("PEAGENT_AGENT_CONFIG")
    data: dict = {}
    if yaml_path and yaml_path.exists():
        import yaml  # local import: optional dependency

        with open(yaml_path) as fh:
            data = yaml.safe_load(fh) or {}
    if data.get("run_dir"):
        cfg.run_dir = Path(str(data["run_dir"])).expanduser()
    if data.get("motif_db"):
        cfg.motif_db = Path(str(data["motif_db"])).expanduser()
    for sp, entry in (data.get("species") or {}).items():
        sp = str(sp).lower()
        if sp not in SPECIES:
            continue
        cfg.species[sp] = SpeciesPaths(
            genome=Path(entry["genome"]).expanduser() if entry.get("genome") else None,
            gff=Path(entry["gff"]).expanduser() if entry.get("gff") else None,
            assembly=str(entry.get("assembly") or ASSEMBLY[sp]),
        )
    # 2. environment variables fill gaps
    if env := _env_path("PEAGENT_RUN_DIR"):
        cfg.run_dir = env
    if env := _env_path("PEAGENT_MOTIF_DB"):
        cfg.motif_db = cfg.motif_db or env
    for sp in SPECIES:
        sp_paths = cfg.species.get(sp, SpeciesPaths(assembly=ASSEMBLY[sp]))
        sp_paths.genome = sp_paths.genome or _env_path("PEAGENT_GENOME_%s" % sp.upper())
        sp_paths.gff = sp_paths.gff or _env_path("PEAGENT_GFF_%s" % sp.upper())
        sp_paths.assembly = sp_paths.assembly or ASSEMBLY[sp]
        cfg.species[sp] = sp_paths
    if cfg.motif_db is None:
        # bundled default: JASPAR 2026 CORE plants non-redundant (927 PFMs, CC BY 4.0, jaspar.elixir.no)
        cfg.motif_db = BUNDLED_MOTIF_DB
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    return cfg

"""Resolution and download of PEAgent runtime assets (model weights, cell-group metadata).

PEAgent Tool ships code only. Model weights and cell-group metadata are runtime
assets that are resolved, in order, from:

1. an explicit ``model_path`` / ``metadata_path`` argument,
2. ``$PEAGENT_ASSET_ROOT`` (a local directory laid out as below),
3. the local cache ``$PEAGENT_CACHE_DIR`` (default ``~/.cache/peagent``),
4. a Hugging Face Hub download, when ``huggingface_hub`` is installed and the
   asset repository is published.

Local layout::

    <root>/
    ├── models/<species>/model.h5
    └── metadata/<species>_group_metadata.csv
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Union

#: Hugging Face repository holding the published PEAgent assets.
#: Override with ``$PEAGENT_HF_REPO`` until the final repository id is fixed.
DEFAULT_HF_REPO = "YAOJ-bioin/peagent"

#: Revision (branch, tag or commit) fetched from the Hugging Face repository.
DEFAULT_HF_REVISION = "main"

SPECIES_IDS = ("soybean", "maize", "rice")

_SPECIES_ALIASES = {
    "glycine max": "soybean",
    "glycine_max": "soybean",
    "soy": "soybean",
    "gmax": "soybean",
    "zea mays": "maize",
    "zea_mays": "maize",
    "corn": "maize",
    "oryza sativa": "rice",
    "oryza_sativa": "rice",
}


def normalize_species(species: str) -> str:
    """Map a species name or alias onto one of ``soybean`` / ``maize`` / ``rice``."""

    key = str(species).strip().lower().replace("-", "_")
    key = _SPECIES_ALIASES.get(key, key)
    if key not in SPECIES_IDS:
        raise KeyError("Unknown species %r. Supported species: %s" % (species, ", ".join(SPECIES_IDS)))
    return key


def hf_repo_id() -> str:
    """Return the Hugging Face repository id used for asset downloads."""

    return os.environ.get("PEAGENT_HF_REPO", DEFAULT_HF_REPO)


def hf_revision() -> str:
    """Return the Hugging Face revision used for asset downloads."""

    return os.environ.get("PEAGENT_HF_REVISION", DEFAULT_HF_REVISION)


def cache_dir() -> Path:
    """Return the local cache directory for downloaded assets."""

    return Path(os.environ.get("PEAGENT_CACHE_DIR", Path.home() / ".cache" / "peagent"))


def asset_root() -> Optional[Path]:
    """Return ``$PEAGENT_ASSET_ROOT`` if it is set, else ``None``."""

    # PEAGENT_BACKEND_ROOT is the pre-0.2 name, still honoured.
    root = os.environ.get("PEAGENT_ASSET_ROOT") or os.environ.get("PEAGENT_BACKEND_ROOT")
    return Path(root) if root else None


def relative_model_path(species: str) -> str:
    """Return the repository-relative path of a species model file."""

    return "models/%s/model.h5" % normalize_species(species)


def relative_metadata_path(species: str) -> str:
    """Return the repository-relative path of a species cell-group metadata file."""

    return "metadata/%s_group_metadata.csv" % normalize_species(species)


def _search_paths(relative: str) -> list:
    candidates = []
    root = asset_root()
    if root is not None:
        candidates.append(root / relative)
    candidates.append(cache_dir() / relative)
    return candidates


def _download_from_hub(relative: str) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise FileNotFoundError(
            "Asset %r was not found locally and huggingface_hub is not installed.\n"
            "Either install the download extra:\n"
            '    pip install huggingface_hub\n'
            "or point PEAGENT_ASSET_ROOT at a directory containing models/ and metadata/." % relative
        ) from exc

    repo = hf_repo_id()
    destination = cache_dir() / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = hf_hub_download(repo_id=repo, filename=relative, revision=hf_revision())
    except Exception as exc:  # noqa: BLE001 - surface any hub error with actionable context
        raise FileNotFoundError(
            "Could not download %r from Hugging Face repository %r (revision %r): %s\n"
            "If the assets are not published yet, set PEAGENT_ASSET_ROOT to a local "
            "directory holding models/<species>/model.h5 and "
            "metadata/<species>_group_metadata.csv." % (relative, repo, hf_revision(), exc)
        ) from exc
    shutil.copyfile(downloaded, destination)
    return destination


def resolve_asset(relative: str, explicit: Optional[Union[str, Path]] = None, download: bool = True) -> Path:
    """Resolve one asset path, optionally downloading it into the local cache.

    Parameters
    ----------
    relative:
        Repository-relative path, e.g. ``models/maize/model.h5``.
    explicit:
        A user-supplied path. When given it is returned as-is and must exist.
    download:
        Whether to fall back to a Hugging Face download when no local copy is found.
    """

    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError("Asset path does not exist: %s" % path)
        return path

    for candidate in _search_paths(relative):
        if candidate.exists():
            return candidate

    if not download:
        searched = "\n  ".join(str(candidate) for candidate in _search_paths(relative))
        raise FileNotFoundError("Asset %r not found. Searched:\n  %s" % (relative, searched))

    return _download_from_hub(relative)


def resolve_model_path(species: str, explicit: Optional[Union[str, Path]] = None, download: bool = True) -> Path:
    """Resolve the model weights file for one species."""

    return resolve_asset(relative_model_path(species), explicit=explicit, download=download)


def resolve_metadata_path(species: str, explicit: Optional[Union[str, Path]] = None, download: bool = True) -> Path:
    """Resolve the cell-group metadata file for one species."""

    return resolve_asset(relative_metadata_path(species), explicit=explicit, download=download)


def fetch_species_assets(species: str, download: bool = True) -> Dict[str, Path]:
    """Resolve both assets for one species and return them as a mapping."""

    return {
        "model": resolve_model_path(species, download=download),
        "metadata": resolve_metadata_path(species, download=download),
    }


def describe_assets(species: str) -> Dict[str, object]:
    """Report where each asset for a species would be read from, without downloading."""

    key = normalize_species(species)
    report: Dict[str, object] = {
        "species": key,
        "asset_root": str(asset_root()) if asset_root() else None,
        "cache_dir": str(cache_dir()),
        "hf_repo": hf_repo_id(),
        "hf_revision": hf_revision(),
    }
    for name, relative in (("model", relative_model_path(key)), ("metadata", relative_metadata_path(key))):
        found = next((c for c in _search_paths(relative) if c.exists()), None)
        report["%s_relative" % name] = relative
        report["%s_path" % name] = str(found) if found else None
        report["%s_available" % name] = found is not None
    return report

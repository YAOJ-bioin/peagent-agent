"""Run directories and provenance receipts.

Every tool call that produces a result gets its own ``run_id`` and directory
under the configured ``run_dir``. The directory holds the raw outputs (TSV,
PNG, PDF, FASTA) and a ``manifest.json`` receipt recording the tool name, the
exact parameters, software versions, the model file that was used (with its
sha256), timings and the sha256 of every output file. The agent cites
``run_id`` values in its report the way it cites DOIs, and the verifier
re-reads manifests to check that quoted numbers exist in the artifacts.

This is the pattern s2f-penguin calls "reproducibility receipts".
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

_SHA_CACHE: Dict[str, str] = {}


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    key = "%s:%s" % (path, path.stat().st_mtime_ns)
    if key in _SHA_CACHE:
        return _SHA_CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def _versions() -> Dict[str, str]:
    out: Dict[str, str] = {"python": platform.python_version()}
    for mod in ("peagent_tool", "numpy", "pandas", "tensorflow", "fastmcp"):
        try:
            m = __import__(mod)
            out[mod] = str(getattr(m, "__version__", "?"))
        except Exception:  # noqa: BLE001 - version reporting must never fail a run
            pass
    return out


@dataclass
class Run:
    tool: str
    params: Dict[str, Any]
    run_dir: Path
    run_id: str
    started: float = field(default_factory=time.time)
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    model: Dict[str, Any] = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return self.run_dir / self.run_id

    def file(self, name: str) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        return self.path / name

    def register_output(self, path: Path, kind: str, description: str = "") -> None:
        path = Path(path)
        entry = {"path": str(path), "kind": kind, "description": description}
        if path.exists():
            entry["sha256"] = sha256_of(path)
            entry["bytes"] = path.stat().st_size
        self.outputs.append(entry)

    def set_model(self, species: str, model_path: Optional[str], metadata_path: Optional[str]) -> None:
        info: Dict[str, Any] = {"species": species, "model_path": model_path, "metadata_path": metadata_path}
        for key in ("model_path", "metadata_path"):
            p = info.get(key)
            if p and Path(p).exists():
                info[key + "_sha256"] = sha256_of(Path(p))
        self.model = info

    def finish(self, summary: Dict[str, Any], status: str = "ok", error: Optional[str] = None) -> Dict[str, Any]:
        manifest = {
            "run_id": self.run_id,
            "tool": self.tool,
            "status": status,
            "error": error,
            "params": self.params,
            "model": self.model,
            "outputs": self.outputs,
            "notes": self.notes,
            "summary": summary,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "duration_s": round(time.time() - self.started, 3),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "versions": _versions(),
        }
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2, default=str)
        return manifest


def new_run(run_dir: Path, tool: str, params: Dict[str, Any]) -> Run:
    run_id = "%s_%s_%s" % (time.strftime("%Y%m%dT%H%M%S"), tool.replace("peagent_", ""), uuid.uuid4().hex[:6])
    return Run(tool=tool, params=params, run_dir=run_dir, run_id=run_id)


def load_manifest(run_dir: Path, run_id: str) -> Dict[str, Any]:
    p = (run_dir / run_id / "manifest.json").resolve()
    if run_dir.resolve() not in p.parents:
        raise ValueError("run_id escapes the run directory")
    if not p.exists():
        raise FileNotFoundError("No manifest for run_id %s" % run_id)
    with open(p) as fh:
        return json.load(fh)

"""Genome and annotation access: region fetch, gene lookup, window helpers.

Only ``pysam`` (FASTA) and a small GFF3 reader are used, so this module works
without TensorFlow. Coordinates in and out are **1-based, inclusive**, which is
the convention of GFF3, VCF and the PEAgent browser; the only 0-based numbers
are internal.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

INPUT_LEN = 1344  # PEAgent model input

_REGION_RE = re.compile(r"^\s*([^:\s]+)\s*:\s*([\d,]+)\s*[-–]\s*([\d,]+)\s*$")


def parse_region(region: str) -> Tuple[str, int, int]:
    """Parse ``chr1:1,000-2,344`` into ``("chr1", 1000, 2344)`` (1-based inclusive)."""
    m = _REGION_RE.match(region)
    if not m:
        raise ValueError("Region must look like chrom:start-end, got %r" % region)
    chrom, start, end = m.group(1), int(m.group(2).replace(",", "")), int(m.group(3).replace(",", ""))
    if end < start:
        start, end = end, start
    return chrom, start, end


def centred_window(start: int, end: int, length: int = INPUT_LEN) -> Tuple[int, int]:
    """Return the ``length``-bp window (1-based inclusive) centred on the midpoint of start–end."""
    centre = (start + end) // 2
    w_start = centre - length // 2 + 1
    return w_start, w_start + length - 1


def fetch_sequence(fasta_path: Path, chrom: str, start: int, end: int) -> str:
    """Fetch an uppercase sequence for a 1-based inclusive interval."""
    import pysam  # optional dependency; only needed when a genome is configured

    with pysam.FastaFile(str(fasta_path)) as fa:
        contigs = set(fa.references)
        name = chrom
        if name not in contigs:
            # tolerate chr/Chr/no-prefix mismatches between user input and FASTA
            alts = [chrom.replace("chr", ""), "chr" + chrom.replace("chr", ""), chrom.capitalize(), chrom.lower()]
            name = next((a for a in alts if a in contigs), None)
            if name is None:
                raise KeyError("Contig %r not in %s (first contigs: %s)" % (chrom, fasta_path.name, sorted(contigs)[:5]))
        length = fa.get_reference_length(name)
        s0 = max(0, start - 1)
        e0 = min(length, end)
        seq = fa.fetch(name, s0, e0).upper()
        # pad with N if the window ran off a chromosome end so the caller still gets the requested length
        left_pad = "N" * (s0 - (start - 1)) if start - 1 < 0 else ""
        right_pad = "N" * (end - e0) if end > length else ""
        return left_pad + seq + right_pad


@dataclass
class GeneRecord:
    gene_id: str
    name: Optional[str]
    chrom: str
    start: int
    end: int
    strand: str
    biotype: Optional[str] = None
    description: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        tss = self.start if self.strand == "+" else self.end
        return {
            "gene_id": self.gene_id,
            "name": self.name,
            "chrom": self.chrom,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "length_bp": self.end - self.start + 1,
            "tss": tss,
            "biotype": self.biotype,
            "description": self.description,
        }


def _open_text(path: Path):
    return gzip.open(path, "rt") if str(path).endswith(".gz") else open(path)


@lru_cache(maxsize=4)
def load_genes(gff_path: str) -> Tuple[Dict[str, GeneRecord], Dict[str, List[str]]]:
    """Parse gene features from a GFF3 file once. Returns (by_id, by_lower_name)."""
    by_id: Dict[str, GeneRecord] = {}
    by_name: Dict[str, List[str]] = {}
    with _open_text(Path(gff_path)) as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] not in ("gene", "pseudogene", "transposable_element_gene"):
                continue
            attrs = dict(kv.split("=", 1) for kv in parts[8].split(";") if "=" in kv)
            gid = attrs.get("ID", attrs.get("gene_id", ""))
            gid = gid.split(":", 1)[-1] if gid.startswith("gene:") else gid
            if not gid:
                continue
            name = attrs.get("Name") or attrs.get("gene_name") or attrs.get("symbol")
            rec = GeneRecord(
                gene_id=gid,
                name=name,
                chrom=parts[0],
                start=int(parts[3]),
                end=int(parts[4]),
                strand=parts[6],
                biotype=attrs.get("biotype") or attrs.get("gene_biotype") or parts[2],
                description=attrs.get("description") or attrs.get("Note"),
            )
            by_id[gid] = rec
            for key in filter(None, {gid, name, attrs.get("Alias"), attrs.get("locus_tag")}):
                for alias in str(key).split(","):
                    by_name.setdefault(alias.strip().lower(), []).append(gid)
    return by_id, by_name


def find_gene(gff_path: Path, query: str, max_hits: int = 10) -> List[GeneRecord]:
    """Exact match on ID/Name/Alias first (case-insensitive), then substring match on IDs and names."""
    by_id, by_name = load_genes(str(gff_path))
    q = query.strip().lower()
    hits = [by_id[g] for g in by_name.get(q, [])]
    if not hits:
        for gid, rec in by_id.items():
            if q in gid.lower() or (rec.name and q in rec.name.lower()):
                hits.append(rec)
                if len(hits) >= max_hits:
                    break
    return hits[:max_hits]


def reverse_complement(seq: str) -> str:
    comp = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(comp)[::-1]

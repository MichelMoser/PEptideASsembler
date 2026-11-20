"""Core data structures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Read:
    """One de novo peptide prediction ("read") for one spectrum.

    ``conf`` holds a per-residue confidence in [0, 1] (Casanovo aa score,
    ``exp(token log prob)`` for InstaNovo).  ``weight`` is the share of the
    spectrum's evidence given to this candidate: a spectrum with several
    candidate peptides (beams / top-n PSMs) spreads one unit of evidence
    over them in proportion to their peptide-level probability.
    """

    id: str
    source: str
    spectrum: str
    seq: str
    conf: list[float]
    score: float
    weight: float = 1.0
    rank: int = 0
    flags: str = ""
    # spectrum identity used to match the same spectrum across tools
    run: str = ""                 # raw file / run name
    index: int | None = None      # 0-based spectrum index in the run, if known
    scan: int | None = None       # scan number, if known
    alt_id: int | None = None     # any other integer id (e.g. trailing number of spectrum_id)
    ppm: float | None = None      # precursor m/z error of this candidate (computed)

    def __post_init__(self) -> None:
        if len(self.seq) != len(self.conf):
            raise ValueError(
                f"read {self.id}: {len(self.seq)} residues but {len(self.conf)} confidences"
            )


@dataclass(slots=True)
class Unit:
    """A distinct sequence with pooled evidence (collapsed identical reads).

    ``conf[i]`` is the weight-averaged confidence of position ``i`` and
    ``weight`` the summed read weight, so ``conf[i] * weight`` is the
    evidence for residue ``seq[i]``.
    """

    seq: str
    conf: list[float]
    weight: float
    members: list[str] = field(default_factory=list)

    def evidence(self) -> list[float]:
        w = self.weight
        return [c * w for c in self.conf]


@dataclass(slots=True)
class Contig:
    seq: str
    conf: list[float]
    depth: list[float]
    alt: list[str] = field(default_factory=list)
    members: set[str] = field(default_factory=set)
    origin: str = ""
    id: str = ""
    passed: bool = True

    @property
    def mean_conf(self) -> float:
        return sum(self.conf) / len(self.conf) if self.conf else 0.0

    @property
    def mean_depth(self) -> float:
        return sum(self.depth) / len(self.depth) if self.depth else 0.0

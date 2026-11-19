"""Read preprocessing: candidate weighting, quality trimming, collapsing.

Analogous to the read-QC stage of short-read assemblers: low-confidence
ends are trimmed (like Phred quality trimming), short or unreliable reads
are dropped and identical sequences are collapsed into :class:`Unit`
objects whose per-position evidence is the confidence-weighted sum over
all supporting spectra.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import Read, Unit


@dataclass
class PreprocessParams:
    min_score: float = 0.0          # minimum peptide-level score/probability
    trim_conf: float = 0.3          # trim terminal residues below this confidence
    min_len: int = 6                # minimum length after trimming
    min_mean_conf: float = 0.4      # drop reads whose mean confidence is lower
    max_rank: int = 4               # keep beams/top-n candidates with rank <= this
    min_candidate_weight: float = 0.05
    source_weights: dict[str, float] | None = None


def weight_candidates(reads: list[Read]) -> None:
    """Spread one unit of evidence per (source, spectrum) over its candidates.

    A spectrum with several candidate peptides (InstaNovo beams, Casanovo
    ``top_match > 1``) contributes their peptide probabilities normalised to
    sum to one, so ambiguous spectra contribute proportionally less to each
    interpretation while a single confident prediction keeps weight 1.
    """
    groups: dict[tuple[str, str], list[Read]] = defaultdict(list)
    for r in reads:
        groups[(r.source, r.spectrum)].append(r)
    for group in groups.values():
        if len(group) == 1:
            continue
        total = sum(max(r.score, 1e-9) for r in group)
        for r in group:
            r.weight *= max(r.score, 1e-9) / total


def trim(read: Read, min_conf: float) -> tuple[int, int]:
    """Return [start, end) after trimming low-confidence ends."""
    start, end = 0, len(read.conf)
    while start < end and read.conf[start] < min_conf:
        start += 1
    while end > start and read.conf[end - 1] < min_conf:
        end -= 1
    return start, end


def preprocess(reads: list[Read], p: PreprocessParams) -> tuple[list[Read], dict[str, int]]:
    """Filter and trim reads.  Returns kept reads and counters."""
    stats = defaultdict(int)
    stats["input"] = len(reads)
    sw = p.source_weights or {}
    reads = [r for r in reads if r.rank <= p.max_rank]
    weight_candidates(reads)
    out: list[Read] = []
    for r in reads:
        r.weight *= sw.get(r.source, 1.0)
        if r.weight < p.min_candidate_weight:
            stats["low_candidate_weight"] += 1
            continue
        if r.score < p.min_score:
            stats["low_score"] += 1
            continue
        s, e = trim(r, p.trim_conf)
        if e - s < p.min_len:
            stats["too_short"] += 1
            continue
        if (s, e) != (0, len(r.seq)):
            stats["trimmed"] += 1
            r.seq, r.conf = r.seq[s:e], r.conf[s:e]
        if sum(r.conf) / len(r.conf) < p.min_mean_conf:
            stats["low_mean_conf"] += 1
            continue
        out.append(r)
    stats["kept"] = len(out)
    return out, dict(stats)


def collapse(reads: list[Read]) -> list[Unit]:
    """Merge reads with identical sequence into evidence-weighted units."""
    acc: dict[str, list] = {}
    for r in reads:
        a = acc.get(r.seq)
        if a is None:
            a = acc[r.seq] = [[0.0] * len(r.seq), 0.0, []]
        ev = a[0]
        for i, c in enumerate(r.conf):
            ev[i] += c * r.weight
        a[1] += r.weight
        a[2].append(r.id)
    units = [
        Unit(seq=s, conf=[e / w for e in ev], weight=w, members=m)
        for s, (ev, w, m) in acc.items()
        if w > 0
    ]
    units.sort(key=lambda u: (-len(u.seq), -u.weight, u.seq))
    return units

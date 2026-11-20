"""Cross-tool agreement: use Casanovo and InstaNovo predictions of the same
spectrum as (partly) independent evidence.

1. **Pairing.**  Spectrum identifiers differ between tools (Casanovo uses a
   0-based ``index=`` or ``scan=``, InstaNovo ``scan_number``,
   ``spectrum_index`` or ``spectrum_id``) and runs may be named differently.
   peas therefore tries every run pairing x identifier type x offset (-1, 0,
   +1) and keeps the combination with the most identical top predictions.
   A wrong pairing gives ~0 identical predictions, so the signal is clear;
   pairings below ``min_identical`` / ``min_identical_rate`` are rejected.
2. **Residue agreement.**  For a matched spectrum every candidate of one tool
   is aligned (difflib matching blocks, so N<->GG style length differences
   are tolerated) to the *best* candidate of the other tool:

   * agreeing residue:  ``c <- 1 - (1 - c) * (1 - s * c_other)``
   * conflicting residue: ``c <- c * (1 - s * c_other_mismatch)``

   with ``s`` = ``strength`` (0 disables; < 1 because the two models' errors
   are correlated - hard spectra are hard for both).
3. **Shared spectrum id.**  Both tools' candidates get the same spectrum id,
   so a spectrum is counted once in the number of supporting spectra.  Each
   prediction keeps its weight by default (``spectrum_share`` = 1; 0.5 makes
   a paired spectrum count as one unit of depth instead of two).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher

from .model import Read

KEY_TYPES = ("index", "scan", "alt_id")


@dataclass
class AgreementParams:
    strength: float = 0.5
    min_identical: int = 10
    min_identical_rate: float = 0.05
    offsets: tuple[int, ...] = (-1, 0, 1)
    spectrum_share: float = 1.0   # weight multiplier for paired predictions


@dataclass
class Pairing:
    run_a: str
    run_b: str
    key_a: str
    key_b: str
    offset: int          # key_a == key_b + offset
    matched: int
    identical: int

    @property
    def rate(self) -> float:
        return self.identical / self.matched if self.matched else 0.0


def _top_by_run(reads: list[Read]) -> dict[str, list[Read]]:
    out: dict[str, list[Read]] = defaultdict(list)
    for r in reads:
        if r.rank == 0:
            out[r.run].append(r)
    return out


def detect_pairings(reads_a: list[Read], reads_b: list[Read], p: AgreementParams) -> list[Pairing]:
    top_a, top_b = _top_by_run(reads_a), _top_by_run(reads_b)
    cands: list[Pairing] = []
    for run_b, rb in top_b.items():
        for kb in KEY_TYPES:
            base = {getattr(r, kb): r.seq for r in rb if getattr(r, kb) is not None}
            if not base:
                continue
            for run_a, ra in top_a.items():
                for ka in KEY_TYPES:
                    keys_a = [(getattr(r, ka), r.seq) for r in ra if getattr(r, ka) is not None]
                    if not keys_a:
                        continue
                    for off in p.offsets:
                        matched = identical = 0
                        for k, seq in keys_a:
                            other = base.get(k - off)
                            if other is not None:
                                matched += 1
                                identical += other == seq
                        if identical:
                            cands.append(Pairing(run_a, run_b, ka, kb, off, matched, identical))
    cands.sort(key=lambda x: (-x.identical, -x.rate))
    used_a, used_b, chosen = set(), set(), []
    for c in cands:
        if c.run_a in used_a or c.run_b in used_b:
            continue
        if c.identical < p.min_identical or c.rate < p.min_identical_rate:
            continue
        used_a.add(c.run_a)
        used_b.add(c.run_b)
        chosen.append(c)
    return chosen


def _update(target: Read, other: Read, s: float) -> tuple[int, int]:
    """Adjust target confidences against the other tool's best candidate."""
    sm = SequenceMatcher(None, target.seq, other.seq, autojunk=False)
    agree: dict[int, int] = {}
    for a, b, n in sm.get_matching_blocks():
        for i in range(n):
            agree[a + i] = b + i
    used_other = set(agree.values())
    unmatched_other = [c for j, c in enumerate(other.conf) if j not in used_other]
    c_mis = sum(unmatched_other) / len(unmatched_other) if unmatched_other else 0.0
    n_agree = n_conflict = 0
    conf = target.conf
    for i in range(len(conf)):
        j = agree.get(i)
        if j is not None:
            conf[i] = 1.0 - (1.0 - conf[i]) * (1.0 - s * other.conf[j])
            n_agree += 1
        elif c_mis > 0:
            conf[i] = conf[i] * (1.0 - s * c_mis)
            n_conflict += 1
    return n_agree, n_conflict


def apply_agreement(reads: list[Read], p: AgreementParams, a: str = "casanovo", b: str = "instanovo") -> dict:
    """Pair spectra across tools and update confidences/weights in place."""
    reads_a = [r for r in reads if r.source == a]
    reads_b = [r for r in reads if r.source == b]
    stats: dict = {"pairings": []}
    if not reads_a or not reads_b:
        return stats
    pairings = detect_pairings(reads_a, reads_b, p)
    by_a: dict[tuple[str, int], list[Read]] = defaultdict(list)
    by_b: dict[tuple[str, int], list[Read]] = defaultdict(list)
    n_spec = n_ident = n_agree = n_conf = 0
    for pr in pairings:
        by_a.clear()
        by_b.clear()
        for r in reads_a:
            k = getattr(r, pr.key_a)
            if r.run == pr.run_a and k is not None:
                by_a[(r.run, k)].append(r)
        for r in reads_b:
            k = getattr(r, pr.key_b)
            if r.run == pr.run_b and k is not None:
                by_b[(r.run, k + pr.offset)].append(r)
        for (run, k), ca in by_a.items():
            cb = by_b.get((pr.run_b, k))
            if not cb:
                continue
            best_a = min(ca, key=lambda r: r.rank)
            best_b = min(cb, key=lambda r: r.rank)
            n_spec += 1
            n_ident += best_a.seq == best_b.seq
            # snapshot the partners before modifying anything
            ref_a = Read(best_a.id, a, "", best_a.seq, list(best_a.conf), best_a.score)
            ref_b = Read(best_b.id, b, "", best_b.seq, list(best_b.conf), best_b.score)
            shared = f"{pr.run_a}|{pr.key_a}={k}"
            for r in ca:
                x, y = _update(r, ref_b, p.strength)
                n_agree, n_conf = n_agree + x, n_conf + y
            for r in cb:
                x, y = _update(r, ref_a, p.strength)
                n_agree, n_conf = n_agree + x, n_conf + y
            for r in ca + cb:
                r.spectrum = shared
                r.weight *= p.spectrum_share
                r.flags = ",".join(f for f in (r.flags, "paired") if f)
    stats["pairings"] = [
        f"{x.run_a}[{x.key_a}] <-> {x.run_b}[{x.key_b}{x.offset:+d}]: "
        f"{x.matched} matched, {x.identical} identical ({x.rate:.0%})"
        for x in pairings
    ]
    stats["paired_spectra"] = n_spec
    stats["identical_top_predictions"] = n_ident
    stats["residues_agree"] = n_agree
    stats["residues_conflict"] = n_conf
    return stats

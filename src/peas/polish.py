"""Map units back to contigs, polish, extend and score contigs.

This is the equivalent of read mapping + consensus polishing (Pilon/Racon)
in genome assembly.  Every unit is placed ungapped on its best contig
diagonal (found via a k-mer index, ties share the weight).  Each contig
column accumulates confidence-weighted votes, which gives

* the polished residue (weighted majority),
* a per-position confidence: mean over covering spectra of
  (residue confidence if the spectrum agrees with the call, else 0),
* the depth (summed spectrum weight) and the best alternative residue.

Reads hanging over a contig end extend it when the overhang column is
supported by enough evidence; unsupported end columns are trimmed.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .model import Contig, Unit


@dataclass
class PolishParams:
    k: int = 4
    min_overlap: int = 5
    min_identity: float = 0.8
    rounds: int = 4
    max_diagonals: int = 6
    extend_min_evidence: float = 1.0   # summed conf*weight of the winning residue
    extend_min_agreement: float = 0.6  # winning share of the column evidence
    trim_min_depth: float = 1e-9


@dataclass(slots=True)
class Placement:
    unit: int
    contig: int
    offset: int       # unit[0] sits at contig[offset]
    identity: float
    share: float


@dataclass
class Profile:
    cols: dict[int, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    depth: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    members: set[int] = field(default_factory=set)


def place_units(contigs: list[str], units: list[Unit], p: PolishParams) -> list[Placement]:
    k = p.k
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ci, s in enumerate(contigs):
        for pos in range(len(s) - k + 1):
            index[s[pos : pos + k]].append((ci, pos))

    placements: list[Placement] = []
    for ui, u in enumerate(units):
        s, conf = u.seq, u.conf
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for q in range(len(s) - k + 1):
            for ci, pos in index.get(s[q : q + k], ()):
                counts[(ci, pos - q)] += 1
        if not counts:
            continue
        cands = sorted(counts.items(), key=lambda x: -x[1])[: p.max_diagonals]
        scored: list[tuple[float, int, int, float]] = []
        for (ci, d), _ in cands:
            c = contigs[ci]
            start, end = max(0, d), min(len(c), d + len(s))
            length = end - start
            if length < min(p.min_overlap, len(s)):
                continue
            matches, score = 0, 0.0
            for i in range(start, end):
                if c[i] == s[i - d]:
                    matches += 1
                    score += conf[i - d]
                else:
                    score -= conf[i - d]
            ident = matches / length
            if ident >= p.min_identity:
                scored.append((score, ci, d, ident))
        if not scored:
            continue
        best = max(x[0] for x in scored)
        top = [x for x in scored if x[0] >= best - 1e-9]
        for score, ci, d, ident in top:
            placements.append(Placement(ui, ci, d, ident, 1.0 / len(top)))
    return placements


def build_profiles(n_contigs: int, units: list[Unit], placements: list[Placement]) -> list[Profile]:
    profiles = [Profile() for _ in range(n_contigs)]
    for pl in placements:
        u = units[pl.unit]
        prof = profiles[pl.contig]
        prof.members.add(pl.unit)
        w = u.weight * pl.share
        for j, (ch, c) in enumerate(zip(u.seq, u.conf)):
            pos = pl.offset + j
            prof.cols[pos][ch] += c * w
            prof.depth[pos] += w
    return profiles


def _call(prof: Profile, pos: int, fallback: str | None):
    col = prof.cols.get(pos)
    dp = prof.depth.get(pos, 0.0)
    if not col:
        return fallback, 0.0, dp, ""
    ranked = sorted(col.items(), key=lambda x: (-x[1], x[0]))
    ch, ev = ranked[0]
    alt = ""
    if len(ranked) > 1:
        a_ch, a_ev = ranked[1]
        alt = f"{a_ch}:{a_ev / dp:.2f}" if dp else ""
    return ch, (ev / dp if dp else 0.0), dp, alt


def polish_contig(seq: str, prof: Profile, p: PolishParams) -> str:
    """One polishing round: majority calls, end extension, end trimming."""
    out = [(_call(prof, i, seq[i])[0] or seq[i]) for i in range(len(seq))]

    def extendable(pos: int) -> str | None:
        col = prof.cols.get(pos)
        if not col:
            return None
        ch, ev = max(col.items(), key=lambda x: (x[1], x[0]))
        tot = sum(col.values())
        if ev >= p.extend_min_evidence and ev / tot >= p.extend_min_agreement:
            return ch
        return None

    left: list[str] = []
    pos = -1
    while (ch := extendable(pos)) is not None:
        left.append(ch)
        pos -= 1
    right: list[str] = []
    pos = len(seq)
    while (ch := extendable(pos)) is not None:
        right.append(ch)
        pos += 1
    new = "".join(left[::-1]) + "".join(out) + "".join(right)

    # trim unsupported ends (only inside the original span)
    lo_trim = 0
    while lo_trim < len(seq) and not left and prof.depth.get(lo_trim, 0.0) <= p.trim_min_depth:
        lo_trim += 1
    hi_trim = 0
    while hi_trim < len(seq) - lo_trim and not right and prof.depth.get(len(seq) - 1 - hi_trim, 0.0) <= p.trim_min_depth:
        hi_trim += 1
    if lo_trim or hi_trim:
        new = new[lo_trim : len(new) - hi_trim]
    return new


def polish(contigs: list[str], units: list[Unit], p: PolishParams) -> list[str]:
    for _ in range(p.rounds):
        placements = place_units(contigs, units, p)
        profiles = build_profiles(len(contigs), units, placements)
        new = [polish_contig(s, prof, p) for s, prof in zip(contigs, profiles)]
        new = [s for s in new if s]
        if new == contigs:
            break
        contigs = new
    return contigs


def score_contigs(contigs: list[str], units: list[Unit], p: PolishParams):
    """Final placement: per-position stats and member lists."""
    placements = place_units(contigs, units, p)
    profiles = build_profiles(len(contigs), units, placements)
    result: list[Contig] = []
    for s, prof in zip(contigs, profiles):
        conf, depth, alt = [], [], []
        for i in range(len(s)):
            dp = prof.depth.get(i, 0.0)
            col = prof.cols.get(i, {})
            # confidence of the residue actually in the contig
            conf.append(col.get(s[i], 0.0) / dp if dp else 0.0)
            depth.append(dp)
            others = [(ev, ch) for ch, ev in col.items() if ch != s[i]]
            if others and dp:
                ev, ch = max(others)
                alt.append(f"{ch}:{ev / dp:.2f}")
            else:
                alt.append("")
        members = {rid for ui in prof.members for rid in units[ui].members}
        result.append(Contig(seq=s, conf=conf, depth=depth, alt=alt, members=members))
    return result, placements


def banded_identity(t: str, s: str, d0: int, band: int) -> tuple[float, list[int]] | None:
    """Fraction of s (within t's span on diagonal d0) matching t on any
    diagonal d0-band..d0+band, and the unmatched positions of s.
    Tolerates small indels without full dynamic programming."""
    start, end = max(0, -d0), min(len(s), len(t) - d0)
    if end <= start:
        return None
    unmatched: list[int] = []
    for j in range(start, end):
        c = s[j]
        for d in range(d0 - band, d0 + band + 1):
            i = j + d
            if 0 <= i < len(t) and t[i] == c:
                break
        else:
            unmatched.append(j)
    return 1.0 - len(unmatched) / (end - start), unmatched


def remove_redundant(
    contigs: list[Contig],
    k: int,
    min_identity: float,
    min_cover: float,
    band: int = 2,
    variant_conf: float = 0.7,
    variant_depth: float = 2.0,
    min_variants: int = 2,
) -> tuple[list[Contig], int]:
    """Purge contigs that lie (ungapped, >= min_identity) inside a better contig.

    Contigs are visited best-first (length x mean confidence).  A contig is
    dropped when at least ``min_cover`` of its length aligns to an already
    kept contig - the analogue of haplotig purging / redundancy removal.
    Its reads are re-placed on the surviving contigs in the next polish.

    Exception (homologues, isoforms, antibody chains): a similar contig is
    kept when it differs at >= ``min_variants`` positions where its own call
    is well supported (confidence >= ``variant_conf`` and depth >=
    ``variant_depth``).  Noise-derived copies differ at weakly supported
    positions only.
    """
    order = sorted(contigs, key=lambda c: (-len(c.seq) * c.mean_conf, c.seq))
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    kept: list[Contig] = []
    dropped = 0
    for c in order:
        s = c.seq
        diags: dict[tuple[int, int], int] = defaultdict(int)
        for q in range(len(s) - k + 1):
            for ci, pos in index.get(s[q : q + k], ()):
                diags[(ci, pos - q)] += 1
        redundant = False
        tried: set[int] = set()
        for (ci, d0), _ in sorted(diags.items(), key=lambda x: -x[1]):
            if ci in tried:
                continue
            tried.add(ci)
            t = kept[ci].seq
            res = banded_identity(t, s, d0, band)
            covered = min(len(t), d0 + len(s)) - max(0, d0)
            if res is None or res[0] < min_identity or covered < min_cover * len(s):
                continue
            supported = sum(
                1 for j in res[1] if c.conf[j] >= variant_conf and c.depth[j] >= variant_depth
            )
            if supported < min_variants:
                redundant = True
                break
            if len(tried) >= 8:
                break
        if redundant:
            dropped += 1
            continue
        ci = len(kept)
        kept.append(c)
        for pos in range(len(s) - k + 1):
            index[s[pos : pos + k]].append((ci, pos))
    return kept, dropped

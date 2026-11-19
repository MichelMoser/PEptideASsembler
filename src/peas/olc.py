"""Overlap-layout-consensus assembly with k-mer seed hashing.

1. **Overlap**: every unit's k-mers are hashed into an index (very frequent
   k-mers are masked, like repeat masking).  Units sharing seeds on the same
   diagonal are compared ungapped over their overlap.  Mismatches are
   tolerated where at least one side is uncertain; a mismatch between two
   confident residues vetoes the overlap.
2. **Containment**: units contained in a longer unit are absorbed (peptides
   from different proteases / missed cleavages are highly nested).
3. **Layout**: greedy best-overlap-first chaining (as in greedy assemblers
   such as TIGR/SSAKE): an overlap is accepted when the left unit has no
   successor yet, the right unit no predecessor, and no cycle would form.
4. **Consensus**: confidence-weighted vote per layout column.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import Unit

# residue pairs that are hard to tell apart (near-isobaric): never count as
# confident mismatches.  I/L are merged by the alphabet already.
AMBIGUOUS = {frozenset("KQ"), frozenset("IL")}


@dataclass
class OLCParams:
    k: int = 4
    min_overlap: int = 5
    min_identity: float = 0.85
    confident_mismatch: float = 0.9   # min(conf_a, conf_b) above this = veto
    mismatch_penalty: float = 2.0
    max_kmer_occ: int = 500
    min_contig_len: int = 8


@dataclass(slots=True)
class Overlap:
    a: int          # left / containing unit
    b: int          # right / contained unit
    offset: int     # b[0] aligns to a[offset]
    length: int
    identity: float
    score: float
    contained: bool


def compare(sa: str, ca: list[float], sb: str, cb: list[float], d: int, p: OLCParams):
    """Ungapped comparison with b[0] placed at a[d].  None if rejected."""
    start = max(0, d)
    end = min(len(sa), d + len(sb))
    length = end - start
    if length < p.min_overlap:
        return None
    matches = 0
    score = 0.0
    for i in range(start, end):
        x, y = sa[i], sb[i - d]
        c = min(ca[i], cb[i - d])
        if x == y:
            matches += 1
            score += c
        else:
            if c >= p.confident_mismatch and frozenset((x, y)) not in AMBIGUOUS:
                return None
            score -= p.mismatch_penalty * c
    identity = matches / length
    if identity < p.min_identity:
        return None
    return length, identity, score


def find_overlaps(units: list[Unit], p: OLCParams) -> list[Overlap]:
    k = p.k
    index: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for ui, u in enumerate(units):
        s = u.seq
        for pos in range(len(s) - k + 1):
            index[s[pos : pos + k]].append((ui, pos))
    for km in [km for km, occ in index.items() if len(occ) > p.max_kmer_occ]:
        del index[km]

    overlaps: list[Overlap] = []
    for ai, ua in enumerate(units):
        sa = ua.seq
        diags: set[tuple[int, int]] = set()
        for pos in range(len(sa) - k + 1):
            for bi, q in index.get(sa[pos : pos + k], ()):
                if bi > ai:
                    diags.add((bi, pos - q))
        best: dict[int, Overlap] = {}
        for bi, d in diags:
            ub = units[bi]
            res = compare(sa, ua.conf, ub.seq, ub.conf, d, p)
            if res is None:
                continue
            length, ident, score = res
            la, lb = len(sa), len(ub.seq)
            if d >= 0 and d + lb <= la:
                ov = Overlap(ai, bi, d, length, ident, score, True)
            elif d <= 0 and d + lb >= la:
                ov = Overlap(bi, ai, -d, length, ident, score, True)
            elif d > 0:
                ov = Overlap(ai, bi, d, length, ident, score, False)
            else:
                ov = Overlap(bi, ai, -d, length, ident, score, False)
            cur = best.get(bi)
            if cur is None or ov.score > cur.score:
                best[bi] = ov
        overlaps.extend(best.values())
    return overlaps


def _order_key(units: list[Unit], i: int):
    u = units[i]
    return (-len(u.seq), -u.weight, i)


def layout(units: list[Unit], overlaps: list[Overlap]):
    """Resolve containment, chain dovetails, return list of member layouts.

    Each layout is a list of (unit index, offset) in contig coordinates.
    """
    n = len(units)
    # containment: parent must precede child in (length, weight) order -> acyclic
    parent: dict[int, tuple[int, int, float]] = {}
    for ov in overlaps:
        if not ov.contained:
            continue
        a, b = ov.a, ov.b
        if _order_key(units, a) > _order_key(units, b):
            if len(units[a].seq) != len(units[b].seq):
                continue
            a, b = b, a  # equal length, d == 0
        cur = parent.get(b)
        if cur is None or ov.score > cur[2]:
            parent[b] = (a, ov.offset if a == ov.a else -ov.offset, ov.score)

    def root(i: int) -> tuple[int, int]:
        off = 0
        seen = 0
        while i in parent and seen <= n:
            pa, d, _ = parent[i]
            off += d
            i = pa
            seen += 1
        return i, off

    free = [i for i in range(n) if i not in parent]
    free_set = set(free)

    # greedy dovetail chaining among non-contained units
    succ: dict[int, tuple[int, int]] = {}
    pred: dict[int, int] = {}
    uf = list(range(n))

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    dovetails = [o for o in overlaps if not o.contained and o.a in free_set and o.b in free_set]
    dovetails.sort(key=lambda o: (-o.score, -o.length, o.a, o.b))
    for o in dovetails:
        if o.a in succ or o.b in pred:
            continue
        ra, rb = find(o.a), find(o.b)
        if ra == rb:
            continue
        succ[o.a] = (o.b, o.offset)
        pred[o.b] = o.a
        uf[ra] = rb

    # place contained units under their root
    children: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i in parent:
        r, off = root(i)
        children[r].append((i, off))

    layouts: list[list[tuple[int, int]]] = []
    for s in free:
        if s in pred:
            continue
        members: list[tuple[int, int]] = []
        cur, off = s, 0
        visited = set()
        while True:
            visited.add(cur)
            members.append((cur, off))
            members.extend((c, off + d) for c, d in children.get(cur, ()))
            if cur not in succ:
                break
            nxt, d = succ[cur]
            if nxt in visited:
                break
            cur, off = nxt, off + d
        layouts.append(members)
    return layouts


def consensus(units: list[Unit], members: list[tuple[int, int]]):
    """Confidence-weighted column vote.  Returns (seq, conf, depth, member ids)."""
    lo = min(off for _, off in members)
    hi = max(off + len(units[i].seq) for i, off in members)
    cols: list[dict[str, float]] = [defaultdict(float) for _ in range(hi - lo)]
    depth = [0.0] * (hi - lo)
    ids: list[str] = []
    for i, off in members:
        u = units[i]
        ids.extend(u.members)
        for j, (ch, c) in enumerate(zip(u.seq, u.conf)):
            col = off - lo + j
            cols[col][ch] += c * u.weight
            depth[col] += u.weight
    seq, conf = [], []
    for col, dp in zip(cols, depth):
        if not col:
            seq.append("X")
            conf.append(0.0)
            continue
        ch, ev = max(col.items(), key=lambda x: (x[1], x[0]))
        seq.append(ch)
        # mean over covering units of (confidence if it agrees with the call else 0)
        conf.append(ev / dp if dp else 0.0)
    return "".join(seq), conf, depth, ids


def assemble_olc(units: list[Unit], p: OLCParams):
    overlaps = find_overlaps(units, p)
    layouts = layout(units, overlaps)
    contigs = []
    for members in layouts:
        seq, conf, depth, ids = consensus(units, members)
        if len(seq) >= p.min_contig_len:
            contigs.append((seq, conf, depth, ids))
    stats = {
        "units": len(units),
        "overlaps": len(overlaps),
        "containments": sum(o.contained for o in overlaps),
        "layouts": len(layouts),
        "contigs": len(contigs),
    }
    return contigs, stats

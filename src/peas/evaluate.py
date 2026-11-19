"""Evaluate contigs against a reference protein set (local alignment)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hit:
    contig: str
    protein: str
    c_start: int
    c_end: int
    r_start: int
    r_end: int
    aligned: int
    matches: int
    gaps: int
    contig_len: int

    @property
    def identity(self) -> float:
        return self.matches / self.aligned if self.aligned else 0.0


def smith_waterman(a: str, b: str, match: int = 3, mismatch: int = -2, gap: int = -4):
    """Local alignment of a (contig) vs b (reference).  Pure Python, O(nm)."""
    n, m = len(a), len(b)
    H = [[0] * (m + 1) for _ in range(n + 1)]
    best, bi, bj = 0, 0, 0
    for i in range(1, n + 1):
        ai = a[i - 1]
        Hi, Hp = H[i], H[i - 1]
        for j in range(1, m + 1):
            s = Hp[j - 1] + (match if ai == b[j - 1] else mismatch)
            u = Hp[j] + gap
            l = Hi[j - 1] + gap
            v = max(0, s, u, l)
            Hi[j] = v
            if v > best:
                best, bi, bj = v, i, j
    # traceback
    i, j = bi, bj
    matches = aligned = gaps = 0
    while i > 0 and j > 0 and H[i][j] > 0:
        if H[i][j] == H[i - 1][j - 1] + (match if a[i - 1] == b[j - 1] else mismatch):
            matches += a[i - 1] == b[j - 1]
            i, j = i - 1, j - 1
        elif H[i][j] == H[i - 1][j] + gap:
            i -= 1
            gaps += 1
        else:
            j -= 1
            gaps += 1
        aligned += 1
    return best, i, bi, j, bj, aligned, matches, gaps


def evaluate(contigs: list[tuple[str, str]], reference: list[tuple[str, str]], merge_il: bool = True) -> dict:
    norm = (lambda s: s.upper().replace("I", "L")) if merge_il else (lambda s: s.upper())
    refs = [(n, norm(s)) for n, s in reference]
    hits: list[Hit] = []
    covered: dict[str, set[int]] = {n: set() for n, _ in refs}
    for cname, cseq in contigs:
        cseq = norm(cseq)
        best = None
        for rname, rseq in refs:
            sc, cs, ce, rs, re_, al, mt, gp = smith_waterman(cseq, rseq)
            if best is None or sc > best[0]:
                best = (sc, Hit(cname, rname, cs, ce, rs, re_, al, mt, gp, len(cseq)))
        if best is not None:
            h = best[1]
            hits.append(h)
            if h.identity >= 0.8:
                covered[h.protein].update(range(h.r_start, h.r_end))
    total_ref = sum(len(s) for _, s in refs)
    lens = sorted((len(s) for _, s in contigs), reverse=True)
    half, acc, n50 = sum(lens) / 2, 0, 0
    for L in lens:
        acc += L
        if acc >= half:
            n50 = L
            break
    aligned = sum(h.aligned for h in hits)
    return {
        "hits": hits,
        "n_contigs": len(contigs),
        "n50": n50,
        "longest": lens[0] if lens else 0,
        "reference_coverage": sum(len(v) for v in covered.values()) / total_ref if total_ref else 0.0,
        "per_protein_coverage": {n: len(covered[n]) / len(s) for n, s in refs},
        "aligned_identity": sum(h.matches for h in hits) / aligned if aligned else 0.0,
        "contig_fraction_aligned": sum(h.c_end - h.c_start for h in hits) / sum(lens) if lens else 0.0,
    }

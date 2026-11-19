"""Confidence-weighted de Bruijn graph assembly.

Inspired by Velvet/SPAdes-style short-read assembly and by ALPS (Tran et
al. 2016), which assembles de novo peptides with PEAKS local confidence:

* **Nodes** are k-mers.  A k-mer occurrence contributes the *minimum*
  residue evidence (confidence x read weight) within it, so one uncertain
  residue caps the reliability of every k-mer covering it.
* **Edges** are (k+1)-mers actually observed in reads ("read-threaded"
  edges).  With the small k needed for 7-30 aa peptides, inferring edges
  from (k-1) overlaps alone would create many spurious connections.
* **Cleaning**: weak k-mers/edges are removed (the analogue of coverage
  cut-offs), branches much weaker than their siblings are cut, tips are
  clipped and simple bubbles (substitution errors, e.g. K/Q) are popped.
* **Contigs** are produced by greedy heaviest-path extension with a
  reciprocal-best check at every branch, starting from the heaviest k-mers.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .model import Unit


@dataclass
class DBGParams:
    k: int = 5
    min_kmer_weight: float = 0.5
    min_edge_weight: float = 0.5
    relative_edge_ratio: float = 0.15
    tip_length: int | None = None      # default: k
    bubble_length: int | None = None   # default: 2k
    min_contig_len: int = 8


class DBG:
    def __init__(self, k: int):
        self.k = k
        self.kmer_w: dict[str, float] = defaultdict(float)
        self.edge_w: dict[str, float] = defaultdict(float)
        self.nodes: dict[str, float] = {}
        self.succ: dict[str, dict[str, float]] = {}
        self.pred: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------ counting
    def add(self, seq: str, evidence: list[float]) -> None:
        k = self.k
        n = len(seq)
        if n < k:
            return
        for i in range(n - k + 1):
            self.kmer_w[seq[i : i + k]] += min(evidence[i : i + k])
        for i in range(n - k):
            self.edge_w[seq[i : i + k + 1]] += min(evidence[i : i + k + 1])

    def add_units(self, units: list[Unit]) -> None:
        for u in units:
            self.add(u.seq, u.evidence())

    # ------------------------------------------------------------ graph
    def build(self, min_kmer_weight: float, min_edge_weight: float) -> None:
        self.nodes = {km: w for km, w in self.kmer_w.items() if w >= min_kmer_weight}
        self.succ = {n: {} for n in self.nodes}
        self.pred = {n: {} for n in self.nodes}
        for e, w in self.edge_w.items():
            if w < min_edge_weight:
                continue
            u, v = e[:-1], e[1:]
            if u == v or u not in self.nodes or v not in self.nodes:
                continue
            self.succ[u][v] = w
            self.pred[v][u] = w

    def remove_edge(self, u: str, v: str) -> None:
        self.succ.get(u, {}).pop(v, None)
        self.pred.get(v, {}).pop(u, None)

    def remove_node(self, n: str) -> None:
        for v in list(self.succ.get(n, {})):
            self.remove_edge(n, v)
        for u in list(self.pred.get(n, {})):
            self.remove_edge(u, n)
        self.nodes.pop(n, None)
        self.succ.pop(n, None)
        self.pred.pop(n, None)

    def n_edges(self) -> int:
        return sum(len(s) for s in self.succ.values())

    # ------------------------------------------------------------ cleaning
    def filter_relative(self, ratio: float) -> int:
        """Cut edges much weaker than the strongest sibling edge."""
        doomed: set[tuple[str, str]] = set()
        for u, outs in self.succ.items():
            if len(outs) > 1:
                m = max(outs.values())
                doomed.update((u, v) for v, w in outs.items() if w < ratio * m)
        for v, ins in self.pred.items():
            if len(ins) > 1:
                m = max(ins.values())
                doomed.update((u, v) for u, w in ins.items() if w < ratio * m)
        for u, v in doomed:
            self.remove_edge(u, v)
        return len(doomed)

    def _tip(self, start: str, fwd: bool, max_len: int) -> list[str] | None:
        """Path from a dead end to (excluding) a junction, if short enough."""
        nxt_of = self.succ if fwd else self.pred
        prv_of = self.pred if fwd else self.succ
        path = [start]
        cur = start
        while True:
            outs = nxt_of[cur]
            if len(outs) != 1:
                return None  # dead end on both sides (isolated chain) or branching
            nxt = next(iter(outs))
            if len(prv_of[nxt]) > 1:
                # junction reached: tip if a competing branch is at least as strong
                tip_w = prv_of[nxt][cur]
                other = max(w for n, w in prv_of[nxt].items() if n != cur)
                return path if other >= tip_w else None
            path.append(nxt)
            cur = nxt
            if len(path) > max_len:
                return None

    def clip_tips(self, max_len: int) -> int:
        removed = 0
        changed = True
        while changed:
            changed = False
            for n in list(self.nodes):
                if n not in self.nodes:
                    continue
                for fwd, deg in ((True, self.pred), (False, self.succ)):
                    if n in self.nodes and not deg[n]:
                        tip = self._tip(n, fwd, max_len)
                        if tip:
                            for t in tip:
                                self.remove_node(t)
                            removed += len(tip)
                            changed = True
        return removed

    def pop_bubbles(self, max_len: int) -> int:
        """Pop simple bubbles: parallel unbranched paths u -> ... -> v."""
        popped = 0
        for u in list(self.nodes):
            if u not in self.nodes or len(self.succ[u]) < 2:
                continue
            ends: dict[str, list[tuple[float, list[str], str]]] = defaultdict(list)
            for v, w0 in list(self.succ[u].items()):
                path: list[str] = []
                cur, total, steps = v, w0, 1
                while (
                    len(self.pred[cur]) == 1
                    and len(self.succ[cur]) == 1
                    and len(path) < max_len
                ):
                    path.append(cur)
                    nxt = next(iter(self.succ[cur]))
                    total += self.succ[cur][nxt]
                    steps += 1
                    cur = nxt
                ends[cur].append((total / steps, path, v))
            for end, branches in ends.items():
                if len(branches) < 2 or end == u:
                    continue
                branches.sort(key=lambda b: -b[0])
                for _, path, first in branches[1:]:
                    if path:
                        for n in path:
                            self.remove_node(n)
                    else:
                        self.remove_edge(u, first)
                    popped += 1
        return popped

    def clean(self, p: DBGParams) -> dict[str, int]:
        k = self.k
        stats = {
            "relative_edges_cut": self.filter_relative(p.relative_edge_ratio),
            "tip_nodes_removed": self.clip_tips(p.tip_length or k),
            "bubbles_popped": self.pop_bubbles(p.bubble_length or 2 * k),
        }
        stats["tip_nodes_removed"] += self.clip_tips(p.tip_length or k)
        # drop isolated nodes with no edges (cannot form a contig >= k+1)
        for n in [n for n in self.nodes if not self.succ[n] and not self.pred[n]]:
            self.remove_node(n)
        stats["nodes"] = len(self.nodes)
        stats["edges"] = self.n_edges()
        return stats

    # ------------------------------------------------------------ traversal
    def unitigs(self) -> list[list[str]]:
        """Maximal non-branching paths (used for GFA output)."""
        def continues_back(n: str) -> bool:
            if len(self.pred[n]) != 1:
                return False
            p = next(iter(self.pred[n]))
            return len(self.succ[p]) == 1 and p != n

        seen: set[str] = set()
        paths: list[list[str]] = []
        starts = [n for n in self.nodes if not continues_back(n)]
        for s in starts + list(self.nodes):  # second pass catches pure cycles
            if s in seen:
                continue
            path = [s]
            seen.add(s)
            cur = s
            while len(self.succ[cur]) == 1:
                nxt = next(iter(self.succ[cur]))
                if len(self.pred[nxt]) != 1 or nxt in seen:
                    break
                path.append(nxt)
                seen.add(nxt)
                cur = nxt
            paths.append(path)
        return paths

    def greedy_contigs(self, min_len: int) -> list[tuple[str, float]]:
        """Greedy heaviest-path walks seeded from the heaviest k-mers."""
        used: set[str] = set()
        contigs: list[tuple[str, float]] = []
        for seed in sorted(self.nodes, key=lambda n: (-self.nodes[n], n)):
            if seed in used:
                continue
            used.add(seed)
            right = self._walk(seed, self.succ, self.pred, used)
            left = self._walk(seed, self.pred, self.succ, used)
            path = left[::-1] + [seed] + right
            seq = path[0] + "".join(n[-1] for n in path[1:])
            if len(seq) >= min_len:
                w = sum(self.nodes[n] for n in path) / len(path)
                contigs.append((seq, w))
        return contigs

    @staticmethod
    def _walk(start, fwd, back, used) -> list[str]:
        path: list[str] = []
        cur = start
        while True:
            cands = [(w, v) for v, w in fwd[cur].items() if v not in used]
            if not cands:
                return path
            w, v = max(cands)
            if w < max(back[v].values()):  # not the reciprocal best: stop at repeat/branch
                return path
            used.add(v)
            path.append(v)
            cur = v

    def to_gfa(self, path) -> None:
        tigs = self.unitigs()
        k = self.k
        first_of = {}
        last_of = {}
        with open(path, "w") as fh:
            fh.write("H\tVN:Z:1.0\n")
            for i, t in enumerate(tigs):
                seq = t[0] + "".join(n[-1] for n in t[1:])
                w = sum(self.nodes[n] for n in t) / len(t)
                fh.write(f"S\tu{i}\t{seq}\tLN:i:{len(seq)}\tKC:i:{round(w * len(t))}\tWT:f:{w:.3f}\n")
                first_of[t[0]] = i
                last_of[t[-1]] = i
            for t_last, i in last_of.items():
                for v in self.succ[t_last]:
                    if v in first_of:
                        fh.write(f"L\tu{i}\t+\tu{first_of[v]}\t+\t{k - 1}M\n")


def assemble_dbg(units: list[Unit], p: DBGParams) -> tuple[list[tuple[str, float]], DBG, dict]:
    g = DBG(p.k)
    g.add_units(units)
    g.build(p.min_kmer_weight, p.min_edge_weight)
    stats = {"kmers": len(g.kmer_w), "solid_kmers": len(g.nodes)}
    stats.update(g.clean(p))
    contigs = g.greedy_contigs(p.min_contig_len)
    stats["contigs"] = len(contigs)
    return contigs, g, stats

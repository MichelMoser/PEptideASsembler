"""Assembly pipeline: reads -> units -> (DBG | OLC | hybrid) -> polish -> contigs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agreement import AgreementParams, apply_agreement
from .dbg import DBG, DBGParams, assemble_dbg
from .io.fasta import write_fasta
from .model import Contig, Read, Unit
from .olc import OLCParams, assemble_olc
from .polish import PolishParams, polish, remove_redundant, score_contigs
from .preprocess import PreprocessParams, collapse, preprocess
from .tokens import Alphabet

MODES = ("dbg", "olc", "hybrid")


@dataclass
class AssemblyParams:
    mode: str = "hybrid"
    ks: list[int] = field(default_factory=lambda: [5])
    pre: PreprocessParams = field(default_factory=PreprocessParams)
    agreement: AgreementParams | None = field(default_factory=AgreementParams)
    dbg: DBGParams = field(default_factory=DBGParams)
    olc: OLCParams = field(default_factory=OLCParams)
    polish: PolishParams = field(default_factory=PolishParams)
    min_contig_len: int = 8
    min_contig_spectra: int = 2
    pass_min_spectra: int = 3          # PASS filter for contigs.fasta
    pass_min_mean_depth: float = 2.0
    pass_min_mean_conf: float = 0.6
    redundancy_identity: float = 0.75  # purge contigs this similar to a better one
    redundancy_cover: float = 0.8      # ... over this fraction of their length
    merge_rounds: int = 3


@dataclass
class AssemblyResult:
    contigs: list[Contig]
    reads: list[Read]
    units: list[Unit]
    placements: list
    graphs: dict[int, DBG]
    stats: dict


def _contig_units(contigs: list[str], weight: float = 1.0) -> list[Unit]:
    return [Unit(seq=s, conf=[1.0] * len(s), weight=weight, members=[]) for s in contigs]


def _drop_substrings(seqs: list[str]) -> list[str]:
    seqs = sorted(set(seqs), key=lambda s: (-len(s), s))
    kept: list[str] = []
    for s in seqs:
        if not any(s in t for t in kept):
            kept.append(s)
    return kept


def assemble(reads: list[Read], params: AssemblyParams) -> AssemblyResult:
    if params.mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    stats: dict = {}
    if params.agreement is not None and params.agreement.strength > 0:
        stats["agreement"] = apply_agreement(reads, params.agreement)
    kept, stats["preprocess"] = preprocess(reads, params.pre)
    units = collapse(kept)
    stats["units"] = len(units)

    graphs: dict[int, DBG] = {}
    dbg_contigs: list[str] = []
    if params.mode in ("dbg", "hybrid"):
        for k in params.ks:
            p = DBGParams(**{**asdict(params.dbg), "k": k})
            contigs, g, st = assemble_dbg(units, p)
            graphs[k] = g
            stats[f"dbg_k{k}"] = st
            dbg_contigs.extend(s for s, _ in contigs)
        dbg_contigs = _drop_substrings(dbg_contigs)

    if params.mode == "dbg":
        draft = dbg_contigs
    else:
        pool = units if params.mode == "olc" else _contig_units(dbg_contigs, 1.0) + units
        pool.sort(key=lambda u: (-len(u.seq), -u.weight, u.seq))
        olc_contigs, stats["olc"] = assemble_olc(pool, params.olc)
        draft = [s for s, *_ in olc_contigs]
    stats["draft_contigs"] = len(draft)

    # polish with all reads; purge redundant contigs; merge overlapping ones; repeat
    contigs = polish(_drop_substrings(draft), units, params.polish)
    merge_olc = OLCParams(**{**asdict(params.olc), "min_contig_len": 1})
    for rnd in range(params.merge_rounds):
        scored, _ = score_contigs(contigs, units, params.polish)
        scored, dropped = remove_redundant(
            scored, params.polish.k, params.redundancy_identity, params.redundancy_cover
        )
        cunits = [Unit(c.seq, c.conf, max(c.mean_depth, 1e-6), []) for c in scored]
        cunits.sort(key=lambda u: (-len(u.seq), -u.weight, u.seq))
        merged, mst = assemble_olc(cunits, merge_olc)
        mst["redundant_dropped"] = dropped
        stats[f"merge_round{rnd + 1}"] = mst
        new = polish(_drop_substrings([s for s, *_ in merged]), units, params.polish)
        if sorted(new) == sorted(contigs):
            break
        contigs = new

    final, placements = score_contigs(contigs, units, params.polish)
    final, dropped = remove_redundant(final, params.polish.k, params.redundancy_identity, params.redundancy_cover)
    stats["final_redundant_dropped"] = dropped
    final, placements = score_contigs([c.seq for c in final], units, params.polish)
    spectrum_of = {r.id: r.spectrum for r in kept}
    out: list[Contig] = []
    for c in final:
        n_spec = len({spectrum_of[m] for m in c.members if m in spectrum_of})
        if len(c.seq) >= params.min_contig_len and n_spec >= params.min_contig_spectra:
            out.append(c)
    out.sort(key=lambda c: (-len(c.seq) * c.mean_conf, c.seq))
    for i, c in enumerate(out, 1):
        c.id = f"contig_{i}"
        n_spec = len({spectrum_of[m] for m in c.members if m in spectrum_of})
        c.passed = (
            n_spec >= params.pass_min_spectra
            and c.mean_depth >= params.pass_min_mean_depth
            and c.mean_conf >= params.pass_min_mean_conf
        )
    # keep placements on reported contigs, labelled with the contig id
    id_of = {id(c): c.id for c in out}
    kept_pl = [(pl, id_of[id(final[pl.contig])]) for pl in placements if id(final[pl.contig]) in id_of]
    stats["contigs"] = len(out)
    stats["contigs_pass"] = sum(c.passed for c in out)
    stats["assembled_length"] = sum(len(c.seq) for c in out)
    stats["units_placed"] = len({pl.unit for pl, _ in kept_pl})
    stats["reads_placed"] = sum(len(units[u].members) for u in {pl.unit for pl, _ in kept_pl})
    return AssemblyResult(out, kept, units, kept_pl, graphs, stats)


# ---------------------------------------------------------------- output


def write_outputs(res: AssemblyResult, alphabet: Alphabet, outdir: str | Path, params: AssemblyParams) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    spectrum_of = {r.id: r.spectrum for r in res.reads}

    def header(c: Contig) -> str:
        return (
            f"{c.id} len={len(c.seq)} mean_conf={c.mean_conf:.3f} mean_depth={c.mean_depth:.2f} "
            f"spectra={len({spectrum_of.get(m) for m in c.members})} "
            f"filter={'PASS' if c.passed else 'LOW_SUPPORT'}"
        )

    write_fasta(outdir / "contigs.fasta", ((header(c), alphabet.plain(c.seq)) for c in res.contigs if c.passed))
    write_fasta(outdir / "contigs_all.fasta", ((header(c), alphabet.plain(c.seq)) for c in res.contigs))

    with open(outdir / "contigs.tsv", "w") as fh:
        fh.write("contig\tfilter\tlength\tmean_conf\tmin_conf\tmean_depth\tn_reads\tn_spectra\tlow_conf_positions\tsequence\tproforma\n")
        for c in res.contigs:
            low = sum(1 for x in c.conf if x < 0.5)
            fh.write(
                f"{c.id}\t{'PASS' if c.passed else 'LOW_SUPPORT'}\t{len(c.seq)}\t{c.mean_conf:.4f}\t{min(c.conf):.4f}\t{c.mean_depth:.3f}\t"
                f"{len(c.members)}\t{len({spectrum_of.get(m) for m in c.members})}\t{low}\t"
                f"{alphabet.plain(c.seq)}\t{alphabet.decode(c.seq)}\n"
            )

    with open(outdir / "positions.tsv", "w") as fh:
        fh.write("contig\tposition\tresidue\tconfidence\tdepth\tbest_alternative\n")
        for c in res.contigs:
            for i, (ch, cf, dp, alt) in enumerate(zip(c.seq, c.conf, c.depth, c.alt), 1):
                fh.write(f"{c.id}\t{i}\t{alphabet.decode(ch)}\t{cf:.4f}\t{dp:.3f}\t{alt}\n")

    with open(outdir / "placements.tsv", "w") as fh:
        fh.write("read\tsource\tspectrum\tpeptide\tread_conf\tcontig\tstart\tend\tidentity\tshare\n")
        reads_by_id = {r.id: r for r in res.reads}
        for pl, cid in res.placements:
            u = res.units[pl.unit]
            for rid in u.members:
                r = reads_by_id[rid]
                rc = ",".join(f"{x:.2f}" for x in r.conf)
                fh.write(
                    f"{rid}\t{r.source}\t{r.spectrum}\t{alphabet.decode(r.seq)}\t{rc}\t{cid}\t"
                    f"{pl.offset + 1}\t{pl.offset + len(u.seq)}\t{pl.identity:.3f}\t{pl.share:.3f}\n"
                )

    for k, g in res.graphs.items():
        g.to_gfa(outdir / f"graph_k{k}.gfa")

    with open(outdir / "run.json", "w") as fh:
        json.dump({"params": asdict(params), "stats": res.stats}, fh, indent=2, default=str)


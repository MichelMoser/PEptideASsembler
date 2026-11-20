"""Command line interface: ``peas assemble | simulate | evaluate``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .agreement import AgreementParams
from .assemble import MODES, AssemblyParams, assemble, write_outputs
from .dbg import DBGParams
from .evaluate import evaluate
from .io import read_casanovo, read_fasta, read_instanovo
from .olc import OLCParams
from .polish import PolishParams
from .preprocess import PreprocessParams
from .simulate import DEFAULT_ENZYMES, ENZYMES, SimParams, simulate
from .tokens import Alphabet


def _log(msg: str) -> None:
    print(f"[peas] {msg}", file=sys.stderr)


def cmd_assemble(a: argparse.Namespace) -> int:
    if not a.casanovo and not a.instanovo:
        _log("error: give at least one --casanovo or --instanovo file")
        return 2
    t0 = time.time()
    alphabet = Alphabet(strip_mods=not a.keep_mods, merge_il=not a.no_merge_il)
    reads = []
    for f in a.casanovo or []:
        r = read_casanovo(f, alphabet, reverse_scores=a.casanovo_reverse_scores)
        _log(f"{f}: {len(r)} Casanovo predictions")
        reads += r
    for f in a.instanovo or []:
        r = read_instanovo(f, alphabet, use_beams=not a.no_beams)
        _log(f"{f}: {len(r)} InstaNovo predictions")
        reads += r

    params = AssemblyParams(
        mode=a.mode,
        agreement=None if a.no_agreement else AgreementParams(strength=a.agreement_strength),
        ks=a.k,
        pre=PreprocessParams(
            min_score=a.min_score, trim_conf=a.trim_conf, trim_method=a.trim_method,
            trim_conf_by_source={k: v for k, v in (("casanovo", a.casanovo_trim_conf),
                                                   ("instanovo", a.instanovo_trim_conf)) if v is not None},
            min_len=a.min_read_len,
            precursor_tol_ppm=a.precursor_tol_ppm,
            precursor_mismatch_weight=a.precursor_mismatch_weight,
            min_mean_conf=a.min_mean_conf, max_rank=a.max_rank,
            source_weights={"casanovo": a.casanovo_weight, "instanovo": a.instanovo_weight},
        ),
        dbg=DBGParams(min_kmer_weight=a.min_kmer_weight, min_edge_weight=a.min_kmer_weight,
                      relative_edge_ratio=a.relative_edge_ratio),
        olc=OLCParams(k=a.olc_k, min_overlap=a.min_overlap, min_identity=a.min_identity,
                      confident_mismatch=a.confident_mismatch),
        polish=PolishParams(k=a.olc_k, min_overlap=a.min_overlap, min_identity=a.map_identity,
                            rounds=a.polish_rounds, extend_min_evidence=a.extend_min_evidence),
        min_contig_len=a.min_contig_len,
        min_contig_spectra=a.min_contig_spectra,
        redundancy_identity=a.redundancy_identity,
        pass_min_spectra=a.pass_min_spectra,
        pass_min_mean_depth=a.pass_min_depth,
        pass_min_mean_conf=a.pass_min_conf,
    )
    res = assemble(reads, params)
    write_outputs(res, alphabet, a.output, params)
    st = res.stats
    ag = st.get("agreement", {})
    for line in ag.get("pairings", []):
        _log(f"paired runs: {line}")
    if ag and not ag.get("pairings"):
        _log("no Casanovo/InstaNovo run pairing found (different spectra or identifiers?)")
    pp = st["preprocess"]
    if pp.get("precursor_ok") or pp.get("precursor_mismatch"):
        _log(f"precursor check: {pp.get('precursor_ok', 0)} within tolerance, "
             f"{pp.get('precursor_mismatch', 0)} outside")
    _log(f"reads kept {st['preprocess'].get('kept', 0)}/{st['preprocess'].get('input', 0)}, "
         f"unique sequences {st['units']}, contigs {st['contigs']} ({st['contigs_pass']} PASS) "
         f"({st['assembled_length']} aa), {st['reads_placed']} reads placed; "
         f"{time.time() - t0:.1f}s -> {a.output}")
    return 0


def cmd_simulate(a: argparse.Namespace) -> int:
    sp = SimParams(seed=a.seed, protein_length=a.length, spectra_per_residue=a.depth,
                   sub_rate=a.sub_rate, enzymes=tuple(a.enzymes), copies=a.copies)
    info = simulate(a.output, sp, a.reference)
    _log(f"simulated {info} -> {a.output}")
    return 0


def cmd_evaluate(a: argparse.Namespace) -> int:
    contigs = read_fasta(a.contigs)
    ref = read_fasta(a.reference)
    ev = evaluate(contigs, ref)
    print("contig\tprotein\tcontig_len\tc_start\tc_end\tref_start\tref_end\taligned\tidentity\tgaps")
    for h in ev["hits"]:
        print(f"{h.contig}\t{h.protein}\t{h.contig_len}\t{h.c_start + 1}\t{h.c_end}\t"
              f"{h.r_start + 1}\t{h.r_end}\t{h.aligned}\t{h.identity:.3f}\t{h.gaps}")
    summary = {k: v for k, v in ev.items() if k != "hits"}
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="peas", description="PEptideASsembler: uncertainty-aware de novo peptide assembly")
    ap.add_argument("--version", action="version", version=f"peas {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("assemble", help="assemble Casanovo/InstaNovo predictions into contigs",
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    io = s.add_argument_group("input/output")
    io.add_argument("--casanovo", nargs="+", metavar="MZTAB", help="Casanovo mzTab file(s)")
    io.add_argument("--instanovo", nargs="+", metavar="CSV", help="InstaNovo prediction CSV file(s)")
    io.add_argument("-o", "--output", required=True, help="output directory")
    io.add_argument("--keep-mods", action="store_true", help="keep modifications as distinct residues")
    io.add_argument("--no-merge-il", action="store_true", help="do not merge I and L")
    io.add_argument("--casanovo-reverse-scores", action="store_true",
                    help="aa scores are C->N (Casanovo < 4.3.0)")
    io.add_argument("--precursor-tol-ppm", type=float, default=50.0,
                    help="precursor m/z tolerance (isotope 0/+1); computed by peas for every candidate")
    io.add_argument("--precursor-mismatch-weight", type=float, default=0.5,
                    help="weight multiplier for candidates outside the precursor tolerance (0 = drop)")
    io.add_argument("--no-agreement", action="store_true",
                    help="do not pair Casanovo/InstaNovo predictions of the same spectrum")
    io.add_argument("--agreement-strength", type=float, default=0.5,
                    help="how strongly cross-tool (dis)agreement changes residue confidence (0-1)")
    io.add_argument("--no-beams", action="store_true", help="use only the top InstaNovo beam")

    pre = s.add_argument_group("read filtering / uncertainty")
    pre.add_argument("--min-score", type=float, default=0.0, help="min peptide score/probability")
    pre.add_argument("--trim-conf", type=float, default=0.3, help="residue confidence threshold for end trimming")
    pre.add_argument("--casanovo-trim-conf", type=float, default=None, help="override --trim-conf for Casanovo")
    pre.add_argument("--instanovo-trim-conf", type=float, default=None, help="override --trim-conf for InstaNovo")
    pre.add_argument("--trim-method", choices=["mott", "simple"], default="mott",
                     help="mott: cut where the summed deficit (thr - conf) is maximal; simple: stop at first good residue")
    pre.add_argument("--min-read-len", type=int, default=6)
    pre.add_argument("--min-mean-conf", type=float, default=0.4)
    pre.add_argument("--max-rank", type=int, default=4, help="max beam/top-n rank used")
    pre.add_argument("--casanovo-weight", type=float, default=1.0)
    pre.add_argument("--instanovo-weight", type=float, default=1.0)

    asm = s.add_argument_group("assembly")
    asm.add_argument("--mode", choices=MODES, default="hybrid")
    asm.add_argument("-k", type=int, nargs="+", default=[5], help="de Bruijn k-mer size(s)")
    asm.add_argument("--min-kmer-weight", type=float, default=0.5)
    asm.add_argument("--relative-edge-ratio", type=float, default=0.15)
    asm.add_argument("--olc-k", type=int, default=4, help="seed k-mer size for overlaps and mapping")
    asm.add_argument("--min-overlap", type=int, default=5)
    asm.add_argument("--min-identity", type=float, default=0.85)
    asm.add_argument("--confident-mismatch", type=float, default=0.9,
                     help="mismatch between residues both above this confidence vetoes an overlap")
    asm.add_argument("--map-identity", type=float, default=0.8)
    asm.add_argument("--polish-rounds", type=int, default=4)
    asm.add_argument("--extend-min-evidence", type=float, default=1.0)
    asm.add_argument("--min-contig-len", type=int, default=8)
    asm.add_argument("--min-contig-spectra", type=int, default=2, help="contigs below are not reported at all")
    asm.add_argument("--redundancy-identity", type=float, default=0.75,
                     help="purge contigs this similar to a better contig (unless they carry supported variants)")
    flt = s.add_argument_group("PASS filter (contigs.fasta; contigs_all.fasta has everything)")
    flt.add_argument("--pass-min-spectra", type=int, default=3)
    flt.add_argument("--pass-min-depth", type=float, default=2.0)
    flt.add_argument("--pass-min-conf", type=float, default=0.6)
    s.set_defaults(func=cmd_assemble)

    s = sub.add_parser("simulate", help="simulate Casanovo/InstaNovo outputs from a protein")
    s.add_argument("-o", "--output", required=True)
    s.add_argument("--reference", help="protein FASTA (default: random protein)")
    s.add_argument("--length", type=int, default=300)
    s.add_argument("--depth", type=float, default=1.5,
                   help="spectra per digest = depth x total protein length (one LC-MS run per enzyme)")
    s.add_argument("--sub-rate", type=float, default=0.04, help="base per-residue substitution rate (scaled per enzyme)")
    s.add_argument("--enzymes", nargs="+", default=list(DEFAULT_ENZYMES), choices=sorted(ENZYMES),
                   help="one digest (file pair) per enzyme")
    s.add_argument("--copies", type=int, default=200, help="protein molecules digested per enzyme")
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("evaluate", help="align contigs to reference proteins")
    s.add_argument("--contigs", required=True)
    s.add_argument("--reference", required=True)
    s.set_defaults(func=cmd_evaluate)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())

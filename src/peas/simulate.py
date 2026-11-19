"""Simulate de novo peptide predictions with realistic error/confidence profiles.

Writes a Casanovo-v5 style mzTab and an InstaNovo-v1 style CSV for the same
set of simulated spectra, plus the ground truth.  Used for development and
benchmarking; not a substitute for real data.

Error model (per prediction):

* I/L reported at random (indistinguishable by mass)
* substitutions, preferentially near-isobaric (K<->Q, N<->D, Q<->E, F<->M),
  with mostly low confidence, a fraction confidently wrong
* adjacent residue swaps (classic de novo error, low confidence)
* N -> GG / GG -> N (same mass, length changing)
* elevated error rate at the termini
* occasional garbage predictions (random sequence, low confidence)
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from .io.fasta import read_fasta, write_fasta

AA_FREQ = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37, "Q": 3.93, "E": 6.75,
    "G": 7.07, "H": 2.27, "I": 5.96, "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86,
    "P": 4.70, "S": 6.56, "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87,
}
SIMILAR = {"K": "Q", "Q": "KE", "N": "D", "D": "N", "E": "Q", "F": "M", "M": "F", "I": "L", "L": "I"}

ENZYMES = {
    "trypsin": (re.compile(r"(?<=[KR])(?!P)"), ),
    "chymotrypsin": (re.compile(r"(?<=[FWYL])(?!P)"), ),
    "aspn": (re.compile(r"(?=D)"), ),
    "gluc": (re.compile(r"(?<=E)"), ),
}


@dataclass
class SimParams:
    seed: int = 1
    protein_length: int = 300
    enzymes: tuple[str, ...] = ("trypsin", "chymotrypsin", "aspn", "gluc")
    missed_cleavages: int = 2
    min_len: int = 7
    max_len: int = 30
    mean_spectra_per_peptide: float = 3.0
    nonspecific_peptides: int = 40
    sub_rate: float = 0.04
    swap_rate: float = 0.015
    terminal_factor: float = 3.0
    confident_error_frac: float = 0.15
    garbage_rate: float = 0.05
    oxidation_rate: float = 0.2


def random_protein(rng: random.Random, length: int) -> str:
    aas, w = zip(*AA_FREQ.items())
    return "M" + "".join(rng.choices(aas, weights=w, k=length - 1))


def digest(protein: str, enzyme: str, missed: int, min_len: int, max_len: int) -> list[tuple[int, str]]:
    sites = [0] + [m.start() for m in ENZYMES[enzyme][0].finditer(protein) if 0 < m.start() < len(protein)] + [len(protein)]
    sites = sorted(set(sites))
    peps = []
    for i in range(len(sites) - 1):
        for j in range(i + 1, min(i + 2 + missed, len(sites))):
            a, b = sites[i], sites[j]
            if min_len <= b - a <= max_len:
                peps.append((a, protein[a:b]))
    return peps


def _poisson(rng: random.Random, lam: float) -> int:
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def corrupt(rng: random.Random, pep: str, sp: SimParams, tool_noise: float):
    """Return (residues, confidences) of a simulated prediction."""
    aas = list(AA_FREQ)
    if rng.random() < sp.garbage_rate:
        n = max(sp.min_len, len(pep) + rng.randint(-2, 2))
        return [rng.choice(aas) for _ in range(n)], [rng.betavariate(2, 4) for _ in range(n)]

    res = [("L" if a in "IL" and rng.random() < 0.5 else ("I" if a in "IL" else a)) for a in pep]
    conf = [rng.betavariate(12, 1) for _ in res]
    n = len(res)
    i = 0
    while i < n:
        term = sp.terminal_factor if (i < 2 or i >= n - 2) else 1.0
        r = rng.random()
        if r < sp.sub_rate * term * tool_noise:
            a = res[i]
            if a in SIMILAR and rng.random() < 0.6:
                res[i] = rng.choice(SIMILAR[a])
            else:
                res[i] = rng.choice([x for x in aas if x != a])
            conf[i] = rng.betavariate(9, 2) if rng.random() < sp.confident_error_frac else rng.betavariate(2, 3)
        elif r < (sp.sub_rate + sp.swap_rate) * term * tool_noise and i < n - 1 and res[i] != res[i + 1]:
            res[i], res[i + 1] = res[i + 1], res[i]
            conf[i], conf[i + 1] = rng.betavariate(2, 3), rng.betavariate(2, 3)
            i += 1
        i += 1
    # mass-equivalent length changes (rare)
    if rng.random() < 0.02 * tool_noise:
        idx = [j for j, a in enumerate(res) if a == "N"]
        if idx:
            j = rng.choice(idx)
            res[j : j + 1] = ["G", "G"]
            conf[j : j + 1] = [rng.betavariate(2, 3), rng.betavariate(2, 3)]
    return res, conf


def _decorate(res: list[str], rng: random.Random, sp: SimParams, style: str) -> list[str]:
    out = []
    for a in res:
        if a == "C":
            out.append("C[Carbamidomethyl]" if style == "casanovo" else "C[UNIMOD:4]")
        elif a == "M" and rng.random() < sp.oxidation_rate:
            out.append("M[Oxidation]" if style == "casanovo" else "M[UNIMOD:35]")
        else:
            out.append(a)
    return out


CASANOVO_COLS = [
    "sequence", "PSM_ID", "accession", "unique", "database", "database_version", "search_engine",
    "search_engine_score[1]", "modifications", "retention_time", "charge", "exp_mass_to_charge",
    "calc_mass_to_charge", "spectra_ref", "pre", "post", "start", "end", "opt_global_aa_scores",
    "opt_global_cv_MS:1003169_proforma_peptidoform_sequence",
]


def simulate(outdir: str | Path, sp: SimParams, reference: str | Path | None = None) -> dict:
    rng = random.Random(sp.seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if reference:
        proteins = read_fasta(reference)
    else:
        proteins = [("sim_protein_1", random_protein(rng, sp.protein_length))]
    write_fasta(outdir / "reference.fasta", proteins)

    peptides: list[tuple[str, int, str]] = []
    for name, prot in proteins:
        seen = set()
        for enz in sp.enzymes:
            for start, pep in digest(prot, enz, sp.missed_cleavages, sp.min_len, sp.max_len):
                if (start, pep) not in seen:
                    seen.add((start, pep))
                    peptides.append((name, start, pep))
        for _ in range(sp.nonspecific_peptides):
            L = rng.randint(sp.min_len, min(sp.max_len, 20))
            s = rng.randint(0, max(0, len(prot) - L))
            peptides.append((name, s, prot[s : s + L]))

    spectra = []
    for name, start, pep in peptides:
        for _ in range(_poisson(rng, sp.mean_spectra_per_peptide)):
            spectra.append((name, start, pep))
    rng.shuffle(spectra)

    with open(outdir / "truth.tsv", "w") as fh:
        fh.write("spectrum_index\tprotein\tstart\tpeptide\n")
        for i, (name, start, pep) in enumerate(spectra):
            fh.write(f"{i}\t{name}\t{start + 1}\t{pep}\n")

    # Casanovo v5 mzTab
    with open(outdir / "casanovo.mztab", "w") as fh:
        fh.write("MTD\tmzTab-version\t1.0.0\nMTD\tmzTab-mode\tSummary\nMTD\tmzTab-type\tIdentification\n")
        fh.write("MTD\tms_run[1]-location\tfile:///simulated.mgf\n")
        fh.write("MTD\tsoftware[1]\t[MS, MS:1003281, Casanovo, 5.x-simulated]\n")
        fh.write("PSH\t" + "\t".join(CASANOVO_COLS) + "\n")
        for i, (name, start, pep) in enumerate(spectra):
            res, conf = corrupt(rng, pep, sp, tool_noise=1.0)
            toks = _decorate(res, rng, sp, "casanovo")
            score = math.prod(conf)
            row = {c: "null" for c in CASANOVO_COLS}
            row.update({
                "sequence": "".join(res), "PSM_ID": str(i + 1), "search_engine": "[MS, MS:1003281, Casanovo, ]",
                "search_engine_score[1]": f"{score}", "charge": "2", "spectra_ref": f"ms_run[1]:index={i}",
                "opt_global_aa_scores": ",".join(f"{c:.5f}" for c in conf),
                "opt_global_cv_MS:1003169_proforma_peptidoform_sequence": "".join(toks),
            })
            fh.write("PSM\t" + "\t".join(row[c] for c in CASANOVO_COLS) + "\n")

    # InstaNovo v1 CSV with 2 beams
    import csv

    with open(outdir / "instanovo.csv", "w", newline="") as fh:
        cols = ["spectrum_id", "scan_number", "precursor_mz", "precursor_charge", "prediction_id",
                "predictions", "predictions_tokenised", "log_probs", "token_log_probs", "delta_mass_ppm"]
        for b in range(2):
            cols += [f"predictions_beam_{b}", f"predictions_log_probability_beam_{b}",
                     f"predictions_token_log_probabilities_beam_{b}"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for i, (name, start, pep) in enumerate(spectra):
            beams = []
            for b in range(2):
                res, conf = corrupt(rng, pep, sp, tool_noise=0.8 if b == 0 else 2.0)
                if b == 1:
                    conf = [c * 0.8 for c in conf]
                toks = _decorate(res, rng, sp, "instanovo")
                lps = [math.log(max(c, 1e-6)) for c in conf]
                beams.append((toks, lps))
            beams.sort(key=lambda x: -sum(x[1]))
            toks, lps = beams[0]
            row = {
                "spectrum_id": f"simulated:{i}", "scan_number": i, "precursor_mz": 0.0, "precursor_charge": 2,
                "prediction_id": i, "predictions": "".join(toks), "predictions_tokenised": ", ".join(toks),
                "log_probs": sum(lps), "token_log_probs": str([round(x, 5) for x in lps]), "delta_mass_ppm": 0.0,
            }
            for b, (bt, bl) in enumerate(beams):
                row[f"predictions_beam_{b}"] = "".join(bt)
                row[f"predictions_log_probability_beam_{b}"] = sum(bl)
                row[f"predictions_token_log_probabilities_beam_{b}"] = str([round(x, 5) for x in bl])
            w.writerow(row)

    return {"proteins": len(proteins), "peptides": len(peptides), "spectra": len(spectra)}

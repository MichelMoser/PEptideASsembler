"""Simulate multi-protease de novo peptide predictions.

For every digest a Casanovo-v5 style mzTab and an InstaNovo-v1 style CSV
are written (``casanovo_<enzyme>.mztab``, ``instanovo_<enzyme>.csv``), as
each digest is normally acquired and sequenced separately, plus the ground
truth.  Used for development and benchmarking; not a substitute for real
data.

Digestion is **probabilistic**: many protein copies are cut, every bond
with its own rule-based cleavage probability (incl. missed cleavages,
low-specificity sites for chymotrypsin, D-cleavage for Glu-C and a broad,
unspecific proteinase K).  Peptide abundance (and thus the number of
spectra) follows from how often a peptide is produced, times a random
detectability factor.

Error model (per prediction), scaled per enzyme because the de novo models
are trained mostly on tryptic peptides (``Enzyme.noise``):

* I/L reported at random (indistinguishable by mass)
* substitutions, preferentially near-isobaric (K<->Q, N<->D, Q<->E, F<->M),
  mostly with low confidence; a fraction confidently wrong (more often for
  non-tryptic peptides: worse confidence calibration)
* adjacent residue swaps (classic de novo error, low confidence)
* N -> GG (same mass, length changing)
* elevated error rate at the termini
* occasional garbage predictions (random sequence, low confidence)
"""

from __future__ import annotations

import csv
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .io.fasta import read_fasta, write_fasta
from .mass import C13_DELTA, PROTON, peptide_mass
from .tokens import tokenize

AA_FREQ = {
    "A": 8.25, "R": 5.53, "N": 4.06, "D": 5.45, "C": 1.37, "Q": 3.93, "E": 6.75,
    "G": 7.07, "H": 2.27, "I": 5.96, "L": 9.66, "K": 5.84, "M": 2.42, "F": 3.86,
    "P": 4.70, "S": 6.56, "T": 5.34, "W": 1.08, "Y": 2.92, "V": 6.87,
}
# near-isobaric / common confusions; "m" = oxidised methionine (147.0354 vs F 147.0684)
SIMILAR = {"K": "Q", "Q": "KE", "N": "D", "D": "N", "E": "Q", "F": "m", "m": "F", "I": "L", "L": "I"}


@dataclass(frozen=True)
class Rule:
    """Cleave C-terminal of ``after`` (or N-terminal of ``before``) residues."""

    residues: str
    prob: float
    not_before: str = ""     # no cleavage if the next residue is one of these
    before: bool = False     # rule describes P1' (e.g. Asp-N) instead of P1


@dataclass(frozen=True)
class Enzyme:
    name: str
    rules: tuple[Rule, ...]
    background: float = 0.002   # unspecific cleavage (in-source / contaminating proteases)
    noise: float = 1.0          # de novo error multiplier for this digest
    miscalibration: float = 1.0  # multiplier of the confidently-wrong fraction

    def site_probs(self, protein: str) -> list[float]:
        """Cleavage probability of the bond before position i (i = 1..n-1)."""
        probs = [0.0] * len(protein)
        for i in range(1, len(protein)):
            p1, p1p = protein[i - 1], protein[i]
            p = self.background
            for r in self.rules:
                hit = (p1p in r.residues) if r.before else (p1 in r.residues)
                if hit and p1p not in r.not_before:
                    p = max(p, r.prob)
            probs[i] = p
        return probs


ENZYMES: dict[str, Enzyme] = {
    # Trypsin/Lys-C mix: Lys-C also cleaves K-P, so K bonds are cut regardless of P
    "trypsin_lysc": Enzyme("trypsin_lysc", (Rule("K", 0.95), Rule("R", 0.9, not_before="P"))),
    "trypsin": Enzyme("trypsin", (Rule("KR", 0.9, not_before="P"),)),
    "lysc": Enzyme("lysc", (Rule("K", 0.95),), noise=1.1),
    # high specificity F/Y/W, low specificity L/M
    "chymotrypsin": Enzyme(
        "chymotrypsin",
        (Rule("FYW", 0.8, not_before="P"), Rule("LM", 0.3, not_before="P")),
        noise=1.5, miscalibration=1.5,
    ),
    # Glu-C: E (bicarbonate); D at lower rate (phosphate buffer activity)
    "gluc": Enzyme("gluc", (Rule("E", 0.85, not_before="P"), Rule("D", 0.15, not_before="P")),
                   noise=1.5, miscalibration=1.5),
    "aspn": Enzyme("aspn", (Rule("D", 0.85, before=True),), noise=1.5, miscalibration=1.5),
    # proteinase K: broad, prefers aliphatic/aromatic P1; effectively unspecific
    "proteinase_k": Enzyme(
        "proteinase_k",
        (Rule("AVLIFYWM", 0.3), Rule("GSTCNQ", 0.08)),
        background=0.03, noise=2.0, miscalibration=2.0,
    ),
}

DEFAULT_ENZYMES = ("trypsin_lysc", "chymotrypsin", "gluc", "proteinase_k")


@dataclass
class SimParams:
    seed: int = 1
    protein_length: int = 300
    enzymes: tuple[str, ...] = DEFAULT_ENZYMES
    copies: int = 200                 # protein molecules digested per enzyme
    min_len: int = 7                  # detectable peptide length range
    max_len: int = 30
    spectra_per_residue: float = 1.5  # spectrum budget per digest = this x total protein length
    max_peptides_per_enzyme: int | None = None
    sub_rate: float = 0.04
    swap_rate: float = 0.015
    terminal_factor: float = 3.0
    confident_error_frac: float = 0.15
    garbage_rate: float = 0.05
    oxidation_rate: float = 0.2       # per Met, part of the true peptidoform
    precursor_sd_ppm: float = 3.0     # instrument precursor m/z error
    isotope_error_rate: float = 0.1   # +1 isotope peak picked as precursor
    difficulty_sd: float = 0.5        # log-sd of per-spectrum difficulty shared by both tools
    enzyme_noise: dict[str, float] = field(default_factory=dict)  # override Enzyme.noise


def random_protein(rng: random.Random, length: int) -> str:
    aas, w = zip(*AA_FREQ.items())
    return "M" + "".join(rng.choices(aas, weights=w, k=length - 1))


def digest(protein: str, enzyme: Enzyme, rng: random.Random, copies: int,
           min_len: int, max_len: int) -> Counter:
    """Digest ``copies`` molecules; return Counter of (start, peptide)."""
    probs = enzyme.site_probs(protein)
    counts: Counter = Counter()
    n = len(protein)
    for _ in range(copies):
        start = 0
        for i in range(1, n + 1):
            if i == n or rng.random() < probs[i]:
                if min_len <= i - start <= max_len:
                    counts[(start, protein[start:i])] += 1
                start = i
    return counts


def _poisson(rng: random.Random, lam: float) -> int:
    if lam > 30:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    l, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= l:
            return k
        k += 1


def corrupt(rng: random.Random, pep: str, sp: SimParams, noise: float, miscal: float = 1.0,
            garbage: bool | None = None):
    """Return (residues, confidences) of a simulated prediction.

    ``pep`` uses "m" for oxidised Met.  ``garbage`` (decided per spectrum, so
    shared by both tools) makes the prediction random."""
    aas = list(AA_FREQ)
    if garbage is None:
        garbage = rng.random() < min(0.5, sp.garbage_rate * noise)
    if garbage:
        n = max(sp.min_len, len(pep) + rng.randint(-2, 2))
        return [rng.choice(aas) for _ in range(n)], [rng.betavariate(2, 4) for _ in range(n)]

    res = [("L" if a in "IL" and rng.random() < 0.5 else ("I" if a in "IL" else a)) for a in pep]
    conf = [rng.betavariate(12, 1) for _ in res]
    n = len(res)
    conf_err = min(0.9, sp.confident_error_frac * miscal)
    i = 0
    while i < n:
        term = sp.terminal_factor if (i < 2 or i >= n - 2) else 1.0
        r = rng.random()
        if r < sp.sub_rate * term * noise:
            a = res[i]
            if a in SIMILAR and rng.random() < 0.6:
                res[i] = rng.choice(SIMILAR[a])
            else:
                res[i] = rng.choice([x for x in aas if x != a])
            conf[i] = rng.betavariate(9, 2) if rng.random() < conf_err else rng.betavariate(2, 3)
        elif r < (sp.sub_rate + sp.swap_rate) * term * noise and i < n - 1 and res[i] != res[i + 1]:
            res[i], res[i + 1] = res[i + 1], res[i]
            conf[i], conf[i + 1] = rng.betavariate(2, 3), rng.betavariate(2, 3)
            i += 1
        i += 1
    if rng.random() < 0.02 * noise:  # mass-equivalent length change
        idx = [j for j, a in enumerate(res) if a == "N"]
        if idx:
            j = rng.choice(idx)
            res[j : j + 1] = ["G", "G"]
            conf[j : j + 1] = [rng.betavariate(2, 3), rng.betavariate(2, 3)]
    return res, conf


def _decorate(res: list[str], style: str) -> list[str]:
    out = []
    for a in res:
        if a == "C":
            out.append("C[Carbamidomethyl]" if style == "casanovo" else "C[UNIMOD:4]")
        elif a == "m":
            out.append("M[Oxidation]" if style == "casanovo" else "M[UNIMOD:35]")
        else:
            out.append(a)
    return out


def _mz(tokens: list[str], charge: int) -> float:
    mass = peptide_mass(tokenize("".join(tokens)))
    return (mass + charge * PROTON) / charge


@dataclass
class Spectrum:
    protein: str
    start: int
    peptide: str        # true peptidoform, "m" = oxidised Met
    charge: int
    precursor_mz: float
    difficulty: float
    garbage: bool


def _make_spectrum(rng: random.Random, name: str, start: int, pep: str, sp: SimParams, noise: float) -> Spectrum:
    pep = "".join("m" if a == "M" and rng.random() < sp.oxidation_rate else a for a in pep)
    charge = 2 if len(pep) <= 12 else (3 if rng.random() < 0.7 else 2)
    mz = _mz(_decorate(list(pep), "casanovo"), charge)
    mz *= 1 + rng.gauss(0, sp.precursor_sd_ppm) * 1e-6
    if rng.random() < sp.isotope_error_rate:
        mz += C13_DELTA / charge
    d = rng.lognormvariate(-sp.difficulty_sd**2 / 2, sp.difficulty_sd)
    garbage = rng.random() < min(0.5, sp.garbage_rate * noise * d)
    return Spectrum(name, start, pep, charge, mz, d, garbage)


CASANOVO_COLS = [
    "sequence", "PSM_ID", "accession", "unique", "database", "database_version", "search_engine",
    "search_engine_score[1]", "modifications", "retention_time", "charge", "exp_mass_to_charge",
    "calc_mass_to_charge", "spectra_ref", "pre", "post", "start", "end", "opt_global_aa_scores",
    "opt_global_cv_MS:1003169_proforma_peptidoform_sequence",
]
INSTANOVO_COLS = ["experiment_name", "spectrum_id", "scan_number", "precursor_mz", "precursor_charge", "prediction_id",
                  "predictions", "predictions_tokenised", "log_probs", "token_log_probs", "delta_mass_ppm"]
for _b in range(2):
    INSTANOVO_COLS += [f"predictions_beam_{_b}", f"predictions_log_probability_beam_{_b}",
                       f"predictions_token_log_probabilities_beam_{_b}"]


def _write_casanovo(path: Path, spectra, rng, sp: SimParams, enz: Enzyme, noise: float) -> None:
    with open(path, "w") as fh:
        fh.write("MTD\tmzTab-version\t1.0.0\nMTD\tmzTab-mode\tSummary\nMTD\tmzTab-type\tIdentification\n")
        fh.write(f"MTD\tms_run[1]-location\tfile:///simulated_{enz.name}.mgf\n")
        fh.write("MTD\tsoftware[1]\t[MS, MS:1003281, Casanovo, 5.x-simulated]\n")
        fh.write("PSH\t" + "\t".join(CASANOVO_COLS) + "\n")
        for i, sp_ in enumerate(spectra):
            res, conf = corrupt(rng, sp_.peptide, sp, noise * sp_.difficulty, enz.miscalibration, sp_.garbage)
            toks = _decorate(res, "casanovo")
            row = {c: "null" for c in CASANOVO_COLS}
            row.update({
                "sequence": "".join(res).upper(), "PSM_ID": str(i + 1),
                "search_engine": "[MS, MS:1003281, Casanovo, ]",
                "search_engine_score[1]": f"{math.prod(conf)}", "charge": str(sp_.charge),
                "exp_mass_to_charge": f"{sp_.precursor_mz:.6f}",
                "calc_mass_to_charge": f"{_mz(toks, sp_.charge):.6f}",
                "spectra_ref": f"ms_run[1]:index={i}",
                "opt_global_aa_scores": ",".join(f"{c:.5f}" for c in conf),
                "opt_global_cv_MS:1003169_proforma_peptidoform_sequence": "".join(toks),
            })
            fh.write("PSM\t" + "\t".join(row[c] for c in CASANOVO_COLS) + "\n")


def _write_instanovo(path: Path, spectra, rng, sp: SimParams, enz: Enzyme, noise: float) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INSTANOVO_COLS)
        w.writeheader()
        for i, sp_ in enumerate(spectra):
            beams = []
            for b in range(2):
                res, conf = corrupt(rng, sp_.peptide, sp, noise * sp_.difficulty * (0.8 if b == 0 else 2.0),
                                    enz.miscalibration, sp_.garbage)
                if b == 1:
                    conf = [c * 0.8 for c in conf]
                toks = _decorate(res, "instanovo")
                beams.append((toks, [math.log(max(c, 1e-6)) for c in conf]))
            beams.sort(key=lambda x: -sum(x[1]))
            toks, lps = beams[0]
            calc = _mz(toks, sp_.charge)
            ppm = min(((sp_.precursor_mz - iso * C13_DELTA / sp_.charge - calc) / calc * 1e6 for iso in (0, 1)), key=abs)
            row = {
                "experiment_name": f"simulated_{enz.name}", "spectrum_id": f"simulated_{enz.name}:{i}",
                "scan_number": i + 1, "precursor_mz": round(sp_.precursor_mz, 6),
                "precursor_charge": sp_.charge, "prediction_id": i, "predictions": "".join(toks),
                "predictions_tokenised": ", ".join(toks), "log_probs": sum(lps),
                "token_log_probs": str([round(x, 5) for x in lps]), "delta_mass_ppm": round(abs(ppm), 3),
            }
            for b, (bt, bl) in enumerate(beams):
                row[f"predictions_beam_{b}"] = "".join(bt)
                row[f"predictions_log_probability_beam_{b}"] = sum(bl)
                row[f"predictions_token_log_probabilities_beam_{b}"] = str([round(x, 5) for x in bl])
            w.writerow(row)


def simulate(outdir: str | Path, sp: SimParams, reference: str | Path | None = None) -> dict:
    rng = random.Random(sp.seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if reference:
        proteins = read_fasta(reference)
    else:
        proteins = [("sim_protein_1", random_protein(rng, sp.protein_length))]
    write_fasta(outdir / "reference.fasta", proteins)

    info: dict = {"proteins": len(proteins), "enzymes": {}}
    truth = open(outdir / "truth.tsv", "w")
    truth.write("enzyme\tspectrum_index\tprotein\tstart\tpeptide\tcharge\tprecursor_mz\tgarbage\n")
    for ename in sp.enzymes:
        if ename not in ENZYMES:
            raise ValueError(f"unknown enzyme {ename!r}; choose from {sorted(ENZYMES)}")
        enz = ENZYMES[ename]
        noise = sp.enzyme_noise.get(ename, enz.noise)
        peptides: list[tuple[str, int, str, int]] = []
        for name, prot in proteins:
            for (start, pep), c in digest(prot, enz, rng, sp.copies, sp.min_len, sp.max_len).items():
                peptides.append((name, start, pep, c))
        if sp.max_peptides_per_enzyme and len(peptides) > sp.max_peptides_per_enzyme:
            peptides.sort(key=lambda x: -x[3])
            peptides = peptides[: sp.max_peptides_per_enzyme]
        # equal spectrum budget per digest (one LC-MS run each), spread over
        # peptides by abundance x random detectability
        budget = sp.spectra_per_residue * sum(len(p) for _, p in proteins)
        intensity = [c * rng.lognormvariate(0, 0.7) for *_, c in peptides]
        tot = sum(intensity) or 1.0
        spectra = []
        for (name, start, pep, _), x in zip(peptides, intensity):
            spectra.extend([(name, start, pep)] * _poisson(rng, budget * x / tot))
        rng.shuffle(spectra)
        spectra = [_make_spectrum(rng, n, st, p, sp, noise) for n, st, p in spectra]
        for i, x in enumerate(spectra):
            truth.write(f"{ename}\t{i}\t{x.protein}\t{x.start + 1}\t{''.join(_decorate(list(x.peptide), 'casanovo'))}"
                        f"\t{x.charge}\t{x.precursor_mz:.6f}\t{int(x.garbage)}\n")
        _write_casanovo(outdir / f"casanovo_{ename}.mztab", spectra, rng, sp, enz, noise)
        _write_instanovo(outdir / f"instanovo_{ename}.csv", spectra, rng, sp, enz, noise)
        lens = [len(p[2]) for p in peptides]
        info["enzymes"][ename] = {
            "distinct_peptides": len(peptides),
            "spectra": len(spectra),
            "peptides_observed": len({(x.protein, x.start, x.peptide.upper()) for x in spectra}),
            "median_len": sorted(lens)[len(lens) // 2] if lens else 0,
        }
    truth.close()
    return info

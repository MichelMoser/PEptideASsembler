"""Peptide masses and precursor m/z error from tokenised peptides."""

from __future__ import annotations

import re

from .tokens import Residue

AA_MONO = {
    "G": 57.021464, "A": 71.037114, "S": 87.032028, "P": 97.052764, "V": 99.068414,
    "T": 101.047679, "C": 103.009185, "L": 113.084064, "I": 113.084064, "N": 114.042927,
    "D": 115.026943, "Q": 128.058578, "K": 128.094963, "E": 129.042593, "M": 131.040485,
    "H": 137.058912, "F": 147.068414, "R": 156.101111, "Y": 163.06332, "W": 186.079313,
}
H2O = 18.010565
PROTON = 1.007276
C13_DELTA = 1.003355

MOD_MASS = {
    "oxidation": 15.994915, "ox": 15.994915, "unimod:35": 15.994915,
    "carbamidomethyl": 57.021464, "unimod:4": 57.021464,
    "deamidated": 0.984016, "deamidation": 0.984016, "unimod:7": 0.984016,
    "acetyl": 42.010565, "unimod:1": 42.010565,
    "carbamyl": 43.005814, "unimod:5": 43.005814,
    "ammonia-loss": -17.026549, "unimod:385": -17.026549,
    "phospho": 79.966331, "p": 79.966331, "unimod:21": 79.966331,
}
_MOD_RE = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)|([+-]\d+(?:\.\d+)?)")


def mod_mass(mod: str) -> float | None:
    """Total mass of a modification string, e.g. ``[Oxidation]``, ``+15.995``,
    ``(+57.02)``, ``[UNIMOD:35]``, ``+43.006-17.027``.  None if unknown."""
    total = 0.0
    pos = 0
    for m in _MOD_RE.finditer(mod):
        if m.start() != pos:
            return None
        pos = m.end()
        content = next(g for g in m.groups() if g is not None)
        try:
            total += float(content)
        except ValueError:
            v = MOD_MASS.get(content.strip().lower())
            if v is None:
                return None
            total += v
    return total if pos == len(mod) else None


def peptide_mass(residues: list[Residue]) -> float | None:
    """Neutral monoisotopic mass, None if a residue/modification is unknown."""
    total = H2O
    for r in residues:
        m = AA_MONO.get(r.aa)
        if m is None:
            return None
        total += m
        for mod in (r.mod, r.nterm):
            if mod:
                mm = mod_mass(mod)
                if mm is None:
                    return None
                total += mm
    return total


def precursor_ppm(
    residues: list[Residue], precursor_mz: float, charge: int, isotopes: tuple[int, ...] = (0, 1)
) -> float | None:
    """Smallest signed ppm error of the observed precursor vs the peptide,
    allowing for selection of a heavier isotope peak."""
    if not charge or charge <= 0 or not precursor_mz or precursor_mz <= 0:
        return None
    mass = peptide_mass(residues)
    if mass is None:
        return None
    calc = (mass + charge * PROTON) / charge
    best = None
    for iso in isotopes:
        ppm = (precursor_mz - iso * C13_DELTA / charge - calc) / calc * 1e6
        if best is None or abs(ppm) < abs(best):
            best = ppm
    return best

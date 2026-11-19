"""Tokenisation of de novo peptide strings and the internal residue alphabet.

De novo tools write peptides in several notations:

* Casanovo v5 ProForma:   ``[Acetyl]-PEPM[Oxidation]C[Carbamidomethyl]K``
* Casanovo v4 mass tags:  ``+42.011PEPM+15.995C+57.021K``
* InstaNovo (UNIMOD):     ``[UNIMOD:1]-PEPM[UNIMOD:35]C[UNIMOD:4]K``
* InstaNovo legacy:       ``PEPM(ox)C(+57.02)K``

Every residue (with its modification and an optional N-terminal
modification folded into the first residue) is one *token*.  Internally a
peptide is a Python ``str`` with **one character per token**: unmodified
residues map to their one-letter code, modified residues map to characters
of the Unicode private use area.  This keeps k-mers as cheap string
slices (and maps 1:1 onto a ``Vec<u8>``/``&[u8]`` representation in a
future Rust core).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM = r"[+-]\d+(?:\.\d+)?"
_MOD = rf"(?:\[[^\]]*\]|\([^)]*\)|{_NUM})"
_NTERM_RE = re.compile(rf"^((?:\[[^\]]*\]|\([^)]*\))-?|(?:{_NUM})+)")
_RES_RE = re.compile(rf"([A-Z])((?:{_MOD})*)")

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"
_PUA_START = 0xE000


@dataclass(frozen=True, slots=True)
class Residue:
    """A residue with optional side-chain and N-terminal modifications."""

    aa: str
    mod: str = ""
    nterm: str = ""

    def __str__(self) -> str:
        prefix = f"{self.nterm}-" if self.nterm else ""
        return f"{prefix}{self.aa}{self.mod}"


def tokenize(peptide: str) -> list[Residue]:
    """Split a peptide string in any supported notation into residues."""
    s = peptide.strip()
    pos = 0
    nterm = ""
    m = _NTERM_RE.match(s)
    if m:
        nterm = m.group(1).rstrip("-")
        pos = m.end()
    residues: list[Residue] = []
    while pos < len(s):
        m = _RES_RE.match(s, pos)
        if m is None:
            raise ValueError(f"cannot parse peptide {peptide!r} at position {pos}")
        residues.append(Residue(m.group(1), m.group(2)))
        pos = m.end()
    if nterm and residues:
        first = residues[0]
        residues[0] = Residue(first.aa, first.mod, nterm)
    return residues


@dataclass
class Alphabet:
    """Maps residues to single characters used throughout the assembler.

    Parameters
    ----------
    strip_mods:
        Ignore all modifications (default).  Fixed modifications such as
        carbamidomethylation carry no assembly information and variable ones
        (oxidation, deamidation) would otherwise split the graph.
    merge_il:
        Isoleucine and leucine are isobaric and indistinguishable for de novo
        sequencing, so ``I`` is written as ``L`` (default).
    """

    strip_mods: bool = True
    merge_il: bool = True
    _to_char: dict[str, str] = field(default_factory=dict, repr=False)
    _to_token: dict[str, Residue] = field(default_factory=dict, repr=False)

    def encode_residue(self, r: Residue) -> str:
        aa = "L" if (self.merge_il and r.aa == "I") else r.aa
        if self.strip_mods or (not r.mod and not r.nterm):
            return aa
        r = Residue(aa, r.mod, r.nterm)
        key = str(r)
        c = self._to_char.get(key)
        if c is None:
            c = chr(_PUA_START + len(self._to_char))
            self._to_char[key] = c
            self._to_token[c] = r
        return c

    def encode(self, peptide: str) -> str:
        return "".join(self.encode_residue(r) for r in tokenize(peptide))

    def encode_residues(self, residues: list[Residue]) -> str:
        return "".join(self.encode_residue(r) for r in residues)

    def decode(self, seq: str) -> str:
        """Encoded sequence -> ProForma-like string (keeps modifications)."""
        return "".join(str(self._to_token[c]) if c in self._to_token else c for c in seq)

    def plain(self, seq: str) -> str:
        """Encoded sequence -> unmodified one-letter sequence (for FASTA)."""
        return "".join(self._to_token[c].aa if c in self._to_token else c for c in seq)

    def normalize_plain(self, seq: str) -> str:
        """Normalise a plain reference sequence the same way reads are."""
        seq = seq.upper()
        return seq.replace("I", "L") if self.merge_il else seq

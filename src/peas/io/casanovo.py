"""Casanovo mzTab parser (v4.x and v5.x).

Relevant facts about the format (checked against the Casanovo sources):

* v5: plain residues in ``sequence``, the full ProForma string in
  ``opt_global_cv_MS:1003169_proforma_peptidoform_sequence``, per-residue
  scores in ``opt_global_aa_scores`` (comma separated, N->C).
* v4.x: ``sequence`` holds the model string with mass tags (``M+15.995``),
  scores in ``opt_ms_run[1]_aa_scores``.  Before v4.3.0 the aa scores were
  written in reversed (C->N) order: use ``reverse_scores=True`` for those.
* Before v5.2.0 the peptide score is lowered by 1 when the prediction does
  not match the precursor m/z, so negative scores flag a precursor mismatch.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..model import Read
from ..tokens import Alphabet, tokenize

PROFORMA_COL = "opt_global_cv_MS:1003169_proforma_peptidoform_sequence"
SCAN_COL = "opt_global_cv_MS:1003057_scan_number"


def _aa_score_col(columns: list[str]) -> str | None:
    for c in columns:
        if c.startswith("opt_") and c.endswith("aa_scores"):
            return c
    return None


def read_casanovo(
    path: str | Path,
    alphabet: Alphabet,
    *,
    reverse_scores: bool = False,
    mismatch_weight: float = 0.5,
    source: str = "casanovo",
) -> list[Read]:
    """Parse a Casanovo mzTab file into :class:`Read` objects.

    Predictions failing the precursor mass check (negative score in
    Casanovo < 5.2) are kept with their weight multiplied by
    ``mismatch_weight`` (0 drops them).
    """
    path = Path(path)
    header: list[str] | None = None
    reads: list[Read] = []
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            if line.startswith("PSH"):
                header = line.split("\t")[1:]
                seq_col = PROFORMA_COL if PROFORMA_COL in header else "sequence"
                score_col = _aa_score_col(header)
                continue
            if not line.startswith("PSM"):
                continue
            if header is None:
                raise ValueError(f"{path}:{line_no}: PSM line before PSH header")
            row = dict(zip(header, line.split("\t")[1:]))
            peptide = row.get(seq_col) or row.get("sequence", "")
            if not peptide or peptide == "null":
                continue
            residues = tokenize(peptide)
            if not residues:
                continue

            score = _float(row.get("search_engine_score[1]"), 0.0)
            flags = ""
            weight = 1.0
            if score < 0:
                score += 1.0
                flags = "precursor_mismatch"
                weight = mismatch_weight
                if weight <= 0:
                    continue

            conf = _parse_scores(row.get(score_col, "") if score_col else "")
            if reverse_scores:
                conf.reverse()
            if len(conf) != len(residues):
                conf = [max(score, 0.0)] * len(residues)
                flags = ",".join(f for f in (flags, "no_aa_scores") if f)
            conf = [min(1.0, max(0.0, c)) for c in conf]

            spectrum = row.get("spectra_ref", "") or row.get("PSM_ID", "")
            scan = row.get(SCAN_COL, "")
            if scan and scan != "null":
                spectrum = f"{spectrum}|{scan}"
            reads.append(
                Read(
                    id=f"{source}:{path.stem}:{row.get('PSM_ID', line_no)}",
                    source=source,
                    spectrum=f"{path.stem}:{spectrum}",
                    seq=alphabet.encode_residues(residues),
                    conf=conf,
                    score=max(0.0, min(1.0, score)),
                    weight=weight,
                    flags=flags,
                )
            )
    if header is None:
        raise ValueError(f"{path}: no PSH header found - is this a Casanovo mzTab file?")
    return reads


def _float(x: str | None, default: float) -> float:
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else v


def _parse_scores(s: str) -> list[float]:
    s = s.strip()
    if not s or s == "null":
        return []
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        return []

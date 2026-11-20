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
  The score is shifted back and the read flagged; peas additionally computes
  the precursor ppm error itself from ``exp_mass_to_charge`` and ``charge``
  (works for all versions, incl. v5.2+ which no longer penalises).
* ``spectra_ref`` is ``ms_run[N]:index=K`` (0-based MGF index) or
  ``ms_run[N]:scan=K``; run names come from the ``ms_run[N]-location`` MTD lines.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ..mass import precursor_ppm
from ..model import Read
from ..tokens import Alphabet, tokenize

_MSRUN_RE = re.compile(r"ms_run\[(\d+)\]")
_INDEX_RE = re.compile(r"index=(\d+)")
_SCAN_RE = re.compile(r"scan=(\d+)")
_TRAIL_RE = re.compile(r"(\d+)\s*$")

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
    source: str = "casanovo",
) -> list[Read]:
    """Parse a Casanovo mzTab file into :class:`Read` objects.

    Precursor-mismatch handling (weighting) happens in preprocessing; here
    the read gets its computed ``ppm`` error and, for Casanovo < 5.2
    negative scores, the ``precursor_mismatch`` flag.
    """
    path = Path(path)
    header: list[str] | None = None
    reads: list[Read] = []
    runs: dict[str, str] = {}
    with open(path) as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\r\n")
            if line.startswith("MTD"):
                parts = line.split("\t")
                if len(parts) >= 3 and parts[1].endswith("-location"):
                    m = _MSRUN_RE.search(parts[1])
                    if m:
                        loc = parts[2].replace("\\", "/").rstrip("/").split("/")[-1]
                        runs[m.group(1)] = loc.rsplit(".", 1)[0] if "." in loc else loc
                continue
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
            if score < 0:
                score += 1.0
                flags = "precursor_mismatch"

            conf = _parse_scores(row.get(score_col, "") if score_col else "")
            if reverse_scores:
                conf.reverse()
            if len(conf) != len(residues):
                conf = [max(score, 0.0)] * len(residues)
                flags = ",".join(f for f in (flags, "no_aa_scores") if f)
            conf = [min(1.0, max(0.0, c)) for c in conf]

            ref = row.get("spectra_ref", "") or ""
            spectrum = ref or row.get("PSM_ID", "")
            scan_field = row.get(SCAN_COL, "")
            if scan_field and scan_field != "null":
                spectrum = f"{spectrum}|{scan_field}"
            m = _MSRUN_RE.search(ref)
            run = runs.get(m.group(1), f"{path.stem}#{m.group(1)}") if m else path.stem
            index = _int_match(_INDEX_RE, ref)
            scan = _int_match(_SCAN_RE, scan_field) or _int_match(_SCAN_RE, ref)
            alt = _int_match(_TRAIL_RE, ref) if index is None and scan is None else None
            charge = int(_float(row.get("charge"), 0.0))
            ppm = precursor_ppm(residues, _float(row.get("exp_mass_to_charge"), 0.0), charge)
            reads.append(
                Read(
                    id=f"{source}:{path.stem}:{row.get('PSM_ID', line_no)}",
                    source=source,
                    spectrum=f"{path.stem}:{spectrum}",
                    seq=alphabet.encode_residues(residues),
                    conf=conf,
                    score=max(0.0, min(1.0, score)),
                    flags=flags,
                    run=run,
                    index=index,
                    scan=scan,
                    alt_id=alt,
                    ppm=ppm,
                )
            )
    if header is None:
        raise ValueError(f"{path}: no PSH header found - is this a Casanovo mzTab file?")
    return reads


def _int_match(rx: re.Pattern, s: str | None) -> int | None:
    if not s:
        return None
    m = rx.search(s)
    return int(m.group(1)) if m else None


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

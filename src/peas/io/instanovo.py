"""InstaNovo CSV parser (``instanovo predict`` output).

Columns used (InstaNovo v1.x defaults, older aliases accepted):

* ``predictions_tokenised`` - tokens joined by ``", "`` (N->C)
* ``token_log_probs``       - per-token log probabilities, a Python list
  repr aligned index-for-index with the tokens (InstaNovo decodes C->N but
  reverses both sequence and log probs before writing).
* ``log_probs``             - peptide log probability
* ``predictions_beam_{i}``, ``predictions_log_probability_beam_{i}``,
  ``predictions_token_log_probabilities_beam_{i}`` - extra beams written
  when ``num_beams > 1`` (beam strings are not separated).

Modifications appear as UNIMOD tags (``M[UNIMOD:35]``, ``[UNIMOD:1]-``).
An N-terminal modification is its own token; its probability is folded
into the first residue.
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

from ..mass import precursor_ppm
from ..model import Read
from ..tokens import Alphabet, Residue, tokenize

SEQ_COLS = ("predictions", "preds", "prediction")
TOK_COLS = ("predictions_tokenised", "preds_tokenised", "prediction_tokenised")
TOKLP_COLS = ("token_log_probs", "token_log_probabilities", "token_log_probs_list")
LP_COLS = ("log_probs", "log_probabilities", "log_prob")
ID_COLS = ("spectrum_id", "scan_number", "spectrum_index", "id", "index", "prediction_id")
_BEAM_RE = re.compile(r"^predictions_beam_(\d+)$")
_TRAIL_RE = re.compile(r"(\d+)\s*$")


def _int(s: str | None) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _first(columns: list[str], options: tuple[str, ...]) -> str | None:
    for o in options:
        if o in columns:
            return o
    return None


def _parse_list(s: str | None) -> list[float]:
    if s is None:
        return []
    s = s.strip().strip("[]()")
    if not s:
        return []
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        return []


def _float(s: str | None) -> float:
    try:
        v = float(s)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return v


def _residues_with_logprobs(
    tokens: list[str], logps: list[float]
) -> tuple[list[Residue], list[float]] | None:
    """Pair tokens with log probs, folding N-terminal mod tokens forward."""
    if len(tokens) != len(logps):
        return None
    residues: list[Residue] = []
    out: list[float] = []
    pending_nterm, pending_lp = "", 0.0
    for tok, lp in zip(tokens, logps):
        tok = tok.strip()
        if not tok:
            continue
        if not tok[0].isalpha():  # N-terminal modification token
            pending_nterm = tok.rstrip("-")
            pending_lp += lp
            continue
        parsed = tokenize(tok)
        if len(parsed) != 1:
            return None
        r = parsed[0]
        if pending_nterm:
            r = Residue(r.aa, r.mod, pending_nterm)
            lp += pending_lp
            pending_nterm, pending_lp = "", 0.0
        residues.append(r)
        out.append(lp)
    return residues, out


def _split_beam_tokens(peptide: str) -> list[str]:
    """Beam strings are concatenated tokens; recover the token list."""
    toks: list[str] = []
    for r in tokenize(peptide):
        if r.nterm:
            toks.append(r.nterm)
        toks.append(f"{r.aa}{r.mod}")
    return toks


def read_instanovo(
    path: str | Path,
    alphabet: Alphabet,
    *,
    use_beams: bool = True,
    source: str = "instanovo",
) -> list[Read]:
    """Parse an InstaNovo prediction CSV into :class:`Read` objects.

    With ``use_beams`` all beams of a spectrum are returned (rank 0 = best);
    their weights are later normalised per spectrum by
    :func:`peas.preprocess.weight_candidates`.
    """
    path = Path(path)
    reads: list[Read] = []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        seq_col = _first(cols, SEQ_COLS)
        tok_col = _first(cols, TOK_COLS)
        toklp_col = _first(cols, TOKLP_COLS)
        lp_col = _first(cols, LP_COLS)
        id_col = _first(cols, ID_COLS)
        if seq_col is None and tok_col is None:
            raise ValueError(f"{path}: no prediction column found (got {cols})")
        beams = sorted(int(m.group(1)) for c in cols if (m := _BEAM_RE.match(c)))

        for row_no, row in enumerate(rdr):
            spec = row.get(id_col, "") if id_col else ""
            spectrum = f"{path.stem}:{spec or row_no}"
            run = row.get("experiment_name") or path.stem
            index = _int(row.get("spectrum_index")) if "spectrum_index" in row else _int(row.get("index"))
            scan = _int(row.get("scan_number"))
            m = _TRAIL_RE.search(row.get("spectrum_id") or "")
            alt = int(m.group(1)) if m else None
            prec_mz = _float(row.get("precursor_mz"))
            charge = _int(row.get("precursor_charge")) or 0
            tool_ppm = _float(row.get("delta_mass_ppm"))
            cands: list[tuple[list[str], list[float], float]] = []
            if use_beams and beams:
                for b in beams:
                    pep = row.get(f"predictions_beam_{b}", "")
                    if not pep or pep.lower() == "nan":
                        continue
                    try:
                        toks = _split_beam_tokens(pep)
                    except ValueError:
                        continue
                    lps = _parse_list(row.get(f"predictions_token_log_probabilities_beam_{b}"))
                    lp = _float(row.get(f"predictions_log_probability_beam_{b}"))
                    cands.append((toks, lps, lp))
            if not cands:
                if tok_col and row.get(tok_col):
                    toks = [t for t in row[tok_col].split(",")]
                elif seq_col and row.get(seq_col):
                    try:
                        toks = _split_beam_tokens(row[seq_col])
                    except ValueError:
                        continue
                else:
                    continue
                lps = _parse_list(row.get(toklp_col)) if toklp_col else []
                lp = _float(row.get(lp_col)) if lp_col else float("nan")
                cands.append((toks, lps, lp))

            for rank, (toks, lps, lp) in enumerate(cands):
                toks = [t.strip() for t in toks if t.strip()]
                if not toks:
                    continue
                flags = ""
                paired = _residues_with_logprobs(toks, lps) if lps else None
                if paired is None:
                    try:
                        residues = tokenize("".join(toks))
                    except ValueError:
                        continue
                    if not residues:
                        continue
                    per = math.exp(lp / len(residues)) if not math.isnan(lp) else 0.5
                    conf = [per] * len(residues)
                    flags = "no_token_scores"
                else:
                    residues, rlps = paired
                    if not residues:
                        continue
                    conf = [math.exp(min(0.0, x)) for x in rlps]
                    if math.isnan(lp):
                        lp = sum(rlps)
                score = math.exp(min(0.0, lp)) if not math.isnan(lp) else 0.0
                ppm = precursor_ppm(residues, prec_mz, charge) if not math.isnan(prec_mz) else None
                if ppm is None and rank == 0 and not math.isnan(tool_ppm):
                    ppm = tool_ppm
                reads.append(
                    Read(
                        id=f"{source}:{spectrum}:{rank}",
                        source=source,
                        spectrum=spectrum,
                        seq=alphabet.encode_residues(residues),
                        conf=conf,
                        score=score,
                        rank=rank,
                        flags=flags,
                        run=run,
                        index=index,
                        scan=scan,
                        alt_id=alt,
                        ppm=ppm,
                    )
                )
    return reads

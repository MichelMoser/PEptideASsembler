import math

import pytest

from peas.io import read_casanovo, read_instanovo
from peas.preprocess import PreprocessParams, collapse, preprocess, trim_mott, trim_simple, weight_candidates
from peas.tokens import Alphabet, Residue, tokenize


@pytest.mark.parametrize(
    "pep, expected",
    [
        ("PEPTIDE", list("PEPTIDE")),
        ("PEPM[Oxidation]K", ["P", "E", "P", "M[Oxidation]", "K"]),
        ("[Acetyl]-PEPK", ["[Acetyl]-P", "E", "P", "K"]),
        ("+42.011PEPM+15.995C+57.021K", ["+42.011-P", "E", "P", "M+15.995", "C+57.021", "K"]),
        ("+43.006-17.027QPK", ["+43.006-17.027-Q", "P", "K"]),
        ("[UNIMOD:1]-PEM[UNIMOD:35]K", ["[UNIMOD:1]-P", "E", "M[UNIMOD:35]", "K"]),
        ("PEM(ox)C(+57.02)K", ["P", "E", "M(ox)", "C(+57.02)", "K"]),
    ],
)
def test_tokenize(pep, expected):
    assert [str(r) for r in tokenize(pep)] == expected


def test_tokenize_rejects_garbage():
    with pytest.raises(ValueError):
        tokenize("PEP?K")


def test_alphabet_strip_and_merge():
    a = Alphabet()
    assert a.encode("PIM[Oxidation]K") == "PLMK"
    b = Alphabet(strip_mods=False, merge_il=False)
    enc = b.encode("PIM[Oxidation]KM[Oxidation]")
    assert len(enc) == 5 and enc[2] == enc[4] and enc[2] != "M"
    assert b.decode(enc) == "PIM[Oxidation]KM[Oxidation]"
    assert b.plain(enc) == "PIMKM"


CASANOVO_V5 = """MTD\tmzTab-version\t1.0.0
PSH\tsequence\tPSM_ID\taccession\tunique\tdatabase\tdatabase_version\tsearch_engine\tsearch_engine_score[1]\tmodifications\tretention_time\tcharge\texp_mass_to_charge\tcalc_mass_to_charge\tspectra_ref\tpre\tpost\tstart\tend\topt_global_aa_scores\topt_global_cv_MS:1003169_proforma_peptidoform_sequence
PSM\tPEPMK\t1\tnull\tnull\tnull\tnull\tx\t0.8\t4-Oxidation (M):UNIMOD:35\tnull\t2\t300.0\t300.0\tms_run[1]:index=0\tnull\tnull\tnull\tnull\t0.90000,0.80000,0.70000,0.60000,0.50000\tPEPM[Oxidation]K
PSM\tLLLLK\t2\tnull\tnull\tnull\tnull\tx\t-0.5\tnull\tnull\t2\t300.0\t300.0\tms_run[1]:index=1\tnull\tnull\tnull\tnull\t0.5,0.5,0.5,0.5,0.5\tLLLLK
"""

CASANOVO_V4 = """MTD\tmzTab-version\t1.0.0
PSH\tsequence\tPSM_ID\taccession\tunique\tdatabase\tdatabase_version\tsearch_engine\tsearch_engine_score[1]\tmodifications\tretention_time\tcharge\texp_mass_to_charge\tcalc_mass_to_charge\tspectra_ref\tpre\tpost\tstart\tend\topt_ms_run[1]_aa_scores
PSM\t+42.011PEPM+15.995K\t1\tnull\tnull\tnull\tnull\tx\t0.8\tnull\tnull\t2\t300.0\t300.0\tms_run[1]:index=0\tnull\tnull\tnull\tnull\t0.1,0.2,0.3,0.4,0.5
"""


def test_casanovo_v5(tmp_path):
    f = tmp_path / "c.mztab"
    f.write_text(CASANOVO_V5)
    reads = read_casanovo(f, Alphabet())
    assert len(reads) == 2
    r = reads[0]
    assert r.seq == "PEPMK" and r.conf == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert reads[1].flags == "precursor_mismatch"
    assert math.isclose(reads[1].score, 0.5)
    assert reads[0].ppm is not None and reads[0].ppm < -10000  # m/z 300.0 is far off
    kept, st = preprocess(reads, PreprocessParams(min_len=4, min_mean_conf=0.0, precursor_mismatch_weight=0))
    assert len(kept) == 0 and st["precursor_mismatch"] == 2


def test_casanovo_v4_and_reverse(tmp_path):
    f = tmp_path / "c4.mztab"
    f.write_text(CASANOVO_V4)
    r = read_casanovo(f, Alphabet(strip_mods=False))[0]
    assert len(r.seq) == 5 and r.conf[0] == 0.1
    r = read_casanovo(f, Alphabet(), reverse_scores=True)[0]
    assert r.seq == "PEPMK" and r.conf[0] == 0.5


INSTANOVO = (
    "spectrum_id,scan_number,predictions,predictions_tokenised,log_probs,token_log_probs\n"
    'a:1,1,[UNIMOD:1]-PEM[UNIMOD:35]K,"[UNIMOD:1], P, E, M[UNIMOD:35], K",-0.4,"[-0.1, -0.1, -0.1, -0.05, -0.05]"\n'
    'a:2,2,PEPTIDEK,"P, E, P, T, I, D, E, K",-1.0,"[-0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.2]"\n'
    "a:3,3,,,,\n"
)

INSTANOVO_BEAMS = (
    "spectrum_id,predictions,predictions_tokenised,log_probs,token_log_probs,"
    "predictions_beam_0,predictions_log_probability_beam_0,predictions_token_log_probabilities_beam_0,"
    "predictions_beam_1,predictions_log_probability_beam_1,predictions_token_log_probabilities_beam_1\n"
    'x:1,PEPK,"P, E, P, K",-0.4,"[-0.1, -0.1, -0.1, -0.1]",PEPK,-0.4,"[-0.1, -0.1, -0.1, -0.1]",'
    'PEQK,-1.0,"[-0.1, -0.1, -0.7, -0.1]"\n'
)


def test_instanovo(tmp_path):
    f = tmp_path / "i.csv"
    f.write_text(INSTANOVO)
    reads = read_instanovo(f, Alphabet())
    assert [r.seq for r in reads] == ["PEMK", "PEPTLDEK"]
    r = reads[0]
    # N-terminal mod log prob folded into the first residue
    assert math.isclose(r.conf[0], math.exp(-0.2))
    assert math.isclose(r.score, math.exp(-0.4))


def test_instanovo_beams_weighting(tmp_path):
    f = tmp_path / "b.csv"
    f.write_text(INSTANOVO_BEAMS)
    reads = read_instanovo(f, Alphabet())
    assert [r.seq for r in reads] == ["PEPK", "PEQK"]
    weight_candidates(reads)
    assert math.isclose(sum(r.weight for r in reads), 1.0)
    assert reads[0].weight > reads[1].weight
    assert len(read_instanovo(f, Alphabet(), use_beams=False)) == 1


def test_preprocess_trim_and_collapse(tmp_path):
    f = tmp_path / "i.csv"
    f.write_text(INSTANOVO)
    reads = read_instanovo(f, Alphabet())
    reads[1].conf[0] = 0.1  # low-confidence N-terminus is trimmed
    kept, stats = preprocess(reads, PreprocessParams(min_len=4))
    assert stats["trimmed"] == 1
    assert [r.seq for r in kept] == ["PEMK", "EPTLDEK"]
    units = collapse(kept + [kept[0]])
    assert units[0].seq == "EPTLDEK"
    assert units[1].weight == 2.0


def test_trim_simple_vs_mott():
    conf = [0.1, 0.5, 0.05, 0.1, 0.9, 0.95, 0.9, 0.95, 0.2]
    assert trim_simple(conf, 0.3) == (1, 8)
    # mott: removed deficits 0.2-0.2+0.25+0.2 = 0.45 -> cut 4 on the left
    assert trim_mott(conf, 0.3) == (4, 8)
    # a confident run outweighs one weak residue
    assert trim_mott([0.25, 0.9, 0.9, 0.9], 0.3) == (1, 4)
    assert trim_mott([0.9] * 5, 0.3) == (0, 5)
    assert trim_mott([0.1] * 5, 0.3) == (5, 5)


def test_per_source_trim_threshold(tmp_path):
    f = tmp_path / "i.csv"
    f.write_text(INSTANOVO)
    reads = read_instanovo(f, Alphabet())
    reads[1].conf[-1] = 0.5
    kept, _ = preprocess(reads, PreprocessParams(min_len=4, trim_conf_by_source={"instanovo": 0.6}))
    assert kept[1].seq == "PEPTLDE"


def test_masses():
    from peas.mass import mod_mass, peptide_mass, precursor_ppm

    assert math.isclose(mod_mass("[Oxidation]"), 15.994915)
    assert math.isclose(mod_mass("+43.006-17.027"), 25.979)
    assert math.isclose(mod_mass("(+57.02)"), 57.02)
    assert mod_mass("[Foo]") is None
    pep = tokenize("PEPTIDEK")
    m = peptide_mass(pep)
    assert math.isclose(m, 927.4549, abs_tol=1e-3)
    mz = (m + 2 * 1.007276) / 2
    assert abs(precursor_ppm(pep, mz, 2)) < 1e-6
    # +1 isotope picked: still ~0 ppm
    assert abs(precursor_ppm(pep, mz + 1.003355 / 2, 2)) < 1e-6
    # K -> Q swap: ~0.036 Da off
    assert 30 < abs(precursor_ppm(tokenize("PEPTIDEQ"), mz, 2)) < 50
    assert precursor_ppm(tokenize("PEPTLDEK"), 0.0, 2) is None


def test_agreement_pairs_and_updates():
    from peas.agreement import AgreementParams, apply_agreement
    from peas.model import Read

    seqs = ["PEPTLDEK", "AAGGLLK", "WWPPEEK", "MKLLPTR", "QWERTYK"] * 4
    ca, ins = [], []
    for i, s in enumerate(seqs):
        ca.append(Read(f"c{i}", "casanovo", f"c:{i}", s, [0.6] * len(s), 0.5, run="r1", index=i))
        t = s if i % 5 else s[:-1] + "R"  # every 5th disagrees at the C-terminus
        ins.append(Read(f"i{i}", "instanovo", f"i:{i}", t, [0.6] * len(t), 0.5, run="r1x", scan=i + 1))
    st = apply_agreement(ca + ins, AgreementParams(min_identical=5))
    assert st["paired_spectra"] == len(seqs)
    assert "[scan-1]" in st["pairings"][0]
    assert ca[1].conf[0] > 0.6 and ca[1].spectrum == ins[1].spectrum
    assert ca[0].conf[-1] < 0.6 and ca[0].conf[0] > 0.6

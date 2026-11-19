import random

import pytest

from peas.assemble import AssemblyParams, assemble, write_outputs
from peas.dbg import DBGParams, assemble_dbg
from peas.evaluate import evaluate
from peas.io import read_casanovo, read_fasta, read_instanovo
from peas.model import Read, Unit
from peas.olc import OLCParams, assemble_olc
from peas.simulate import SimParams, random_protein, simulate
from peas.tokens import Alphabet

PROTEIN = random_protein(random.Random(42), 120).replace("I", "L")


def tiled_units(seq, length=14, step=4, conf=0.95):
    units = []
    for s in range(0, len(seq) - length + 1, step):
        units.append(Unit(seq[s : s + length], [conf] * length, 1.0, [f"r{s}"]))
    tail = seq[-length:]
    units.append(Unit(tail, [conf] * length, 1.0, ["tail"]))
    units.sort(key=lambda u: (-len(u.seq), -u.weight, u.seq))
    return units


def test_dbg_reconstructs_perfect_tiling():
    contigs, g, stats = assemble_dbg(tiled_units(PROTEIN), DBGParams(k=5, min_kmer_weight=0.1, min_edge_weight=0.1))
    assert max(contigs, key=lambda c: len(c[0]))[0] == PROTEIN


def test_olc_reconstructs_perfect_tiling():
    contigs, stats = assemble_olc(tiled_units(PROTEIN), OLCParams())
    assert [c[0] for c in contigs] == [PROTEIN]


def test_olc_tolerates_uncertain_mismatch_but_not_confident():
    u = tiled_units(PROTEIN)
    # introduce a low-confidence error in one read: still assembles
    i = 3
    seq = list(u[i].seq)
    seq[6] = "W" if seq[6] != "W" else "A"
    conf = list(u[i].conf)
    conf[6] = 0.2
    u[i] = Unit("".join(seq), conf, 1.0, ["err"])
    contigs, _ = assemble_olc(u, OLCParams())
    assert any(c[0] == PROTEIN for c in contigs)


def test_dbg_bubble_popped():
    units = tiled_units(PROTEIN, conf=0.95)
    # weak erroneous variant of a read -> bubble / tip, should not break contig
    s = list(units[2].seq)
    s[7] = "W" if s[7] != "W" else "A"
    units.append(Unit("".join(s), [0.3] * len(s), 1.0, ["err"]))
    contigs, g, stats = assemble_dbg(units, DBGParams(k=5, min_kmer_weight=0.1, min_edge_weight=0.1))
    assert max(contigs, key=lambda c: len(c[0]))[0] == PROTEIN


@pytest.mark.parametrize("mode", ["dbg", "olc", "hybrid"])
def test_end_to_end_simulation(tmp_path, mode):
    simulate(tmp_path / "sim", SimParams(seed=5, protein_length=200))
    alpha = Alphabet()
    reads = read_casanovo(tmp_path / "sim/casanovo.mztab", alpha)
    reads += read_instanovo(tmp_path / "sim/instanovo.csv", alpha)
    params = AssemblyParams(mode=mode)
    res = assemble(reads, params)
    write_outputs(res, alpha, tmp_path / "out", params)
    for f in ("contigs.fasta", "contigs_all.fasta", "contigs.tsv", "positions.tsv", "placements.tsv", "run.json"):
        assert (tmp_path / "out" / f).exists()
    ev = evaluate(read_fasta(tmp_path / "out/contigs.fasta"), read_fasta(tmp_path / "sim/reference.fasta"))
    assert ev["reference_coverage"] >= 0.9
    assert ev["aligned_identity"] >= 0.95
    if mode != "olc":
        assert ev["longest"] >= 150


def test_read_length_conf_mismatch_raises():
    with pytest.raises(ValueError):
        Read("x", "s", "sp", "PEP", [1.0], 1.0)

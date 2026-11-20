# PEptideASsembler (`peas`)

Uncertainty-aware assembly of **de novo peptide predictions** from
[Casanovo](https://github.com/Noble-Lab/casanovo) and
[InstaNovo](https://github.com/instadeepai/InstaNovo) into protein-level
contigs — for example to reconstruct antibody or unknown protein sequences
from bottom-up MS/MS with several proteases.

The design borrows from short-read DNA assemblers:

| DNA assembly                          | peas                                                              |
|---------------------------------------|-------------------------------------------------------------------|
| read + Phred qualities                | peptide + per-residue confidence (aa score / token probability)   |
| quality trimming                      | trim low-confidence peptide termini                               |
| k-mer counting, solid k-mers          | **confidence-weighted** k-mer counting                            |
| de Bruijn graph, tips, bubbles        | weighted DBG, tip clipping, bubble popping (e.g. K/Q errors)      |
| overlap detection with hashed seeds   | k-mer seed index + ungapped, confidence-aware overlap scoring     |
| greedy / OLC layout + consensus       | containment removal, greedy best-overlap chaining, weighted vote  |
| read mapping + polishing (Pilon)      | remap all peptides, re-call residues, extend contig ends          |
| haplotig purging                      | purge noisy redundant contigs, keep well-supported homologues     |

Pure Python (≥3.10), no runtime dependencies. Python is the reference
implementation; hot loops are written so they port directly to Rust (see
the [roadmap](#roadmap-rust)).

## Install

```bash
uv venv && uv pip install -e '.[dev]'     # or: pip install -e '.[dev]'
```

## Quick start

```bash
# simulate a test protein with Casanovo- and InstaNovo-style outputs
peas simulate -o sim --length 300

# assemble (any number of files from either tool)
peas assemble --casanovo sim/casanovo.mztab --instanovo sim/instanovo.csv -o asm

# compare contigs with the true sequence(s)
peas evaluate --contigs asm/contigs.fasta --reference sim/reference.fasta
```

Real data:

```bash
casanovo sequence -o run1.mztab spectra.mgf
instanovo predict data_path=spectra.mgf output_path=run1.csv
peas assemble --casanovo run1.mztab --instanovo run1.csv -o asm
```

## Input formats

**Casanovo mzTab** (v4.x and v5.x)
- Peptide: the ProForma column `opt_global_cv_MS:1003169_proforma_peptidoform_sequence` (v5). If that column is absent, `sequence` (v4 mass-tag notation such as `M+15.995`).
- Residue confidence: `opt_global_aa_scores` (v5) or `opt_ms_run[1]_aa_scores` (v4).
- Casanovo < 4.3.0 wrote the aa scores in reverse order. Use `--casanovo-reverse-scores` for those files.
- Casanovo < 5.2 lowers the peptide score by 1 when the precursor m/z doesn't match. peas reads a negative score as a flagged precursor mismatch and multiplies that peptide's weight by `--precursor-mismatch-weight` (default 0.5; 0 drops it).

**InstaNovo CSV** (`instanovo predict`, v1.x)
- Peptide: `predictions_tokenised` (falls back to `predictions`).
- Residue confidence: `exp(token_log_probs)`.
- When `predictions_beam_*` columns are present (`num_beams > 1`), every beam is used as a candidate peptide (`--no-beams` keeps only the best). An N-terminal modification token is merged into the first residue.

Modifications are stripped for assembly by default (`--keep-mods` keeps them as distinct residues). I and L are merged (`--no-merge-il` keeps them apart).

## How uncertainty is used

1. **Per-residue confidence** `c ∈ [0,1]` is attached to every residue.
2. **Candidate weighting.** Each spectrum (per tool) contributes one unit of evidence, shared between its candidate peptides (beams / top-n PSMs) in proportion to their peptide probabilities. An ambiguous spectrum therefore counts less toward each interpretation. `--casanovo-weight` and `--instanovo-weight` scale whole tools.
3. **Trimming.** Peptide ends are trimmed with the modified-Mott algorithm, as BWA does for reads (`--trim-method mott`, default). From each end, the cut is placed where the summed deficit `threshold − confidence` of the removed residues is largest. A single confident residue among uncertain ones therefore does not stop trimming, while a confident run is kept. `--trim-method simple` cuts until the first residue above the threshold instead. The threshold is `--trim-conf` (default 0.3). Casanovo scores and InstaNovo probabilities are on different scales, so each tool can get its own threshold with `--casanovo-trim-conf` and `--instanovo-trim-conf`. Uncertain residues in the middle of a peptide are never cut; they are down-weighted instead (steps 4–6).
4. **Evidence.** Residue evidence = `c × weight`. Identical peptides are collapsed and their evidence summed.
5. **DBG.** A k-mer occurrence counts with the *minimum* evidence of its residues. Weak k-mers and edges are removed. Branches much weaker than their siblings are cut (`--relative-edge-ratio`).
6. **Overlaps.** A mismatch is tolerated if at least one of the two residues is uncertain. A mismatch between two residues that are both above `--confident-mismatch` rejects the overlap. Overlap scores are confidence weighted.
7. **Consensus.** Each contig column is a confidence-weighted vote. The reported per-position confidence is the mean over covering spectra of "confidence if the spectrum agrees with the call, else 0".

## Algorithm

```
parse → weight candidates → trim/filter → collapse identical peptides (units)
  ├─ dbg:    weighted de Bruijn graph (k = -k, can be several) → clean → greedy heaviest paths
  ├─ olc:    k-mer seed index → overlaps → containment → greedy layout → consensus
  └─ hybrid: DBG contigs + all units → OLC   (default)
→ polish: map units to contigs (k-mer seeds, best diagonal), re-call residues,
          extend ends with overhanging peptides, trim unsupported ends   (×rounds)
→ purge redundant contigs (banded identity; keep supported variants) → merge overlapping contigs (OLC) → polish
→ final mapping, per-position statistics, PASS filter
```

## Outputs (`-o DIR`)

| file                | content                                                                          |
|---------------------|----------------------------------------------------------------------------------|
| `contigs.fasta`     | PASS contigs (≥ `--pass-min-spectra`, depth, confidence)                         |
| `contigs_all.fasta` | all contigs, header carries `filter=PASS/LOW_SUPPORT`                            |
| `contigs.tsv`       | per contig: filter, length, mean/min confidence, depth, #reads, #spectra, ProForma |
| `positions.tsv`     | per residue: confidence, depth, best alternative residue and its share           |
| `placements.tsv`    | which peptide/spectrum supports which contig, where, identity                    |
| `graph_k*.gfa`      | cleaned de Bruijn graph unitigs (view in Bandage)                                |
| `run.json`          | parameters and per-stage statistics                                              |

## Simulated benchmark

`peas simulate` digests the reference **probabilistically** with each protease and writes one mzTab/CSV pair per digest, as each digest is normally its own LC-MS run. Every digest gets the same spectrum budget, spread over peptides by abundance × detectability.

| digest (`--enzymes`) | cleavage model | de novo error multiplier |
|---|---|---|
| `trypsin_lysc` | after K (also K-P, Lys-C) and R (not R-P) | 1.0 |
| `chymotrypsin` | after F/Y/W (high), L/M (low), not before P | 1.5 |
| `gluc` | after E, some after D | 1.5 |
| `proteinase_k` | broad: after aliphatic/aromatic residues, high background | 2.0 |
| `trypsin`, `lysc`, `aspn` | also available | |

The error multiplier reflects that Casanovo and InstaNovo are trained mostly on tryptic data. Non-tryptic digests also get more *confidently wrong* residues. The error model:
- I/L chosen at random
- near-isobaric substitutions (K/Q, N/D, F/M)
- adjacent residue swaps
- N ↔ GG
- higher error rates at the peptide termini
- garbage predictions

The test set is 4 proteins, 1,490 aa in total; two of them are ~87% identical homologues. Each digest has ~2,600 spectra, and both tools are used. All runs are in hybrid mode.

| digests | contigs | reference coverage | identity | N50 |
|---|---|---|---|---|
| Trypsin/Lys-C | 58 | 0.72 | 0.985 | 28 |
| proteinase K | 19 | 0.84 | 0.966 | 168 |
| Trypsin/Lys-C + chymotrypsin | 19 | 0.84 | 0.980 | 116 |
| + Glu-C | 19 | 0.83 | 0.965 | 194 |
| Trypsin/Lys-C + proteinase K | 25 | 0.85 | 0.955 | 450 |
| all four | 35 | 0.89 | 0.964 | 220 |

Main findings:
- **The assembler doesn't need to know the enzyme.** It only sees overlapping peptides.
- **Orthogonal digests create the overlaps across cleavage sites.** A single tryptic digest cannot be assembled into long contigs.
- **Proteinase K tiles the protein like shotgun reads.** Even on its own it gives long contigs, but it adds the most noise.
- **The two unique proteins are fully reconstructed** by any combination of two or more digests.
- **The loss of coverage comes from the homologous pair.**

## Known limitations / next steps

- **Ungapped alignment.** Mass-equivalent length changes (N↔GG, Q↔AG) and other indels only partly survive. Next step: a spectrum-aware banded alignment where residue blocks of equal mass match.
- **Mass information is not yet used.** Precursor m/z and fragment-ion evidence could validate ambiguous columns.
- **Pairwise overlaps can grow quadratically** for very abundant peptides. They are capped by masking frequent k-mers (`max_kmer_occ`), but large datasets need the Rust core.
- **No template-guided mode yet.** Stitch-style assembly against germline antibody templates is not supported.

## Roadmap: Rust

The Python code is the reference implementation. Its data model is deliberately simple:
- sequences are `str` with **one character per residue** (modified residues use private-use code points), which maps 1:1 to `Vec<u8>`;
- confidences are flat float lists;
- k-mers are slices.

Planned port as a PyO3/maturin extension `peas-core`, keeping the Python CLI:
1. k-mer counting and the weighted DBG (rolling 5-bit residue hashing, `u64` k-mers up to k = 12)
2. seed index and overlap detection (`rayon`-parallel over units)
3. unit-to-contig mapping and profile accumulation
4. later, a native binary once the algorithms settle

The test suite and `peas simulate`/`peas evaluate` serve as the regression benchmark for the port.

## Related work

- ALPS (Tran et al., *Sci Rep* 2016): weighted DBG assembly of PEAKS de novo peptides
- Stitch (Schulte et al., 2021): template-based antibody assembly
- Supernovo, PASS, and short-read assemblers: Velvet, SPAdes, SSAKE


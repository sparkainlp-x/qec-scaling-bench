# qec-scaling-bench: a classical toy simulation of surface-code memory scaling

[![tests](https://github.com/sparkainlp-x/qec-scaling-bench/actions/workflows/tests.yml/badge.svg)](https://github.com/sparkainlp-x/qec-scaling-bench/actions/workflows/tests.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue.svg)](.github/workflows/tests.yml)
[![Status: classical toy simulation](https://img.shields.io/badge/status-classical%20toy%20simulation-orange.svg)](#what-this-is-not)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23241631.svg)](https://doi.org/10.5281/zenodo.23241631)

**This is a classical toy simulation.** Every number in this repository was produced on an ordinary classical CPU by the [Stim](https://github.com/quantumlib/Stim) stabilizer-circuit simulator and the [PyMatching](https://github.com/oscarhiggott/PyMatching) decoder, under an idealized, hand-chosen noise model. No quantum computer or quantum hardware was used, nothing was measured, and nothing here demonstrates quantum error correction, fault tolerance or any hardware capability. Every reported result is classical simulation output only.

## What it computes

A seeded Monte Carlo study of how the simulated logical-memory failure probability of two textbook codes changes with code distance under a circuit-level Pauli noise model:

- **Codes:** Stim's built-in rotated surface-code Z-memory circuit (`surface_code:rotated_memory_z`) and repetition-code memory circuit (`repetition_code:memory`), at distances **d = 3, 5, 7**, each with **R = d, 2d, 4d** repeated syndrome rounds.
- **Decoder:** PyMatching minimum-weight perfect matching on the detector error model Stim derives from each circuit (`decompose_errors=True`). This is a standard baseline, not an optimal decoder.
- **Noise model (independent stochastic Pauli/readout faults):** depolarization after every two-qubit/Clifford gate (`gate`), measurement-result flips (`readout`), flips after reset (`reset`, fixed 0.001) and data-qubit depolarization before each round (`data`, fixed 0.001). Gate and readout are swept independently over six development points: `(0.001, 0.001)`, `(0.001, 0.005)`, `(0.003, 0.003)` (the **"center"** point), `(0.005, 0.001)`, `(0.005, 0.005)` and `(0.01, 0.01)`.
- **Held-out check:** one off-grid profile (gate 0.004, readout 0.002) run with two other seeds (314159, 271828), not used to choose anything.
- **Reference rows:** an unencoded one-qubit memory (reset flip, per-round data depolarization and readout flip; final readout used without correction) and zero-noise controls for all three families.
- **Shots:** 20,000 per condition (243 conditions; about 15 s on one CPU core).

For each condition the report gives the memory-failure count, a two-sided 95% **Wilson** interval, and, for the encoded memories only, an **IID symmetric-flip-derived endpoint-probability estimate** `q` obtained by inverting `P_fail(R) = [1 − (1 − 2q)^R]/2`. `q` is a model-based conversion, not an observed per-round rate. For each code family, noise point and round multiplier it reports the ratio `q_d / q_(d+2)` with a 95% **independent two-sample percentile bootstrap** interval (2,000 resamples; each distance's binary outcomes are resampled separately, since different distances share no fault trajectories). When resamples have zero events the interval is reported as unresolved or one-sided instead of inventing a finite value; the fraction of such resamples is recorded.

The optional trials CSV contains per-shot outcomes (`0`/`1`) at the center point and `R = 4d` for all three families. Its `shot_index` is a row index only, not a shared trajectory.

## Run it

Python 3.10 or later. The committed results were produced with Python 3.13, Stim 1.16.0, PyMatching 2.4.0 and NumPy 2.5.3.

```bash
python3 -m pip install -e '.[test]'
python3 -m pytest -q

qec-scaling-bench \
  --shots 20000 \
  --seed 1823 \
  --bootstrap 2000 \
  --output results/scaling_sweep_seed_1823.json \
  --trials-csv results/center_noise_trials_seed_1823.csv

python3 tools/compare_reports.py results/scaling_sweep_seed_1823.json path/to/your/fresh.json
```

Without `--output` the JSON report is written to stdout (status messages go to stderr).

**Reproducibility.** Every circuit condition and every bootstrap contrast gets its own 64-bit seed, derived by SHA-256 from the master seed and a descriptive key (both mappings are stored in the JSON, with collision checking). Stim documents that a fixed seed reproduces samples exactly only with the same Stim version on the same machine (SIMD code path). Bootstrap intervals also depend on the NumPy version's random-stream implementation. On another machine or NumPy version you may therefore get statistically equivalent rather than bit-identical output; `tools/compare_reports.py` reports `IDENTICAL` or checks every count with a Bonferroni-corrected two-proportion z-test. CI re-runs the full sweep on Python 3.10–3.12: in the first CI run all Stim counts and the trials CSV were bit-identical on every version, the full report was identical on Python 3.12 (NumPy 2.5.3), and only bootstrap-interval endpoints differed on 3.10/3.11 (older NumPy). Timing fields are always excluded from comparisons. SHA-256 checksums of the committed results are in [`results/SHA256SUMS`](results/SHA256SUMS).

## Results (classical simulation, seed 1823)

Memory-failure counts out of 20,000 shots at `R = 4d`, and the surface-code ratio `q_d / q_(d+2)` with its 95% bootstrap interval:

| noise point (gate, readout) | surface d=3 / 5 / 7 | ratio d3→5 | ratio d5→7 | repetition d=3 / 5 / 7 | unencoded memory, R=12 / 20 / 28 |
|---|---|---|---|---|---|
| (0.001, 0.001) | 55 / 10 / 1 | 9.2 [5.0, 22.3] | 14 [3.3, unbounded] | 11 / 0 / 0 | 218 / 289 / 386 |
| (0.001, 0.005) | 88 / 18 / 4 | 8.2 [5.2, 14.9] | 6.3 [2.3, 33.6] | 6 / 0 / 0 | 272 / 356 / 513 |
| (0.003, 0.003) center | 310 / 164 / 56 | 3.17 [2.60, 3.84] | 4.12 [3.11, 5.74] | 30 / 0 / 0 | 227 / 392 / 414 |
| (0.005, 0.001) | 611 / 477 / 303 | 2.15 [1.91, 2.42] | 2.22 [1.92, 2.56] | 46 / 3 / 0 | 174 / 293 / 397 |
| (0.005, 0.005) | 702 / 595 / 382 | 1.97 [1.78, 2.21] | 2.20 [1.92, 2.52] | 69 / 4 / 2 | 287 / 318 / 496 |
| (0.01, 0.01) | 2229 / 3004 / 3503 | 1.17 [1.11, 1.25] | 1.16 [1.09, 1.23] | 192 / 23 / 10 | 386 / 470 / 579 |
| held-out (0.004, 0.002), seed 314159 | 464 / 295 / 139 | 2.64 [2.28, 3.07] | 2.99 [2.44, 3.68] | 28 / 1 / 0 | 246 / 313 / 396 |
| held-out (0.004, 0.002), seed 271828 | 482 / 314 / 151 | 2.58 [2.24, 2.97] | 2.93 [2.43, 3.59] | 34 / 2 / 0 | 213 / 302 / 386 |

How to read it, and what not to read into it:

- In this model, the estimated per-round `q` of the simulated surface code decreases with distance at every tested noise point; the ratio shrinks toward 1 as gate/readout noise grows (about 1.2 at 0.01). No crossing point was reached, so **no threshold value is estimated** here.
- At (0.01, 0.01) the raw counts rise with `d` because `R = 4d` is longer for larger `d`; the per-round estimate still falls slightly.
- Low-noise ratios rest on very few events (for example 1 failure at d=7) and have wide or one-sided intervals. Most repetition-code ratios are not estimable because both distances have zero or near-zero failures; more shots would be needed.
- The repetition code only protects against bit flips, so its lower failure counts are not a like-for-like comparison with the surface code.
- The unencoded row has no gate noise (it has no gates), so encoded-versus-unencoded differences are not a break-even comparison. At the center point the simulated d=3 surface code fails more often than the unencoded row.
- The 95% intervals are per comparison and are not corrected for the many comparisons in the table.

The full JSON also records circuit resources (the d = 3, 5, 7 rotated surface-code circuits use 17, 49 and 97 simulated circuit qubits; this is simulator bookkeeping, not hardware), decoder construction and decode wall-clock times on this CPU (software timing only, not decoder latency on any device), and a "not used" dense-state-vector memory estimate for context.

## What this is not

- **Not quantum hardware and not a hardware benchmark.** No device was used; no device connectivity, crosstalk, leakage, drift, calibration error, correlated noise or real-time decoding is modelled. The Stim-generated rotated planar check schedule is fixed.
- **Not a demonstration of quantum error correction or fault tolerance.** It reproduces well-known qualitative behaviour of a textbook circuit noise model in simulation.
- **Not comparable with experimental results.** Experimental surface-code memory studies (for example Google Quantum AI and collaborators, *Nature* 638, 920–926 (2025), [doi:10.1038/s41586-024-08449-y](https://doi.org/10.1038/s41586-024-08449-y)) measure real devices with different noise. This simulation is not tuned to, and must not be compared with, such results.
- **Not an optimal-decoder study.** PyMatching's graph-like decomposition handles correlated (hyperedge) faults approximately.

Related, separate repository: [`quantum-error-correction-demo`](https://github.com/sparkainlp-x/quantum-error-correction-demo) is a classical formula toy with no circuits or decoder; this repository adds an actual stabilizer-circuit simulation, still entirely classical.

## Repository layout

```
qec_scaling_bench.py          simulation, statistics and CLI (single module)
tests/                        pytest suite
tools/compare_reports.py      exact / statistical comparison against the committed report
results/                      committed seed-1823 JSON report, trials CSV and SHA256SUMS
```

## Citation

See [`CITATION.cff`](CITATION.cff). Archived on Zenodo: concept DOI [10.5281/zenodo.23241631](https://doi.org/10.5281/zenodo.23241631) (all versions); v0.1.0: [10.5281/zenodo.23241632](https://doi.org/10.5281/zenodo.23241632). Please also cite Stim (Gidney, *Quantum* 5, 497 (2021), [doi:10.22331/q-2021-07-06-497](https://doi.org/10.22331/q-2021-07-06-497)) and PyMatching (Higgott & Gidney, *Quantum* 9, 1600 (2025), [doi:10.22331/q-2025-01-20-1600](https://doi.org/10.22331/q-2025-01-20-1600)).

## License

Copyright (C) 2026 Jean-François Brisson / Spark AI NLP. Licensed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE). Commercial licensing: see [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md). Stim and PyMatching are separate projects under their own (Apache-2.0) licences and are not redistributed here.

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-07

First public version. Classical toy simulation only (Stim + PyMatching on a classical CPU); no hardware, no QEC demonstration.

### Added
- `qec_scaling_bench.py`: seeded Monte Carlo of Stim's rotated surface-code and repetition-code memory circuits (d = 3, 5, 7; R = d, 2d, 4d) decoded with PyMatching, an unencoded one-qubit reference and zero-noise controls; six-point gate/readout development sweep plus an off-grid held-out profile with two independent seeds; Wilson intervals, IID symmetric-flip-derived per-round estimate `q`, and independent two-sample bootstrap intervals for `q_d / q_(d+2)`.
- `tools/compare_reports.py`: exact comparison of a fresh report with the committed one (timings ignored), with a Bonferroni-corrected two-proportion z-test fallback for machines where Stim samples are not bit-identical.
- Committed results for seed 1823 (`results/`), with `SHA256SUMS`.
- AGPL-3.0-only `LICENSE`, `COMMERCIAL-LICENSE.md`, `NOTICE`, `CITATION.cff`, `.zenodo.json`, GitHub Actions CI (pytest on Python 3.10–3.12 plus a full-sweep reproduction check).

### Fixed (relative to the pre-release prototype)
- Bootstrap seeds were `seed XOR f(distance, multiplier, family)` and did not depend on the noise profile or scope, so every noise profile reused the same resampling stream. They are now SHA-256-derived per contrast, recorded in `experiment_design.bootstrap_seed_mapping` and in each contrast.
- The unencoded reference ignored the profile's declared reset-flip probability; it now applies it. Its lack of gate noise is documented: it is not a break-even comparison.
- With `--trials-csv` but no `--output`, a status line was printed into the JSON on stdout. Status lines now go to stderr.
- Bootstrap contrasts now record the fraction of undefined (0/0) and infinite resamples, the resample count and the seed.

### Changed
- Wording everywhere states that this is a classical toy simulation. Removed hardware result figures and an unverified description of a journal correction from the report and README; the hardware paper is cited only as background that must not be compared with.
- Licence changed from the prototype's MIT note (which had no LICENSE file) to AGPL-3.0-only with a commercial-licence option. Package renamed `qec-scaling-bench` (was `qec-scaling-lab`).
- Dropped a byte-identical duplicate of the trials CSV from the hand-off package.

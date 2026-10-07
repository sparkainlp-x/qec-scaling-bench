# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qec_scaling_bench import (
    _allocate_sampler_seed,
    circuit_resources,
    equivalent_per_cycle_rate,
    make_bare_physical_circuit,
    make_repetition_code_circuit,
    make_surface_code_circuit,
    normalize_noise_profile,
    per_cycle_metrics,
    run_benchmark,
    suppression_comparison,
    wilson_interval,
    write_trials_csv,
)


def profile(name="test", gate=0.003, readout=0.002, data=0.001, reset=0.001):
    return normalize_noise_profile({
        "name": name,
        "gate_depolarization": gate,
        "readout_flip_probability": readout,
        "data_depolarization": data,
        "reset_flip_probability": reset,
    })


def test_surface_code_physical_qubit_scaling_is_explicit():
    expected = {3: 17, 5: 49, 7: 97}
    for distance, count in expected.items():
        circuit = make_surface_code_circuit(distance, rounds=2 * distance, profile=profile())
        resources = circuit_resources(circuit, "surface_code", distance, 2 * distance)
        assert resources["active_circuit_qubits"] == count
        assert resources["data_qubits"] == distance**2
        assert resources["syndrome_qubits"] == distance**2 - 1
        assert circuit.num_detectors > 0
        assert circuit.num_observables == 1


def test_repetition_code_resource_counts_are_explicit():
    for distance in (3, 5, 7):
        circuit = make_repetition_code_circuit(distance, rounds=distance, profile=profile())
        resources = circuit_resources(circuit, "repetition_code", distance, distance)
        assert resources["active_circuit_qubits"] == 2 * distance - 1
        assert resources["data_qubits"] == distance
        assert resources["syndrome_qubits"] == distance - 1


def test_bare_physical_circuit_has_repeated_detectors_and_final_observable():
    circuit = make_bare_physical_circuit(rounds=5, profile=profile())
    assert circuit.num_qubits == 1
    assert circuit.num_detectors == 5
    assert circuit.num_observables == 1
    detections, observables = circuit.compile_detector_sampler(seed=41).sample(
        shots=12, separate_observables=True
    )
    assert detections.shape == (12, 5)
    assert observables.shape == (12, 1)


def test_per_cycle_transform_inverts_symmetric_flip_memory_model():
    per_cycle = 0.012
    rounds = 20
    memory_rate = (1 - (1 - 2 * per_cycle) ** rounds) / 2
    assert np.isclose(equivalent_per_cycle_rate(memory_rate, rounds), per_cycle)
    assert equivalent_per_cycle_rate(0.6, rounds) is None


def test_wilson_interval_is_bounded_and_nonzero_for_zero_observed_errors():
    lo, hi = wilson_interval(0, 100)
    assert lo == 0.0
    assert 0.0 < hi < 0.1
    assert wilson_interval(100, 100)[1] == 1.0


def test_suppression_comparison_uses_independent_two_sample_bootstrap():
    lower = np.zeros(1000, dtype=np.uint8)
    higher_aligned = np.zeros(1000, dtype=np.uint8)
    higher_disjoint = np.zeros(1000, dtype=np.uint8)
    lower[:40] = 1
    higher_aligned[:20] = 1
    higher_disjoint[500:520] = 1
    result = suppression_comparison(lower, 12, higher_aligned, 20, seed=9, bootstrap=500)
    reordered = suppression_comparison(lower, 12, higher_disjoint, 20, seed=9, bootstrap=500)
    assert result["low_distance_shots"] == 1000
    assert result["high_distance_shots"] == 1000
    assert result["bootstrap_method"].startswith("independent two-sample percentile bootstrap")
    assert result["ci95_independent_two_sample_percentile_bootstrap"] == reordered["ci95_independent_two_sample_percentile_bootstrap"]
    assert result["low_distance_per_cycle_rate"] > result["high_distance_per_cycle_rate"]
    assert result["suppression_factor_low_rate_divided_by_high_rate"] > 1
    assert result["ci95_independent_two_sample_percentile_bootstrap"][0] > 0
    unequal_higher = np.zeros(800, dtype=np.uint8)
    unequal_higher[:16] = 1
    unequal = suppression_comparison(lower, 12, unequal_higher, 20, seed=9, bootstrap=500)
    assert unequal["low_distance_shots"] == 1000
    assert unequal["high_distance_shots"] == 800


def test_trials_csv_uses_shot_index_not_pair_id(tmp_path):
    path = tmp_path / "trials.csv"
    write_trials_csv(path, {"low": np.array([0, 1]), "high": np.array([1, 0])}, seed=7)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ["shot_index", "master_seed", "low", "high"]
    assert rows[1] == ["0", "7", "0", "1"]


def test_per_cycle_q_is_labeled_as_iid_endpoint_derived_not_observed():
    metrics = per_cycle_metrics(np.array([0, 0, 1, 0], dtype=np.uint8), rounds=5)
    assert metrics["per_cycle_estimator_method"] == "IID symmetric-flip-derived endpoint-probability estimate"
    assert "not an observed per-cycle error rate" in metrics["per_cycle_estimator_assumption"]


def test_readme_matches_package_install_and_simulation_only_methods():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    assert "Python 3.10 or later" in readme
    assert "[project.optional-dependencies]" in pyproject
    assert "test = [\"pytest>=7\"]" in pyproject
    assert "[project.scripts]" in pyproject
    assert "qec-scaling-bench = \"qec_scaling_bench:main\"" in pyproject
    assert "python3 -m pip install -e '.[test]'" in readme
    assert "python3 -m pytest -q" in readme
    assert "qec-scaling-bench \\" in readme
    assert "IID symmetric-flip-derived endpoint-probability estimate" in readme
    assert "independent two-sample percentile bootstrap" in readme
    assert "Every reported result is classical simulation output only." in readme
    assert "classical toy simulation" in readme.lower()
    assert 'license = "AGPL-3.0-only"' in pyproject


def test_public_copy_avoids_hardware_and_achievement_language():
    root = Path(__file__).parents[1]
    forbidden = ("QPU", "quantum advantage", "holographic", "on real quantum hardware we",
                 "achieved quantum error correction", "94.92")
    for name in ("README.md", "CITATION.cff", ".zenodo.json", "pyproject.toml", "qec_scaling_bench.py"):
        text = (root / name).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase.lower() not in text.lower(), f"{phrase!r} found in {name}"


def _scientific_counts(report):
    signatures = []
    for condition in report["development_sweep"]["conditions"]:
        for curve in condition["curves"]:
            for family, metric in curve["models"].items():
                signatures.append((
                    condition["noise_profile"]["name"],
                    curve["distance"], curve["rounds"], family,
                    metric["memory_failure_count"], metric["memory_failure_rate"],
                    metric["equivalent_per_cycle_error_rate"],
                    metric["ci95_wilson_memory_failure_rate"],
                ))
    return signatures


def _assert_sampler_seed_mapping(report):
    mapping = report["experiment_design"]["sampler_seed_mapping"]
    assert len(mapping) == len(set(mapping.values()))
    conditions = [*report["development_sweep"]["conditions"], report["ideal_zero_noise_control"]]
    conditions.extend(report["held_out_validation"]["conditions_by_seed"])
    observed_keys = set()
    for condition in conditions:
        for curve in condition["curves"]:
            for family, metric in curve["models"].items():
                key = metric["sampler_seed_key"]
                assert mapping[key] == metric["sampler_seed"]
                assert f"distance={curve['distance']}" in key
                assert f"rounds={curve['rounds']}" in key
                assert f"family={family}" in key
                observed_keys.add(key)
    assert observed_keys == set(mapping)


def test_benchmark_reproducibility_heldout_split_and_ideal_controls():
    center = {
        "name": "center_balanced",
        "gate_depolarization": 0.003,
        "readout_flip_probability": 0.003,
        "data_depolarization": 0.001,
        "reset_flip_probability": 0.001,
    }
    heldout = {
        "name": "heldout_asymmetric",
        "gate_depolarization": 0.004,
        "readout_flip_probability": 0.002,
        "data_depolarization": 0.001,
        "reset_flip_probability": 0.001,
    }
    kwargs = dict(
        shots=64,
        seed=1234,
        bootstrap=200,
        noise_grid=(center,),
        held_out_noise=heldout,
        held_out_seeds=(5678,),
        distances=(3,),
        round_multipliers=(1, 4),
    )
    first, first_trials = run_benchmark(**kwargs)
    second, second_trials = run_benchmark(**kwargs)
    assert _scientific_counts(first) == _scientific_counts(second)
    assert first["experiment_design"]["sampler_seed_mapping"] == second["experiment_design"]["sampler_seed_mapping"]
    _assert_sampler_seed_mapping(first)
    assert first_trials.keys() == second_trials.keys()
    for key in first_trials:
        assert len(first_trials[key]) == 64
        np.testing.assert_array_equal(first_trials[key], second_trials[key])
    assert first["evidence_tag"] == "SIMULATED_MODEL_DEPENDENT_MONTE_CARLO"
    assert first["scope"]["quantum_hardware_used"] is False
    assert first["scope"]["ghz_or_graph_connectivity_test_included"] is False
    assert first["held_out_validation"]["conditions_by_seed"][0]["seed"] == 5678
    assert first["held_out_validation"]["conditions_by_seed"][0]["noise_profile"]["name"] == "heldout_asymmetric"
    for curve in first["ideal_zero_noise_control"]["curves"]:
        for metric in curve["models"].values():
            assert metric["memory_failure_count"] == 0


def test_bare_physical_applies_declared_reset_noise_only_when_nonzero():
    noisy = make_bare_physical_circuit(rounds=3, profile=profile(reset=0.01))
    assert "X_ERROR(0.01) 0" in str(noisy)
    clean = make_bare_physical_circuit(rounds=3, profile=profile(reset=0.0))
    assert "X_ERROR" not in str(clean)
    # A certain reset flip makes every raw final readout wrong (no readout/data noise).
    certain = make_bare_physical_circuit(rounds=2, profile=profile(gate=0.0, readout=0.0, data=0.0, reset=0.999999))
    _, observables = certain.compile_detector_sampler(seed=3).sample(shots=200, separate_observables=True)
    assert observables.mean() > 0.99


def test_bootstrap_seeds_are_distinct_per_profile_and_contrast():
    grid = (
        {"name": "a", "gate_depolarization": 0.003, "readout_flip_probability": 0.003},
        {"name": "b", "gate_depolarization": 0.005, "readout_flip_probability": 0.005},
    )
    report, _ = run_benchmark(
        shots=32, seed=11, bootstrap=100, noise_grid=grid, held_out_noise=None, held_out_seeds=(),
        distances=(3, 5), round_multipliers=(1,),
    )
    mapping = report["experiment_design"]["bootstrap_seed_mapping"]
    seeds = []
    for condition in [*report["development_sweep"]["conditions"], report["ideal_zero_noise_control"]]:
        for item in condition["distance_plus_2_suppression"]:
            assert mapping[item["bootstrap_seed_key"]] == item["bootstrap_seed"]
            assert 0 <= item["bootstrap_undefined_resample_fraction"] <= 1
            seeds.append(item["bootstrap_seed"])
    # 3 conditions (2 profiles + ideal control) x 2 families x 1 distance pair x 1 multiplier
    assert len(seeds) == 6
    assert len(set(seeds)) == len(seeds) == len(mapping)


def test_seed_allocation_is_deterministic_and_rejects_duplicate_keys():
    mapping: dict[str, int] = {}
    first = _allocate_sampler_seed(1, "k", mapping)
    assert _allocate_sampler_seed(1, "k", {}) == first
    try:
        _allocate_sampler_seed(1, "k", mapping)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("duplicate key accepted")


def test_cli_stdout_is_pure_json_even_with_trials_csv(tmp_path):
    code = (
        "import sys, qec_scaling_bench as q;"
        "orig = q.run_benchmark;"
        "q.run_benchmark = lambda **kw: orig(noise_grid=q.DEFAULT_NOISE_GRID[2:3], distances=(3,),"
        " held_out_seeds=(), held_out_noise=None, **kw);"
        "sys.exit(q.main(sys.argv[1:]))"
    )
    csv_path = tmp_path / "trials.csv"
    result = subprocess.run(
        [sys.executable, "-c", code, "--shots", "50", "--bootstrap", "100", "--trials-csv", str(csv_path)],
        capture_output=True, text=True, check=True, cwd=Path(__file__).parents[1],
    )
    report = json.loads(result.stdout)
    assert report["scope"]["classical_toy_simulation"] is True
    assert report["scope"]["quantum_hardware_used"] is False
    assert "Shot-indexed outcomes" in result.stderr
    assert csv_path.read_text(encoding="utf-8").startswith("shot_index,master_seed,d3_r12_bare_physical")

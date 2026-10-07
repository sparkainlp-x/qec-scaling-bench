#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP
"""Classical toy simulation of surface-code and repetition-code memory scaling.

Runs entirely on a classical computer with the Stim stabilizer-circuit simulator
and the PyMatching decoder. All outcomes are model-dependent Monte Carlo results
under an idealized Pauli/readout noise model: not quantum-hardware data, not a
hardware benchmark, and not a demonstration of quantum error correction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pymatching
import stim

DISTANCES = (3, 5, 7)
ROUND_MULTIPLIERS = (1, 2, 4)
DEFAULT_SHOTS = 20_000
DEFAULT_SEED = 1823
DEFAULT_BOOTSTRAP = 2_000
DEFAULT_NOISE_GRID = (
    {"name": "low_gate_low_readout", "gate_depolarization": 0.001, "readout_flip_probability": 0.001},
    {"name": "low_gate_high_readout", "gate_depolarization": 0.001, "readout_flip_probability": 0.005},
    {"name": "center_balanced", "gate_depolarization": 0.003, "readout_flip_probability": 0.003},
    {"name": "high_gate_low_readout", "gate_depolarization": 0.005, "readout_flip_probability": 0.001},
    {"name": "high_gate_high_readout", "gate_depolarization": 0.005, "readout_flip_probability": 0.005},
    {"name": "stress_balanced", "gate_depolarization": 0.01, "readout_flip_probability": 0.01},
)
DEFAULT_HELD_OUT_NOISE = {
    "name": "heldout_asymmetric_gate_readout",
    "gate_depolarization": 0.004,
    "readout_flip_probability": 0.002,
}
DEFAULT_HELD_OUT_SEEDS = (314159, 271828)
DEFAULT_FIXED_DATA_NOISE = 0.001
DEFAULT_FIXED_RESET_NOISE = 0.001
_PRIMARY_NOISE_NAME = "center_balanced"
_LONGEST_ROUND_MULTIPLIER = 4
_Z_975 = statistics.NormalDist().inv_cdf(0.975)
_FAMILIES = ("bare_physical", "repetition_code", "surface_code")
__version__ = "0.1.0"


def normalize_noise_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and expand independently specified gate/readout/data/reset rates."""
    name = profile.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("each noise profile needs a non-empty name")
    normalized: dict[str, Any] = {"name": name}
    defaults = {
        "gate_depolarization": 0.0,
        "readout_flip_probability": 0.0,
        "data_depolarization": DEFAULT_FIXED_DATA_NOISE,
        "reset_flip_probability": DEFAULT_FIXED_RESET_NOISE,
    }
    for key, default in defaults.items():
        value = profile.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError(f"{key} must be finite and in [0, 1)")
        normalized[key] = float(value)
    return normalized


def _validate_settings(
    shots: int,
    seed: int,
    bootstrap: int,
    distances: Sequence[int],
    round_multipliers: Sequence[int],
) -> None:
    if isinstance(shots, bool) or not isinstance(shots, int) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an integer in [0, 2**64)")
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 100:
        raise ValueError("bootstrap must be an integer >= 100")
    if not distances or any(d not in DISTANCES for d in distances):
        raise ValueError(f"distances must be a non-empty subset of {DISTANCES}")
    if len(set(distances)) != len(distances):
        raise ValueError("distances must not contain duplicates")
    if not round_multipliers or any(isinstance(m, bool) or not isinstance(m, int) or m < 1 for m in round_multipliers):
        raise ValueError("round_multipliers must be positive integers")
    if len(set(round_multipliers)) != len(round_multipliers):
        raise ValueError("round_multipliers must not contain duplicates")


def _stim_noise_parameters(profile: Mapping[str, Any]) -> dict[str, float]:
    return {
        "after_clifford_depolarization": float(profile["gate_depolarization"]),
        "before_measure_flip_probability": float(profile["readout_flip_probability"]),
        "after_reset_flip_probability": float(profile["reset_flip_probability"]),
        "before_round_data_depolarization": float(profile["data_depolarization"]),
    }


def make_surface_code_circuit(distance: int, rounds: int, profile: Mapping[str, Any]) -> stim.Circuit:
    """Create a rotated surface-code Z-memory circuit with an explicit round count."""
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        **_stim_noise_parameters(profile),
    )


def make_repetition_code_circuit(distance: int, rounds: int, profile: Mapping[str, Any]) -> stim.Circuit:
    """Create Stim's repeated-measurement repetition-code memory circuit."""
    if distance not in DISTANCES:
        raise ValueError(f"distance must be one of {DISTANCES}")
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    return stim.Circuit.generated(
        "repetition_code:memory",
        distance=distance,
        rounds=rounds,
        **_stim_noise_parameters(profile),
    )


def make_bare_physical_circuit(rounds: int, profile: Mapping[str, Any]) -> stim.Circuit:
    """Create an uncorrected one-qubit Z-memory comparator.

    The qubit is reset (followed by the profile's reset-flip probability, as in
    the encoded circuits), then each round applies a single-qubit data
    depolarization opportunity followed by a noisy Z measurement. Consecutive
    measurement parities are detectors; the final physical measurement is the
    observable and is not decoded. No gate noise is applied because this
    comparator has no gates, so it is not a like-for-like break-even reference.
    """
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 1:
        raise ValueError("rounds must be a positive integer")
    circuit = stim.Circuit()
    circuit.append("R", [0])
    reset_flip = float(profile["reset_flip_probability"])
    if reset_flip > 0:
        circuit.append("X_ERROR", [0], reset_flip)
    for round_index in range(rounds):
        circuit.append("DEPOLARIZE1", [0], float(profile["data_depolarization"]))
        circuit.append("M", [0], float(profile["readout_flip_probability"]))
        if round_index == 0:
            circuit.append("DETECTOR", [stim.target_rec(-1)])
        else:
            circuit.append("DETECTOR", [stim.target_rec(-1), stim.target_rec(-2)])
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
    return circuit


def count_active_qubits(circuit: stim.Circuit) -> int:
    """Count distinct qubit IDs actually targeted, not the numeric ID span."""
    active: set[int] = set()
    for instruction in circuit.flattened():
        for target in instruction.targets_copy():
            if target.is_qubit_target:
                active.add(target.value)
    return len(active)


def circuit_resources(circuit: stim.Circuit, family: str, distance: int, rounds: int) -> dict[str, Any]:
    """Report implementation footprint separately from simulated performance."""
    qubits = count_active_qubits(circuit)
    if family == "surface_code":
        data_qubits, syndrome_qubits = distance * distance, distance * distance - 1
    elif family == "repetition_code":
        data_qubits, syndrome_qubits = distance, distance - 1
    elif family == "bare_physical":
        data_qubits, syndrome_qubits = 1, 0
    else:
        raise ValueError(f"unknown family: {family}")
    if data_qubits + syndrome_qubits != qubits:
        raise RuntimeError(
            f"resource-count mismatch for {family} d={distance}: "
            f"counted {qubits}, expected {data_qubits + syndrome_qubits}"
        )
    return {
        "active_circuit_qubits": qubits,
        "data_qubits": data_qubits,
        "syndrome_qubits": syndrome_qubits,
        "detectors": circuit.num_detectors,
        "observables": circuit.num_observables,
        "flattened_instruction_count_including_noise": sum(1 for _ in circuit.flattened()),
        "dense_statevector_complex128_memory_estimate_bytes_not_used": str(16 * (1 << qubits)),
    }


def wilson_interval(errors: int, shots: int, z: float = _Z_975) -> list[float]:
    """Two-sided Wilson score interval for a binomial probability."""
    if shots < 1 or not 0 <= errors <= shots:
        raise ValueError("require shots >= 1 and 0 <= errors <= shots")
    rate = errors / shots
    z2 = z * z
    denominator = 1 + z2 / shots
    center = (rate + z2 / (2 * shots)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / shots + z2 / (4 * shots * shots)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def equivalent_per_cycle_rate(memory_failure_rate: float, rounds: int) -> float | None:
    """Derive an IID symmetric-flip q estimate from an endpoint probability.

    Under independent, identically distributed symmetric logical flips,
    P(memory failure after R cycles) = (1 - (1 - 2q)^R) / 2. This is a model-
    derived endpoint-probability estimate, not an observed per-cycle rate or a
    fitted multi-duration cycle curve.
    """
    if rounds < 1 or not math.isfinite(memory_failure_rate) or not 0 <= memory_failure_rate <= 0.5:
        return None
    if memory_failure_rate == 0.5:
        return 0.5
    return -math.expm1(math.log1p(-2 * memory_failure_rate) / rounds) / 2


def per_cycle_metrics(errors: np.ndarray, rounds: int) -> dict[str, Any]:
    shots = int(len(errors))
    error_count = int(np.count_nonzero(errors))
    memory_rate = error_count / shots
    memory_ci = wilson_interval(error_count, shots)
    cycle_rate = equivalent_per_cycle_rate(memory_rate, rounds)
    cycle_ci = [equivalent_per_cycle_rate(memory_ci[0], rounds), equivalent_per_cycle_rate(min(0.5, memory_ci[1]), rounds)]
    return {
        "shots": shots,
        "memory_failure_count": error_count,
        "memory_failure_rate": memory_rate,
        "ci95_wilson_memory_failure_rate": memory_ci,
        "equivalent_per_cycle_error_rate": cycle_rate,
        "ci95_wilson_transformed_per_cycle_rate": cycle_ci,
        "per_cycle_estimator_method": "IID symmetric-flip-derived endpoint-probability estimate",
        "per_cycle_estimator_assumption": (
            "q is derived from the repeated-memory endpoint probability under an IID symmetric logical-flip model; "
            "it is not an observed per-cycle error rate"
        ),
    }


def _cycle_rate_ratio(low_rate: float, low_rounds: int, high_rate: float, high_rounds: int) -> float:
    low = equivalent_per_cycle_rate(low_rate, low_rounds)
    high = equivalent_per_cycle_rate(high_rate, high_rounds)
    if low is None or high is None or high == 0:
        return math.inf if low is not None and low > 0 and high == 0 else math.nan
    return low / high


def suppression_comparison(
    low_distance_errors: np.ndarray,
    low_rounds: int,
    high_distance_errors: np.ndarray,
    high_rounds: int,
    seed: int,
    bootstrap: int,
) -> dict[str, Any]:
    """Estimate q_d/q_(d+2) with an independent two-sample percentile bootstrap."""
    low_errors = np.asarray(low_distance_errors, dtype=np.int8).reshape(-1)
    high_errors = np.asarray(high_distance_errors, dtype=np.int8).reshape(-1)
    if low_errors.size == 0 or high_errors.size == 0:
        raise ValueError("suppression comparison requires two non-empty outcome samples")
    low_mem = float(low_errors.mean())
    high_mem = float(high_errors.mean())
    estimate = _cycle_rate_ratio(low_mem, low_rounds, high_mem, high_rounds)
    if math.isnan(estimate):
        estimate_json: float | None = None
        status = "per_cycle_rate_not_estimable"
    elif math.isinf(estimate):
        estimate_json = None
        status = "not_estimable_zero_observed_higher_distance_errors"
    else:
        estimate_json = estimate
        status = "estimated"

    # For binary outcomes, separate binomial resampling is equivalent to
    # independently resampling each distance-specific empirical sample.
    rng = np.random.default_rng(seed)
    low_draw_counts = rng.binomial(low_errors.size, low_mem, size=bootstrap)
    high_draw_counts = rng.binomial(high_errors.size, high_mem, size=bootstrap)
    ratios = np.empty(bootstrap, dtype=float)
    for index, (low_count, high_count) in enumerate(zip(low_draw_counts, high_draw_counts)):
        low_rate = float(low_count / low_errors.size)
        high_rate = float(high_count / high_errors.size)
        ratios[index] = _cycle_rate_ratio(low_rate, low_rounds, high_rate, high_rounds)
    undefined_fraction = float(np.isnan(ratios).mean())
    infinite_fraction = float(np.isinf(ratios).mean())
    if np.isnan(ratios).any():
        interval: list[float | None] = [None, None]
        if status == "estimated":
            status = "bootstrap_interval_unresolved_zero_event_resamples"
    else:
        ordered = sorted(float(value) for value in ratios)

        def nearest_rank(probability: float) -> float:
            rank = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
            return ordered[rank]

        q_lo, q_hi = nearest_rank(0.025), nearest_rank(0.975)
        interval = [float(q_lo) if math.isfinite(q_lo) else None, float(q_hi) if math.isfinite(q_hi) else None]
        if status == "estimated" and (math.isinf(q_lo) or math.isinf(q_hi)):
            status = "estimated_with_unbounded_bootstrap_upper_or_lower_limit"
    return {
        "distance_low": None,
        "distance_high": None,
        "low_distance_per_cycle_rate": equivalent_per_cycle_rate(low_mem, low_rounds),
        "high_distance_per_cycle_rate": equivalent_per_cycle_rate(high_mem, high_rounds),
        "suppression_factor_low_rate_divided_by_high_rate": estimate_json,
        "ci95_independent_two_sample_percentile_bootstrap": interval,
        "bootstrap_method": (
            "independent two-sample percentile bootstrap; each distance-specific binary sample is resampled separately"
        ),
        "bootstrap_resamples": int(bootstrap),
        "bootstrap_seed": int(seed),
        "bootstrap_undefined_resample_fraction": undefined_fraction,
        "bootstrap_infinite_resample_fraction": infinite_fraction,
        "low_distance_shots": int(low_errors.size),
        "high_distance_shots": int(high_errors.size),
        "status": status,
    }


def _allocate_sampler_seed(master_seed: int, condition_key: str, seed_mapping: dict[str, int]) -> int:
    """Derive a stable uint64 seed and retry deterministically if it collides.

    Used for both circuit-sampler seeds and bootstrap seeds (separate mappings).
    """
    if condition_key in seed_mapping:
        raise ValueError(f"duplicate seed condition key: {condition_key}")
    used_seeds = set(seed_mapping.values())
    collision_counter = 0
    while True:
        encoded = json.dumps(
            [master_seed, condition_key, collision_counter], separators=(",", ":")
        ).encode("utf-8")
        candidate = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
        if candidate not in used_seeds:
            seed_mapping[condition_key] = candidate
            return candidate
        collision_counter += 1


def _simulate(
    circuit: stim.Circuit,
    shots: int,
    seed: int,
    decoder: bool,
) -> tuple[np.ndarray, dict[str, float | None]]:
    times: dict[str, float | None] = {
        "sampler_compile_seconds": None,
        "detector_error_model_seconds": None,
        "decoder_construction_seconds": None,
        "sampling_seconds": None,
        "decoder_decode_seconds": None,
        "decoder_decode_seconds_per_shot": None,
    }
    start = time.perf_counter()
    sampler = circuit.compile_detector_sampler(seed=seed)
    times["sampler_compile_seconds"] = time.perf_counter() - start

    matching = None
    if decoder:
        start = time.perf_counter()
        detector_error_model = circuit.detector_error_model(decompose_errors=True)
        times["detector_error_model_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        matching = pymatching.Matching.from_detector_error_model(detector_error_model)
        times["decoder_construction_seconds"] = time.perf_counter() - start

    start = time.perf_counter()
    detections, actual_observables = sampler.sample(shots=shots, separate_observables=True)
    times["sampling_seconds"] = time.perf_counter() - start
    if matching is None:
        errors = np.any(actual_observables, axis=1)
    else:
        start = time.perf_counter()
        predicted = matching.decode_batch(detections)
        times["decoder_decode_seconds"] = time.perf_counter() - start
        times["decoder_decode_seconds_per_shot"] = float(times["decoder_decode_seconds"]) / shots
        errors = np.any(predicted != actual_observables, axis=1)
    return errors, times


def _model_metrics(
    circuit: stim.Circuit,
    family: str,
    distance: int,
    rounds: int,
    shots: int,
    errors: np.ndarray,
    timings: Mapping[str, float | None],
) -> dict[str, Any]:
    rate_label = {
        "bare_physical": "uncorrected final physical memory readout error",
        "repetition_code": "decoded repetition-code logical memory failure",
        "surface_code": "decoded surface-code logical memory failure",
    }[family]
    if family == "bare_physical":
        error_count = int(np.count_nonzero(errors))
        shots_count = int(len(errors))
        rate_stats = {
            "shots": shots_count,
            "memory_failure_count": error_count,
            "memory_failure_rate": error_count / shots_count,
            "ci95_wilson_memory_failure_rate": wilson_interval(error_count, shots_count),
            "equivalent_per_cycle_error_rate": None,
            "ci95_wilson_transformed_per_cycle_rate": [None, None],
            "per_cycle_estimator_method": "not estimated for the unencoded baseline",
            "per_cycle_estimator_assumption": (
                "not estimated for the unencoded baseline because final-readout noise is not separable from stored-state faults"
            ),
        }
    else:
        rate_stats = per_cycle_metrics(errors, rounds)
    return {
        "rate_semantics": rate_label,
        "distance": distance,
        "syndrome_rounds": rounds,
        **circuit_resources(circuit, family, distance, rounds),
        **rate_stats,
        "decoder": {
            "name": "PyMatching minimum-weight matching" if family != "bare_physical" else "none; raw final readout",
            "timing_seconds": dict(timings),
            "timing_interpretation": "local software wall-clock only; not hardware decoder latency",
        },
    }


def _condition(
    profile: Mapping[str, Any],
    seed: int,
    shots: int,
    bootstrap: int,
    distances: Sequence[int],
    round_multipliers: Sequence[int],
    collect_trial_vectors: bool,
    sampler_seed_mapping: dict[str, int] | None = None,
    seed_scope: str = "development",
    bootstrap_seed_mapping: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if sampler_seed_mapping is None:
        sampler_seed_mapping = {}
    if bootstrap_seed_mapping is None:
        bootstrap_seed_mapping = {}
    curves: list[dict[str, Any]] = []
    vector_map: dict[str, np.ndarray] = {}
    index: dict[tuple[int, int, str], tuple[np.ndarray, dict[str, Any]]] = {}
    for distance in distances:
        for multiplier in round_multipliers:
            rounds = distance * multiplier
            models: dict[str, Any] = {}
            for family in _FAMILIES:
                generation_start = time.perf_counter()
                if family == "surface_code":
                    circuit = make_surface_code_circuit(distance, rounds, profile)
                elif family == "repetition_code":
                    circuit = make_repetition_code_circuit(distance, rounds, profile)
                else:
                    circuit = make_bare_physical_circuit(rounds, profile)
                generation_seconds = time.perf_counter() - generation_start
                seed_key = (
                    f"{seed_scope}|profile={profile['name']}|master_seed={seed}|"
                    f"distance={distance}|rounds={rounds}|family={family}"
                )
                sampler_seed = _allocate_sampler_seed(seed, seed_key, sampler_seed_mapping)
                errors, timings = _simulate(circuit, shots, sampler_seed, decoder=family != "bare_physical")
                metric = _model_metrics(circuit, family, distance, rounds, shots, errors, timings)
                metric["circuit_generation_seconds"] = generation_seconds
                metric["sampler_seed"] = sampler_seed
                metric["sampler_seed_key"] = seed_key
                models[family] = metric
                index[(distance, multiplier, family)] = (errors, metric)
                if collect_trial_vectors and profile["name"] == _PRIMARY_NOISE_NAME and multiplier == _LONGEST_ROUND_MULTIPLIER:
                    key = f"d{distance}_r{rounds}_{family}"
                    vector_map[key] = errors
            curves.append({
                "distance": distance,
                "round_multiplier": multiplier,
                "rounds": rounds,
                "models": models,
            })

    suppression: list[dict[str, Any]] = []
    for multiplier in round_multipliers:
        for distance_low, distance_high in zip(distances, distances[1:]):
            if distance_high != distance_low + 2:
                continue
            low_rounds, high_rounds = distance_low * multiplier, distance_high * multiplier
            for family in ("repetition_code", "surface_code"):
                low_errors, _ = index[(distance_low, multiplier, family)]
                high_errors, _ = index[(distance_high, multiplier, family)]
                # Bootstrap seeds are SHA-256-derived per (scope, profile, family,
                # distance pair, multiplier) so no two contrasts in a run share a
                # resampling stream (earlier versions reused one stream per
                # distance/multiplier/family across all noise profiles).
                bootstrap_key = (
                    f"bootstrap|{seed_scope}|profile={profile['name']}|master_seed={seed}|"
                    f"distance_low={distance_low}|distance_high={distance_high}|"
                    f"round_multiplier={multiplier}|family={family}"
                )
                contrast_seed = _allocate_sampler_seed(seed, bootstrap_key, bootstrap_seed_mapping)
                comparison = suppression_comparison(
                    low_errors, low_rounds, high_errors, high_rounds, contrast_seed, bootstrap
                )
                comparison["bootstrap_seed_key"] = bootstrap_key
                comparison["distance_low"] = distance_low
                comparison["distance_high"] = distance_high
                comparison["round_multiplier"] = multiplier
                comparison["code_family"] = family
                suppression.append(comparison)

    return {
        "noise_profile": dict(profile),
        "seed": seed,
        "shots_per_curve_point": shots,
        "curves": curves,
        "distance_plus_2_suppression": suppression,
    }, vector_map


def run_benchmark(
    shots: int = DEFAULT_SHOTS,
    seed: int = DEFAULT_SEED,
    bootstrap: int = DEFAULT_BOOTSTRAP,
    noise_grid: Sequence[Mapping[str, Any]] = DEFAULT_NOISE_GRID,
    held_out_noise: Mapping[str, Any] | None = DEFAULT_HELD_OUT_NOISE,
    held_out_seeds: Sequence[int] = DEFAULT_HELD_OUT_SEEDS,
    distances: Sequence[int] = DISTANCES,
    round_multipliers: Sequence[int] = ROUND_MULTIPLIERS,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run exploratory noise grid and independent held-out seeds/noise profile."""
    _validate_settings(shots, seed, bootstrap, distances, round_multipliers)
    profiles = [normalize_noise_profile(item) for item in noise_grid]
    if not profiles:
        raise ValueError("noise_grid must not be empty")
    names = [p["name"] for p in profiles]
    if len(set(names)) != len(names):
        raise ValueError("noise profile names must be unique")
    heldout_profile = normalize_noise_profile(held_out_noise) if held_out_noise is not None else None
    if heldout_profile and heldout_profile["name"] in names:
        raise ValueError("held-out noise profile name must not appear in the development sweep")
    ideal_profile = normalize_noise_profile({
        "name": "ideal_zero_noise_control",
        "gate_depolarization": 0.0,
        "readout_flip_probability": 0.0,
        "data_depolarization": 0.0,
        "reset_flip_probability": 0.0,
    })
    for held_seed in held_out_seeds:
        if isinstance(held_seed, bool) or not isinstance(held_seed, int) or not 0 <= held_seed < 2**64:
            raise ValueError("held-out seeds must be integers in [0, 2**64)")
        if held_seed == seed:
            raise ValueError("held-out seeds must differ from the development seed")
    if len(set(held_out_seeds)) != len(held_out_seeds):
        raise ValueError("held-out seeds must not contain duplicates")

    development_conditions: list[dict[str, Any]] = []
    primary_vectors: dict[str, np.ndarray] = {}
    sampler_seed_mapping: dict[str, int] = {}
    bootstrap_seed_mapping: dict[str, int] = {}
    for profile in profiles:
        condition, vectors = _condition(
            profile, seed, shots, bootstrap, distances, round_multipliers,
            collect_trial_vectors=True,
            sampler_seed_mapping=sampler_seed_mapping,
            seed_scope="development",
            bootstrap_seed_mapping=bootstrap_seed_mapping,
        )
        development_conditions.append(condition)
        if profile["name"] == _PRIMARY_NOISE_NAME:
            primary_vectors = vectors

    ideal_control, _ = _condition(
        ideal_profile, seed, shots, bootstrap, distances, round_multipliers,
        collect_trial_vectors=False,
        sampler_seed_mapping=sampler_seed_mapping,
        seed_scope="ideal_zero_noise_control",
        bootstrap_seed_mapping=bootstrap_seed_mapping,
    )

    heldout_conditions: list[dict[str, Any]] = []
    if heldout_profile is not None:
        for held_seed in held_out_seeds:
            condition, _ = _condition(
                heldout_profile, held_seed, shots, bootstrap, distances, round_multipliers,
                collect_trial_vectors=False,
                sampler_seed_mapping=sampler_seed_mapping,
                seed_scope=f"held_out_seed_{held_seed}",
                bootstrap_seed_mapping=bootstrap_seed_mapping,
            )
            heldout_conditions.append(condition)

    report = {
        "evidence_tag": "SIMULATED_MODEL_DEPENDENT_MONTE_CARLO",
        "title": "Classical toy simulation of repeated-syndrome surface-code memory scaling",
        "software_version": __version__,
        "scope": {
            "classical_toy_simulation": True,
            "execution": "classical Stim stabilizer-circuit simulation plus classical PyMatching decoder, run on a classical CPU",
            "quantum_hardware_used": False,
            "hardware_measurement": False,
            "hardware_performance_claim": False,
            "fault_tolerance_claim": False,
            "quantum_error_correction_demonstrated": False,
            "ghz_or_graph_connectivity_test_included": False,
            "connectivity_test_boundary": "Any GHZ/graph-state experiment is a separate connectivity stress test and is excluded from logical-error-correction metrics.",
            "interpretation": (
                "All rates are conditional on the generated circuits, chosen decoder, finite shots, "
                "fixed seeds, and declared independent stochastic Pauli/readout model."
            ),
        },
        "axes": {
            "code_distance": list(distances),
            "syndrome_round_multipliers_of_distance": list(round_multipliers),
            "gate_noise": sorted({p["gate_depolarization"] for p in profiles}),
            "readout_noise": sorted({p["readout_flip_probability"] for p in profiles}),
            "reset_noise": "fixed separately per profile; see each noise_profile",
            "data_noise": "fixed separately per profile; see each noise_profile",
            "connectivity": {
                "model": "Stim-generated rotated planar surface-code local check schedule",
                "swept": False,
                "note": "No hardware layout constraints, long-range links, SWAP routing, or connectivity-dependent gate overhead are modeled.",
            },
            "decoder_cost": {
                "decoder": "PyMatching minimum-weight matching",
                "reported_separately": ["detector_error_model_seconds", "decoder_construction_seconds", "decoder_decode_seconds", "decoder_decode_seconds_per_shot"],
                "note": "Wall-clock software cost in this run environment, not physical decoder latency or a hardware scaling claim.",
            },
            "qubit_footprint": "reported as a separate circuit resource descriptor, not as a usefulness or performance score",
        },
        "experiment_design": {
            "development_seed": seed,
            "shots_per_curve_point": shots,
            "noise_sweep_profiles": profiles,
            "held_out_noise_profile": heldout_profile,
            "held_out_seeds": list(held_out_seeds),
            "sampler_seed_derivation": (
                "SHA-256 of [master_seed, sampler_seed_key, collision_counter], taking the first 8 digest bytes as an unsigned big-endian integer; "
                "collision_counter starts at 0 and increments until the seed is unused in this benchmark run"
            ),
            "sampler_seed_mapping": sampler_seed_mapping,
            "bootstrap_seed_derivation": (
                "same SHA-256 scheme as sampler seeds, keyed by 'bootstrap|<scope>|profile|master_seed|distance pair|"
                "round_multiplier|family'; one distinct seed per suppression contrast"
            ),
            "bootstrap_seed_mapping": bootstrap_seed_mapping,
            "bare_physical_noise": (
                "reset flip, per-round data depolarization and readout flip only; no gate noise because the comparator has no gates, "
                "so encoded-vs-bare differences are not a like-for-like break-even comparison"
            ),
            "reproducibility_caveat": (
                "Stim documents that a fixed seed reproduces samples only with the same Stim version on the same machine "
                "(SIMD code path); other machines give statistically equivalent but not bit-identical samples."
            ),
            "rounds": "R = distance × each configured round multiplier; repeated syndrome rounds are sampled for every point",
            "individual_intervals": (
                "two-sided 95% Wilson score interval on endpoint memory-failure probability, "
                "transformed to an IID symmetric-flip-derived endpoint-probability estimate of q"
            ),
            "suppression_intervals": (
                f"95% independent two-sample percentile bootstrap, {bootstrap} resamples, for q_d / q_(d+2); "
                "each distance-specific outcome sample is resampled separately; a null endpoint can result when a bootstrap resample has zero rate. "
                "Intervals are per-contrast and not adjusted for multiple comparisons."
            ),
            "trial_index_caveat": (
                "The optional CSV shot_index is a row index only. Different code-distance circuits do not share asserted fault trajectories, "
                "and the suppression interval does not pair outcomes by index."
            ),
            "per_cycle_method": (
                "IID symmetric-flip-derived endpoint-probability estimate q from P_fail(R) = [1 - (1 - 2q)^R]/2; "
                "not an observed per-cycle rate, a multi-duration fitted rate, or a hardware measurement."
            ),
            "decoder_model": "PyMatching graphlike matching from Stim detector error model with decompose_errors=True; correlated/hyperedge handling is approximate.",
            "software_versions": {"stim": stim.__version__, "pymatching": pymatching.__version__, "numpy": np.__version__},
        },
        "external_reference_not_compared": {
            "source": "Google Quantum AI and collaborators, 'Quantum error correction below the surface code threshold', Nature 638, 920-926 (2025), doi:10.1038/s41586-024-08449-y",
            "relation_to_this_run": (
                "Cited only as background on what an experimental (hardware) surface-code memory study measures. "
                "This classical simulation uses a different, idealized noise model and a PyMatching baseline; it does not reproduce, "
                "validate, approximate or compete with any hardware result, and its numbers must not be compared with them."
            ),
        },
        "ideal_zero_noise_control": ideal_control,
        "development_sweep": {"conditions": development_conditions},
        "held_out_validation": {
            "purpose": "independent seed and off-grid noise-profile check; held-out outputs are not used to tune the noise grid",
            "conditions_by_seed": heldout_conditions,
        },
    }
    return report, primary_vectors


def write_trials_csv(path: str | Path, trial_vectors: Mapping[str, np.ndarray], seed: int) -> None:
    """Write shot-indexed outcomes; row indices do not imply shared fault trajectories."""
    if not trial_vectors:
        raise ValueError("no trial vectors available; include the center_balanced profile and round multiplier 4")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    keys = list(trial_vectors)
    lengths = {len(trial_vectors[key]) for key in keys}
    if len(lengths) != 1:
        raise ValueError("all shot-indexed trial vectors must have the same length")
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["shot_index", "master_seed", *keys])
        for shot_index in range(next(iter(lengths))):
            writer.writerow([shot_index, seed, *(int(trial_vectors[key][shot_index]) for key in keys)])


def _primary_summary(report: Mapping[str, Any]) -> str:
    lines = [
        "CLASSICAL TOY SIMULATION OUTPUT ONLY — NOT HARDWARE DATA, NOT A FAULT-TOLERANCE OR QEC-DEMONSTRATION CLAIM",
        f"shots/point={report['experiment_design']['shots_per_curve_point']}; development seed={report['experiment_design']['development_seed']}",
        "Balanced center noise (gate=0.003, readout=0.003), longest run R=4d:",
        "d  R    code              memory failures     q (95% transformed Wilson; IID symmetric-flip-derived endpoint-probability estimate, encoded only)    decode ns/shot",
    ]
    center = next((c for c in report["development_sweep"]["conditions"] if c["noise_profile"]["name"] == _PRIMARY_NOISE_NAME), None)
    if center:
        for curve in center["curves"]:
            if curve["round_multiplier"] != _LONGEST_ROUND_MULTIPLIER:
                continue
            for family in _FAMILIES:
                metric = curve["models"][family]
                q = metric["equivalent_per_cycle_error_rate"]
                lo, hi = metric["ci95_wilson_transformed_per_cycle_rate"]
                decode_s = metric["decoder"]["timing_seconds"]["decoder_decode_seconds_per_shot"]
                if family == "bare_physical":
                    q_text = "not reported; raw memory rate above"
                else:
                    q_text = "not estimable" if q is None else f"{q:.6g} [{lo:.6g}, {hi:.6g}]"
                decode_text = "—" if decode_s is None else f"{decode_s * 1e9:.1f}"
                lines.append(
                    f"{curve['distance']:<2} {curve['rounds']:<4} {family:<18} "
                    f"{metric['memory_failure_count']:>6}/{metric['shots']:<7} "
                    f"{q_text:<36} {decode_text}"
                )
        lines.append("\nDistance +2 suppression (surface code, q_d / q_(d+2)):")
        for item in center["distance_plus_2_suppression"]:
            if item["code_family"] == "surface_code" and item["round_multiplier"] == _LONGEST_ROUND_MULTIPLIER:
                estimate = item["suppression_factor_low_rate_divided_by_high_rate"]
                ci = item["ci95_independent_two_sample_percentile_bootstrap"]
                text = "not estimable" if estimate is None else f"{estimate:.4g} CI95={ci}"
                lines.append(f"d={item['distance_low']}→{item['distance_high']}: {text} ({item['status']})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--shots", type=int, default=DEFAULT_SHOTS, help=f"shots per point (default {DEFAULT_SHOTS})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"development seed (default {DEFAULT_SEED})")
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP, help=f"independent two-sample bootstrap resamples (default {DEFAULT_BOOTSTRAP})")
    parser.add_argument("--output", type=Path, help="write aggregate JSON report to this path")
    parser.add_argument("--trials-csv", type=Path, help="write shot-indexed outcomes at center noise and R=4d (indices are not matched trajectories)")
    args = parser.parse_args(argv)
    try:
        report, vectors = run_benchmark(shots=args.shots, seed=args.seed, bootstrap=args.bootstrap)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(_primary_summary(report))
        print(f"JSON report: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(encoded)
    if args.trials_csv:
        write_trials_csv(args.trials_csv, vectors, args.seed)
        # Status goes to stderr so that `qec-scaling-bench > report.json` stays valid JSON.
        print(f"Shot-indexed outcomes (not paired trajectories): {args.trials_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

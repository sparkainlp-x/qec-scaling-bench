#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Jean-François Brisson / Spark AI NLP
"""Compare a freshly generated report with the committed classical-simulation report.

Exit 0 if every non-timing field is identical (same Stim version and SIMD code path),
or, failing that, if every memory-failure count is statistically consistent with the
committed count (two-proportion z-test, Bonferroni-corrected at family-wise alpha 0.001).
Stim only promises bit-identical samples for the same version on the same machine, and the
bootstrap intervals depend on NumPy's random-stream implementation, so the statistical
fallback is what other machines or NumPy versions (for example CI runners) can be held to.
Exit 1 otherwise. Timing fields (*_seconds*) and software_versions are always ignored.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

IGNORED_KEY_PARTS = ("seconds", "software_versions")


def _diff(a: Any, b: Any, path: str, out: list[str]) -> None:
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if any(part in key for part in IGNORED_KEY_PARTS):
                continue
            if key not in a or key not in b:
                out.append(f"{path}/{key}: missing on one side")
            else:
                _diff(a[key], b[key], f"{path}/{key}", out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for index, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{index}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")


def _counts(report: dict[str, Any]) -> dict[str, tuple[int, int]]:
    conditions = [("development", c) for c in report["development_sweep"]["conditions"]]
    conditions.append(("ideal", report["ideal_zero_noise_control"]))
    conditions += [(f"heldout_{c['seed']}", c) for c in report["held_out_validation"]["conditions_by_seed"]]
    counts = {}
    for scope, condition in conditions:
        for curve in condition["curves"]:
            for family, metric in curve["models"].items():
                key = f"{scope}|{condition['noise_profile']['name']}|d={curve['distance']}|R={curve['rounds']}|{family}"
                counts[key] = (metric["memory_failure_count"], metric["shots"])
    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: compare_reports.py COMMITTED.json FRESH.json", file=sys.stderr)
        return 2
    committed, fresh = (json.loads(Path(p).read_text(encoding="utf-8")) for p in argv)
    differences: list[str] = []
    _diff(committed, fresh, "", differences)
    if not differences:
        print("IDENTICAL: all non-timing fields match the committed report")
        return 0
    print(f"NOT BIT-IDENTICAL: {len(differences)} non-timing differences "
          "(expected with a different NumPy version, which changes bootstrap intervals, or a different Stim SIMD path)")
    for line in differences[:5]:
        print(f"  e.g. {line}")
    a, b = _counts(committed), _counts(fresh)
    if set(a) != set(b):
        print("FAIL: the two reports cover different conditions")
        return 1
    z_crit = statistics.NormalDist().inv_cdf(1 - 0.001 / (2 * len(a)))
    worst = 0.0
    failures = []
    for key in sorted(a):
        (x1, n1), (x2, n2) = a[key], b[key]
        pooled = (x1 + x2) / (n1 + n2)
        if pooled in (0.0, 1.0):  # both samples all-zero or all-one: identical rates
            continue
        z = (x1 / n1 - x2 / n2) / math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
        worst = max(worst, abs(z))
        if abs(z) > z_crit:
            failures.append(f"{key}: committed {x1}/{n1}, fresh {x2}/{n2}, z={z:.2f}")
    print(f"statistical check over {len(a)} conditions: max |z| = {worst:.2f}, critical |z| = {z_crit:.2f}")
    if failures:
        print("FAIL: statistically inconsistent counts:\n  " + "\n  ".join(failures))
        return 1
    print("CONSISTENT: all memory-failure counts are statistically consistent with the committed report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

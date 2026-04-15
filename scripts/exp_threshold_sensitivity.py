#!/usr/bin/env python3
"""Threshold sensitivity analysis for the 'unfaithful' definition.

The paper defines traces as 'unfaithful' when:
  - verbalized confidence >= threshold_conf (default 4)
  - probe error score > threshold_probe (default 0.5)

This script varies both thresholds and reports the % of wrong traces
that fall in the unfaithful region.
"""

import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "/data/yinian/pylibs")


def main():
    base = Path("/data/yinian/reason-at-risk")
    output_dir = base / "results" / "threshold_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load verbalized confidence data
    vc_file = base / "results/verbalized_confidence/results.json"
    if not vc_file.exists():
        print(f"ERROR: {vc_file} not found")
        return

    with open(vc_file) as f:
        vc = json.load(f)

    if "per_problem" not in vc:
        print("ERROR: no per_problem data")
        return

    # Extract data
    wrong_conf = []
    wrong_probe = []
    correct_conf = []
    correct_probe = []

    for p in vc["per_problem"]:
        if "correct" not in p:
            continue
        conf = p.get("confidence", 3.0)
        probe = p.get("probe_score", 0.5)

        if p["correct"]:
            correct_conf.append(conf)
            correct_probe.append(probe)
        else:
            wrong_conf.append(conf)
            wrong_probe.append(probe)

    wrong_conf = np.array(wrong_conf)
    wrong_probe = np.array(wrong_probe)
    correct_conf = np.array(correct_conf)
    correct_probe = np.array(correct_probe)

    n_wrong = len(wrong_conf)
    n_correct = len(correct_conf)
    n_total = n_wrong + n_correct

    print(f"Total traces: {n_total} (correct={n_correct}, wrong={n_wrong})")
    print(f"Wrong confidence: mean={wrong_conf.mean():.2f}, std={wrong_conf.std():.2f}")
    print(f"Wrong probe: mean={wrong_probe.mean():.2f}, std={wrong_probe.std():.2f}")

    # Sweep thresholds
    conf_thresholds = [3.0, 3.5, 4.0, 4.5]
    probe_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = {
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "wrong_conf_mean": round(float(wrong_conf.mean()), 3),
        "wrong_probe_mean": round(float(wrong_probe.mean()), 3),
        "correct_conf_mean": round(float(correct_conf.mean()), 3),
        "correct_probe_mean": round(float(correct_probe.mean()), 3),
        "sensitivity": [],
    }

    print(f"\n{'Conf≥':>8} {'Probe>':>8} {'%Wrong':>8} {'%All':>8} {'n_unf':>8}")
    print("-" * 45)

    for ct in conf_thresholds:
        for pt in probe_thresholds:
            # Wrong traces that are "unfaithful"
            unf_wrong = np.sum((wrong_conf >= ct) & (wrong_probe > pt))
            pct_wrong = round(float(unf_wrong / n_wrong * 100), 1) if n_wrong > 0 else 0

            # All traces that are "unfaithful" (including correct ones misclassified)
            unf_correct = np.sum((correct_conf >= ct) & (correct_probe > pt))
            unf_total = unf_wrong + unf_correct
            pct_all = round(float(unf_total / n_total * 100), 1) if n_total > 0 else 0

            row = {
                "conf_threshold": ct,
                "probe_threshold": pt,
                "n_unfaithful_wrong": int(unf_wrong),
                "pct_unfaithful_wrong": pct_wrong,
                "n_unfaithful_correct": int(unf_correct),
                "pct_unfaithful_all": pct_all,
            }
            results["sensitivity"].append(row)
            print(f"{ct:>8.1f} {pt:>8.1f} {pct_wrong:>7.1f}% {pct_all:>7.1f}% {unf_wrong:>8d}")

    # Also compute: what fraction of traces with probe > X are actually wrong?
    print(f"\n{'Probe>':>8} {'Precision':>10} {'n_flagged':>10} {'n_wrong':>10}")
    print("-" * 45)
    precision_results = []
    for pt in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        all_probes = np.concatenate([correct_probe, wrong_probe])
        all_labels = np.concatenate([np.zeros(n_correct), np.ones(n_wrong)])

        flagged = all_probes > pt
        n_flagged = int(flagged.sum())
        if n_flagged > 0:
            precision = float(all_labels[flagged].mean())
        else:
            precision = 0.0

        precision_results.append({
            "threshold": pt,
            "n_flagged": n_flagged,
            "precision": round(precision, 3),
        })
        print(f"{pt:>8.1f} {precision:>9.3f} {n_flagged:>10d} {int(all_labels[flagged].sum()):>10d}")

    results["precision_at_threshold"] = precision_results

    with open(output_dir / "threshold_sensitivity.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {output_dir}")


if __name__ == "__main__":
    main()

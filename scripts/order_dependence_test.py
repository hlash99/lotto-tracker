#!/usr/bin/env python3
"""
Test the within-draw order-dependence hypothesis on drawn-order Powerball data.

Hypothesis under test: the ball drawn at position i physically influences the
ball at position i+1 (airflow, ball wear, chamber dynamics), so the ORDER in
which a given set of five balls emerges is not random.

Null hypothesis: given the set of five balls in a draw, all 120 orderings are
equally likely (order exchangeability). This null is tested by permutation:
we shuffle the order WITHIN each draw thousands of times and compare the
observed statistics against that null distribution. Because only order is
shuffled — never the sets — the test is immune to number-frequency quirks
and to the negative correlation that sampling without replacement induces.

Statistics tested:
  1. lag-1 correlation      value at position i vs position i+1 (pooled)
  2. mean |step|            mean absolute difference between successive balls
  3. position effect        variance of per-position mean values
  4. first-ball mean        is the first ball out systematically high/low?

Usage:
  python3 order_dependence_test.py [--csv drawn_order.csv]
                                   [--era current|all] [--perms 10000]
"""

import argparse
import csv
import sys

import numpy as np

CURRENT_MATRIX_START = "2015-10-07"


def load(csv_path, era):
    draws = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if era == "current" and row["draw_date"] < CURRENT_MATRIX_START:
                continue
            draws.append([int(row[f"d{i}"]) for i in range(1, 6)])
    return np.array(draws, dtype=float)


def lag1_corr(x):
    a = x[:, :4].ravel()
    b = x[:, 1:].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def mean_abs_step(x):
    return float(np.abs(np.diff(x, axis=1)).mean())


def position_effect(x):
    return float(x.mean(axis=0).var())


def first_ball_mean(x):
    return float(x[:, 0].mean())


STATS = [
    ("lag-1 correlation (ball i vs ball i+1)", lag1_corr, "two"),
    ("mean |step| between successive balls", mean_abs_step, "two"),
    ("position effect (var of position means)", position_effect, "one"),
    ("mean value of first ball drawn", first_ball_mean, "two"),
]


def permute_within_rows(x, rng):
    """Independently shuffle the order of each row (draw)."""
    keys = rng.random(x.shape)
    idx = np.argsort(keys, axis=1)
    return np.take_along_axis(x, idx, axis=1)


def main():
    ap = argparse.ArgumentParser(description="Permutation test for within-draw order dependence")
    ap.add_argument("--csv", default="drawn_order.csv")
    ap.add_argument("--era", choices=["current", "all"], default="current",
                    help="'current' = 69/26 matrix only (default); 'all' = full history")
    ap.add_argument("--perms", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    x = load(args.csv, args.era)
    if len(x) < 50:
        sys.exit(f"Only {len(x)} draws loaded — not enough to test.")

    rng = np.random.default_rng(args.seed)
    print(f"Draws analyzed: {len(x)} "
          f"({'current 69-ball matrix' if args.era == 'current' else 'all eras pooled'})")
    print(f"Permutations: {args.perms} (order shuffled within each draw)\n")

    observed = [fn(x) for _, fn, _ in STATS]
    null = np.empty((args.perms, len(STATS)))
    for p in range(args.perms):
        xp = permute_within_rows(x, rng)
        for j, (_, fn, _) in enumerate(STATS):
            null[p, j] = fn(xp)

    print(f"{'statistic':<42} {'observed':>10} {'null mean':>10} {'p-value':>8}")
    print("-" * 74)
    any_signif = False
    for j, (name, _, sided) in enumerate(STATS):
        obs, ns = observed[j], null[:, j]
        if sided == "two":
            centered = np.abs(ns - ns.mean())
            p = (1 + np.sum(centered >= abs(obs - ns.mean()))) / (args.perms + 1)
        else:
            p = (1 + np.sum(ns >= obs)) / (args.perms + 1)
        flag = ""
        if p < 0.05 / len(STATS):  # Bonferroni across the 4 tests
            flag = "  <-- significant after correction"
            any_signif = True
        elif p < 0.05:
            flag = "  (not significant after multiple-test correction)"
        print(f"{name:<42} {obs:>10.4f} {ns.mean():>10.4f} {p:>8.4f}{flag}")

    print(f"""
Per-position mean values (null expectation: all equal):
  {'  '.join(f'pos{i+1}={m:.2f}' for i, m in enumerate(x.mean(axis=0)))}

Interpretation:
  {'At least one statistic deviates from order-exchangeability more than'
   ' chance allows. Before concluding physics: check for data-entry'
   ' conventions, era boundaries, and rerun on a held-out date range.'
   if any_signif else
   'The observed drawn-order sequences are statistically indistinguishable'
   ' from random orderings of the same ball sets. No detectable influence'
   ' of ball i on ball i+1 at this sample size.'}
  Note: MUSL rotates between multiple machines and ball sets, selected
  randomly before each draw, so any real aerodynamic effect would also have
  to survive averaging across equipment.""")


if __name__ == "__main__":
    main()

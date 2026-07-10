#!/usr/bin/env python3
"""
JMP-style pairwise-density cluster picker (port of f5.jsl) + walk-forward backtest.

Method (per game, on a recent window of draws):
  1. For every pair of number positions (i,j) — all 10 pairs, like the JSL
     Fit Group — fit a 2D Gaussian KDE and select the observed points whose
     min-max-normalized density is in [0.94, 1.0] (JMP: Select Points by
     Density(.94, 1)).
  2. Pool the selected values per position, weighted by density, to build a
     small candidate pool per position.
  3. Score every combination of candidates by the sum of log pair-densities
     across all 10 pairs (the "overlapping clusters" criterion) and emit the
     top-scoring distinct sets as suggested tickets.
  4. Special ball (Powerball / Mega — drawn in its own chamber, so no pairwise
     structure): 1D KDE over the recent window, find density modes, cluster
     values to their nearest mode, and pick from the cluster whose members
     appeared most recently.

Backtest: walk forward through history. At each draw t, fit on the window
ending at t-1, generate tickets, score them against draw t. Tally match
counts vs the exact hypergeometric expectation for random tickets, so the
method's skill (or lack of it) is measured, not assumed.

Powerball runs on DRAWN ORDER (d1..d5). SuperLotto runs on sorted positions
(the CA feed publishes no drawn order).

Usage:
  python3 cluster_picker.py suggest  --game powerball|superlotto [options]
  python3 cluster_picker.py backtest --game powerball|superlotto [options]
Options: --window 100  --tickets 5  --density-low 0.94  --pool 3
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from itertools import combinations, product

import numpy as np
from scipy.stats import gaussian_kde

GAMES = {
    "powerball": {
        "csv": "drawn_order.csv", "cols": [f"d{i}" for i in range(1, 6)],
        "special_col": "powerball", "white_max": 69, "special_max": 26,
        "special_name": "PB", "order": "drawn order (TX feed)",
        "start": "2015-10-07",
    },
    "superlotto": {
        # The pairwise num-i vs num-j method is only meaningful on DRAWN ORDER.
        # CA publishes sorted numbers only, so this game stays disabled until a
        # drawn-order source exists (e.g. a CPRA records request — see README).
        "csv": "superlotto_drawn_order.csv", "cols": [f"d{i}" for i in range(1, 6)],
        "special_col": "mega", "white_max": 47, "special_max": 27,
        "special_name": "Mega", "order": "drawn order (required)",
        "start": "2000-06-01",
    },
}

# Adjacent draw positions only: ball i can plausibly influence ball i+1;
# non-adjacent pairs (e.g. 2 vs 4) have no physical mechanism and are excluded.
PAIRS = [(0, 1), (1, 2), (2, 3), (3, 4)]
EPS = 1e-12


def load(game):
    dates, whites, specials = [], [], []
    with open(game["csv"], newline="") as f:
        for row in csv.DictReader(f):
            if row["draw_date"] < game["start"]:
                continue
            dates.append(row["draw_date"])
            whites.append([int(row[c]) for c in game["cols"]])
            specials.append(int(row[game["special_col"]]))
    return dates, np.array(whites, float), np.array(specials, float)


# ---------------- core method ----------------

def high_density_points(x, y, lo):
    """Indices of points whose min-max-normalized KDE density is in [lo, 1]."""
    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return np.array([], int), None
    d = kde(np.vstack([x, y]))
    rng = d.max() - d.min()
    if rng <= 0:
        return np.array([], int), kde
    dn = (d - d.min()) / rng
    return np.where(dn >= lo)[0], kde


def candidate_pools(win, lo, pool_size):
    """Per-position candidate values from high-density pair points, plus the
       fitted pair KDEs for combination scoring."""
    votes = [defaultdict(float) for _ in range(5)]
    kdes = {}
    for i, j in PAIRS:
        idx, kde = high_density_points(win[:, i], win[:, j], lo)
        if kde is not None:
            kdes[(i, j)] = kde
        for k in idx:
            votes[i][win[k, i]] += 1.0
            votes[j][win[k, j]] += 1.0
    pools = []
    for p in range(5):
        ranked = sorted(votes[p].items(), key=lambda kv: -kv[1])
        vals = [int(v) for v, _ in ranked[:pool_size]]
        if not vals:  # no clear clustering at this position: fall back to mode
            vals_, counts = np.unique(win[:, p], return_counts=True)
            vals = [int(vals_[np.argmax(counts)])]
        pools.append(vals)
    return pools, kdes


def pair_score_tables(pools, kdes):
    """Evaluate each pair KDE once on its candidate grid -> log-density lookup."""
    tables = {}
    for (i, j), kde in kdes.items():
        xs, ys = pools[i], pools[j]
        grid = np.array([[x for x in xs for _ in ys],
                         [y for _ in xs for y in ys]], float)
        dens = kde(grid)
        tables[(i, j)] = {(x, y): math.log(float(d) + EPS)
                          for (x, y), d in zip(((x, y) for x in xs for y in ys), dens)}
    return tables


def suggest_whites(win, lo, pool_size, n_tickets):
    pools, kdes = candidate_pools(win, lo, pool_size)
    tables = pair_score_tables(pools, kdes)
    scored = []
    for combo in product(*pools):
        if len(set(combo)) != 5:
            continue
        s = sum(t[(combo[i], combo[j])] for (i, j), t in tables.items())
        scored.append((s, combo))
    scored.sort(reverse=True)
    out, seen = [], set()
    for s, combo in scored:
        key = frozenset(combo)
        if key in seen:
            continue
        seen.add(key)
        out.append((s, list(combo)))
        if len(out) >= n_tickets:
            break
    return out, pools


def suggest_special(specials, smax):
    """Cluster recent specials with a 1D KDE; pick the highest-density value
       from the cluster that appeared most recently."""
    kde = gaussian_kde(specials)
    grid = np.arange(1, smax + 1, dtype=float)
    dens = kde(grid)
    # modes = local maxima on the integer grid
    modes = [int(grid[k]) for k in range(len(grid))
             if (k == 0 or dens[k] >= dens[k - 1])
             and (k == len(grid) - 1 or dens[k] >= dens[k + 1])]
    if not modes:
        modes = [int(grid[np.argmax(dens)])]
    cluster_of = {int(v): min(modes, key=lambda m: abs(m - v)) for v in range(1, smax + 1)}
    # most recent draw decides the active cluster
    active = cluster_of[int(specials[-1])]
    members = [v for v in range(1, smax + 1) if cluster_of[v] == active]
    best = max(members, key=lambda v: dens[v - 1])
    return best, active, members


# ---------------- backtest ----------------

def hypergeom_pmf(k, white_max):
    """P(exactly k of 5 whites match) for a random ticket."""
    return (math.comb(5, k) * math.comb(white_max - 5, 5 - k)
            / math.comb(white_max, 5))


def run_backtest(whites, specials, game, window, lo, pool_size, n_tickets,
                 start_frac=0.0, end_frac=1.0):
    """Walk-forward backtest over test draws in [start_frac, end_frac) of
       history. Returns a metrics dict."""
    n = len(whites)
    t0 = max(window, int(start_frac * n))
    t1 = int(end_frac * n)
    tallies = np.zeros(6, int)          # match counts 0..5, summed over tickets
    special_hits = special_preds = 0
    tickets_played = 0
    tested = 0

    for t in range(t0, t1):
        win = whites[t - window:t]
        try:
            tix, _ = suggest_whites(win, lo, pool_size, n_tickets)
        except Exception:
            continue
        if not tix:
            continue
        actual = set(int(v) for v in whites[t])
        for _, combo in tix:
            tallies[len(actual & set(combo))] += 1
            tickets_played += 1
        try:
            pred, _, _ = suggest_special(specials[t - window:t], game["special_max"])
            special_preds += 1
            if pred == int(specials[t]):
                special_hits += 1
        except Exception:
            pass
        tested += 1
        if tested % 200 == 0:
            print(f"  ...{tested} draws backtested (window={window})", file=sys.stderr)

    mean_m = sum(k * tallies[k] for k in range(6)) / max(1, tickets_played)
    exp_mean = 25 / game["white_max"]
    p3 = sum(hypergeom_pmf(k, game["white_max"]) for k in range(3, 6))
    exp3, obs3 = p3 * tickets_played, int(tallies[3:].sum())
    z3 = (obs3 - exp3) / math.sqrt(exp3) if exp3 > 0 else float("nan")
    return {
        "window": window, "tested": tested, "tickets": tickets_played,
        "tallies": [int(x) for x in tallies],
        "mean_matches": round(mean_m, 4), "random_mean": round(exp_mean, 4),
        "mean_ratio": round(mean_m / exp_mean, 4) if exp_mean else None,
        "obs3": obs3, "exp3": round(exp3, 2), "z3": round(z3, 2),
        "special_hits": special_hits, "special_preds": special_preds,
        "special_rate": round(special_hits / special_preds, 4) if special_preds else None,
        "special_random": round(1 / game["special_max"], 4),
    }


def backtest(dates, whites, specials, game, window, lo, pool_size, n_tickets,
             start_frac=0.0, end_frac=1.0, as_json=False):
    m = run_backtest(whites, specials, game, window, lo, pool_size, n_tickets,
                     start_frac, end_frac)
    if as_json:
        import json
        print(json.dumps(m))
        return
    tallies = m["tallies"]
    tickets_played = m["tickets"]
    tested = m["tested"]
    special_hits, special_preds = m["special_hits"], m["special_preds"]

    print(f"\n=== BACKTEST: {game['special_name']=='PB' and 'Powerball' or 'SuperLotto Plus'} "
          f"(positions = {game['order']}) ===")
    print(f"Walk-forward draws tested: {tested} (window={window}, "
          f"{n_tickets} tickets/draw, density>={lo})")
    print(f"Tickets played: {tickets_played}\n")
    print(f"{'matches':>8} {'method hits':>12} {'random expectation':>19} {'ratio':>7}")
    for k in range(2, 6):
        exp = hypergeom_pmf(k, game["white_max"]) * tickets_played
        obs = tallies[k]
        ratio = obs / exp if exp > 0 else float("nan")
        print(f"{k}/5{'':>4} {obs:>12} {exp:>19.2f} {ratio:>7.2f}")
    print(f"\nMean white matches/ticket: {m['mean_matches']:.4f} "
          f"(random: {m['random_mean']:.4f}, ratio {m['mean_ratio']:.3f})")
    if special_preds:
        print(f"{game['special_name']} cluster pick hit rate: "
              f"{special_hits}/{special_preds} = {m['special_rate']:.4f} "
              f"(random: {m['special_random']:.4f})")
    z = m["z3"]
    print(f"3+ matches: {m['obs3']} vs {m['exp3']:.1f} expected -> z = {z:+.2f} "
          f"({'consistent with chance' if abs(z) < 2 else 'outside chance range — investigate'})")


# ---------------- suggest (today) ----------------

def suggest_today(dates, whites, specials, game, window, lo, pool_size, n_tickets):
    win = whites[-window:]
    tix, pools = suggest_whites(win, lo, pool_size, n_tickets)
    pred, active, members = suggest_special(specials[-window:], game["special_max"])

    name = "Powerball" if game["special_name"] == "PB" else "SuperLotto Plus"
    print(f"=== {name} cluster suggestions ===")
    print(f"Window: last {len(win)} draws ({dates[-len(win)]} to {dates[-1]}), "
          f"positions = {game['order']}")
    print(f"Candidate pools per position (from density>={lo} pair overlaps):")
    for p, pool in enumerate(pools, 1):
        print(f"  pos{p}: {pool}")
    print(f"\n{game['special_name']}: most-recently-active cluster around {active} "
          f"(members {members[0]}–{members[-1]}) -> pick {pred}\n")
    print("Suggested tickets (sorted for playslip; score = sum of log pair densities):")
    hist_sets = [frozenset(int(v) for v in w) for w in whites]
    for i, (s, combo) in enumerate(tix, 1):
        played = sorted(set(combo))
        # check against ALL past draws for hits
        best_k, best_at = 0, None
        exact = 0
        for d, hs in zip(dates, hist_sets):
            k = len(hs & set(combo))
            if k > best_k:
                best_k, best_at = k, d
            if k == 5:
                exact += 1
        print(f"  {i}. {' '.join(f'{v:2d}' for v in played)}  "
              f"{game['special_name']} {pred:2d}   (score {s:.1f}; "
              f"best past overlap {best_k}/5 on {best_at}"
              + (f"; {exact} exact past hits!" if exact else "") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("command", choices=["suggest", "backtest"])
    ap.add_argument("--game", choices=sorted(GAMES), default="powerball")
    ap.add_argument("--window", type=int, default=100,
                    help="Recent draws used for clustering (default 100)")
    ap.add_argument("--tickets", type=int, default=5)
    ap.add_argument("--density-low", type=float, default=0.94,
                    help="Normalized density selection threshold (JSL used .94)")
    ap.add_argument("--pool", type=int, default=3,
                    help="Candidate values kept per position (default 3)")
    ap.add_argument("--start-frac", type=float, default=0.0,
                    help="Backtest only draws from this fraction of history")
    ap.add_argument("--end-frac", type=float, default=1.0,
                    help="Backtest only draws before this fraction of history")
    ap.add_argument("--json", action="store_true",
                    help="Backtest: print metrics as JSON instead of a report")
    args = ap.parse_args()

    game = GAMES[args.game]
    try:
        dates, whites, specials = load(game)
    except FileNotFoundError:
        raise SystemExit(
            f"{game['csv']} not found.\n"
            "SuperLotto cluster analysis requires DRAWN-ORDER data — on sorted\n"
            "numbers the pairwise structure is an artifact of sorting and the\n"
            "method is meaningless. CA publishes sorted results only (no stream,\n"
            "no video archive). To obtain drawn order, file a CPRA records\n"
            "request with the CA Lottery for draw results in the order drawn\n"
            "(see cpra_request_draft.md), then save them as superlotto_drawn_order.csv\n"
            "with columns: draw_date,d1,d2,d3,d4,d5,mega")
    if len(whites) <= args.window:
        raise SystemExit(f"Need more than {args.window} draws, have {len(whites)}")

    if args.command == "suggest":
        suggest_today(dates, whites, specials, game,
                      args.window, args.density_low, args.pool, args.tickets)
    else:
        backtest(dates, whites, specials, game,
                 args.window, args.density_low, args.pool, args.tickets,
                 args.start_frac, args.end_frac, args.json)


if __name__ == "__main__":
    main()

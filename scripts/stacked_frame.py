#!/usr/bin/env python3
"""
Stacked-frame / hot-pixel check — the astrophotography dark-frame idea
applied to lottery draws.

Every draw is one "frame". Stacking all frames gives each ball's count; under
a fair draw the stacked frame is flat, with per-ball noise that shrinks
relative to the signal as frames accumulate. A physically biased ball (heavy,
chipped, or a machine quirk) would behave like a sensor's hot pixel:

  1. it sticks out of the stacked frame by more than the noise allows, after
     accounting for looking at every ball at once (family-wise threshold from
     the max |z| of simulated fair histories), and
  2. it is *persistent* — it sticks out the same way in independent sub-stacks
     (first half vs second half of history). Noise doesn't repeat; defects do.

A ball is a confirmed hot (or dead) pixel only if it passes both. The
recommendation, and the recommended tickets, are built from confirmed pixels
only; with none, every ball is treated equally.

Usage (standalone):  python3 scripts/stacked_frame.py [--game powerball]
"""

import argparse
import math
from pathlib import Path

import numpy as np

import picker

SIMS = 2000
FAMILY_ALPHA = 0.05      # family-wise false-alarm rate across all balls
HALF_Z = 1.645           # one-sided 5% replication bar in each half
WATCHLIST = 5


def _sim_white_counts(rng, n_draws, n_balls, sims):
    """Exact fair-draw simulation: each draw is 5 distinct balls."""
    out = np.empty((sims, n_balls), dtype=np.int32)
    for s in range(sims):
        picks = np.argpartition(rng.random((n_draws, n_balls)), 5, axis=1)[:, :5]
        out[s] = np.bincount(picks.ravel(), minlength=n_balls)
    return out


def _sim_special_counts(rng, n_draws, n_balls, sims):
    return rng.multinomial(n_draws, [1.0 / n_balls] * n_balls, size=sims)


def _z(counts, n_draws, k, n_balls):
    """Per-ball z-score of counts vs a fair draw (marginally Binomial(n, k/N))."""
    p = k / n_balls
    mu = n_draws * p
    sd = math.sqrt(n_draws * p * (1 - p))
    return (np.asarray(counts, dtype=float) - mu) / sd, mu, sd


def _corr_rows(a, b):
    a = a - a.mean(axis=-1, keepdims=True)
    b = b - b.mean(axis=-1, keepdims=True)
    return (a * b).sum(axis=-1) / np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))


def analyse_channel(rows, n_balls, k, rng, sims=SIMS):
    """rows: list of per-draw ball lists (k balls each). Returns the frame block."""
    n = len(rows)
    h = n // 2
    first, second = rows[:h], rows[h:]

    def counts(rs):
        c = np.zeros(n_balls, dtype=int)
        for r in rs:
            for v in r:
                c[v - 1] += 1
        return c

    c_all, c1, c2 = counts(rows), counts(first), counts(second)
    z_all, mu, sd = _z(c_all, n, k, n_balls)
    z1, _, _ = _z(c1, len(first), k, n_balls)
    z2, _, _ = _z(c2, len(second), k, n_balls)

    sim = _sim_white_counts if k > 1 else _sim_special_counts
    s1 = sim(rng, len(first), n_balls, sims)
    s2 = sim(rng, len(second), n_balls, sims)
    sz_all, _, _ = _z(s1 + s2, n, k, n_balls)
    sz1, _, _ = _z(s1, len(first), k, n_balls)
    sz2, _, _ = _z(s2, len(second), k, n_balls)

    # Family-wise: how extreme is the most extreme ball in a fair history?
    null_max = np.abs(sz_all).max(axis=1)
    z_crit = float(np.quantile(null_max, 1 - FAMILY_ALPHA))
    fw_p = [(1 + int(np.sum(null_max >= abs(z)))) / (sims + 1) for z in z_all]

    # Persistence: do two independent sub-stacks agree on which balls are off?
    # Halves (early vs late history) and interleaved (odd vs even draws, same
    # equipment eras in both). A real fixed pattern must show in both splits.
    r_obs = float(_corr_rows(z1, z2))
    r_null = _corr_rows(sz1, sz2)
    r_p = (1 + int(np.sum(r_null >= r_obs))) / (sims + 1)
    odd, even = rows[0::2], rows[1::2]
    zo, _, _ = _z(counts(odd), len(odd), k, n_balls)
    ze, _, _ = _z(counts(even), len(even), k, n_balls)
    ri_obs = float(_corr_rows(zo, ze))
    so = sim(rng, len(odd), n_balls, sims)
    se = sim(rng, len(even), n_balls, sims)
    ri_null = _corr_rows(_z(so, len(odd), k, n_balls)[0], _z(se, len(even), k, n_balls)[0])
    ri_p = (1 + int(np.sum(ri_null >= ri_obs))) / (sims + 1)

    # Chi-square across the whole frame (same numbers the game panels show)
    chi_stat = float(((c_all - mu) ** 2 / mu).sum())
    chi_p = picker.chi_square_pvalue(chi_stat, n_balls - 1)

    balls = []
    for i in range(n_balls):
        z, a, b = float(z_all[i]), float(z1[i]), float(z2[i])
        sign = 1 if z > 0 else -1
        persistent = a * sign >= HALF_Z and b * sign >= HALF_Z
        stands_out = abs(z) >= z_crit
        balls.append({
            "n": i + 1, "count": int(c_all[i]), "z": round(z, 2),
            "z_h1": round(a, 2), "z_h2": round(b, 2),
            "fw_p": round(fw_p[i], 3),
            "rate_ratio": round(float(c_all[i]) / mu, 3),
            "stands_out": stands_out, "persistent": persistent,
            "pixel": ("hot" if z > 0 else "dead") if (stands_out and persistent) else None,
        })

    # Detection limit: smallest bias (as a fraction of the fair rate) that
    # would reach the family-wise bar at the current depth, and the number of
    # frames needed to resolve a 10% bias.
    p = k / n_balls
    min_bias = z_crit * sd / mu
    frames_10pct = math.ceil(z_crit ** 2 * (1 - p) / (p * 0.10 ** 2))

    watch = sorted(balls, key=lambda b: -abs(b["z"]))[:WATCHLIST]
    return {
        "frames": n, "expected": round(mu, 2), "sd": round(sd, 2),
        "z_crit": round(z_crit, 2), "band2": [round(mu - 2 * sd, 1), round(mu + 2 * sd, 1)],
        "bandfw": [round(mu - z_crit * sd, 1), round(mu + z_crit * sd, 1)],
        "chi2": {"stat": round(chi_stat, 1), "df": n_balls - 1, "p": round(chi_p, 4)},
        "persistence": {"r": round(r_obs, 3), "p": round(r_p, 3),
                        "halves": [len(first), len(second)],
                        "r_interleaved": round(ri_obs, 3), "p_interleaved": round(ri_p, 3),
                        "fixed_pattern": r_p < 0.05 and ri_p < 0.05},
        "min_detectable_bias": round(min_bias, 3),
        "frames_for_10pct": frames_10pct,
        "balls": balls,
        "watchlist": [b["n"] for b in watch],
        "hot": [b["n"] for b in balls if b["pixel"] == "hot"],
        "dead": [b["n"] for b in balls if b["pixel"] == "dead"],
    }


def recommended_tickets(game, draws, white, special, n_tickets=5):
    """Screened uniform tickets, adjusted only by confirmed pixels: confirmed
    hot whites are included (up to 5), dead whites excluded; a confirmed hot
    special is used on every ticket, dead specials excluded; otherwise each
    ticket gets a different special (same EV, ~5x the chance of any
    special-ball prize vs repeating one)."""
    rng = picker.rng
    past = {(frozenset(w), s) for _, w, s in draws}
    must = sorted(white["hot"], key=lambda n: -white["balls"][n - 1]["z"])[:5]
    banned_w = set(white["dead"])
    pool_w = [v for v in range(1, game["white_max"] + 1)
              if v not in banned_w and v not in must]
    hot_s = sorted(special["hot"], key=lambda n: -special["balls"][n - 1]["z"])
    pool_s = [v for v in range(1, game["special_max"] + 1) if v not in set(special["dead"])]
    specials = ([hot_s[0]] * n_tickets if hot_s
                else rng.sample(pool_s, min(n_tickets, len(pool_s))))
    out = []
    for i in range(n_tickets):
        sp = specials[i % len(specials)]
        for _ in range(10000):
            whites = sorted(must + rng.sample(pool_w, 5 - len(must)))
            if not picker.is_popular_pattern(whites, sp, past):
                break
        out.append({"whites": whites, "special": sp})
    return out


def recommendation(white, special, special_label):
    hot_w, dead_w = white["hot"], white["dead"]
    hot_s, dead_s = special["hot"], special["dead"]
    found = hot_w or dead_w or hot_s or dead_s
    if found:
        parts = []
        if hot_w: parts.append(f"include hot white ball(s) {', '.join(map(str, hot_w))}")
        if dead_w: parts.append(f"avoid dead white ball(s) {', '.join(map(str, dead_w))}")
        if hot_s: parts.append(f"play {special_label} {hot_s[0]}")
        if dead_s: parts.append(f"avoid {special_label} {', '.join(map(str, dead_s))}")
        return {"status": "pixels", "headline": "Hot pixel(s) confirmed — " + "; ".join(parts),
                "detail": ("These balls stand out beyond the family-wise noise bar AND repeat "
                           "in both independent halves of history. The recommended tickets "
                           "below are built around them. Re-check after each refresh: a real "
                           "defect stays lit; a fluke fades as frames accumulate.")}
    return {"status": "clean",
            "headline": "Clean frame — no hot pixels. Don't favor or avoid any ball.",
            "detail": (f"No ball clears the family-wise bar (|z| ≥ {white['z_crit']}) and "
                       f"repeats in both halves. Every ball is equally likely, so the best "
                       f"play is screened random tickets with a different {special_label} on "
                       f"each — same expected value, ~5× the chance of any {special_label} "
                       f"prize versus repeating one, and fewer split jackpots.")}


def frame_block(draws, game, special_label, seed=0):
    rng = np.random.default_rng(seed)
    white = analyse_channel([w for _, w, _ in draws], game["white_max"], 5, rng)
    special = analyse_channel([[s] for _, _, s in draws], game["special_max"], 1, rng)
    return {
        "white": white, "special": special,
        "recommendation": recommendation(white, special, special_label),
        "tickets": recommended_tickets(game, draws, white, special),
        "sims": SIMS,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=sorted(picker.GAMES), default="powerball")
    args = ap.parse_args()
    game = dict(picker.GAMES[args.game])
    csv_name = {"powerball": "official_results.csv",
                "superlotto": "superlotto_results.csv"}[args.game]
    game["csv"] = str(Path(__file__).resolve().parent.parent / "data" / csv_name)
    draws = picker.load_draws(game)
    out = frame_block(draws, game, game["special_name"])
    for ch in ("white", "special"):
        b = out[ch]
        print(f"{ch}: {b['frames']} frames, E={b['expected']} sd={b['sd']} "
              f"z_crit={b['z_crit']} chi2 p={b['chi2']['p']} "
              f"persistence r={b['persistence']['r']} p={b['persistence']['p']} "
              f"interleaved r={b['persistence']['r_interleaved']} p={b['persistence']['p_interleaved']} "
              f"min bias={b['min_detectable_bias']:.0%} hot={b['hot']} dead={b['dead']}")
        for n in b["watchlist"]:
            x = b["balls"][n - 1]
            print(f"   ball {n:2d}: count {x['count']} z={x['z']:+.2f} "
                  f"(halves {x['z_h1']:+.2f} / {x['z_h2']:+.2f}) fw_p={x['fw_p']}")
    print(out["recommendation"]["headline"])
    for t in out["tickets"]:
        print("  ", t["whites"], game["special_name"], t["special"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Honest Powerball number picker + stats, built on official draw data.

Two subcommands:

  stats   Frequency tables and a chi-square goodness-of-fit test on the
          official history (current 69/26 matrix only). Spoiler: the draws
          are consistent with uniform randomness — "hot" numbers are noise.

  pick    Generate tickets. Every ticket is drawn uniformly at random
          (cryptographic RNG), then re-drawn if it matches patterns humans
          disproportionately play (all-birthday numbers, arithmetic
          sequences, past jackpot combinations). This does NOT change your
          odds of winning — nothing can — but if you do win, unpopular
          combinations are less likely to split the jackpot.

Usage:
  python3 powerball_picker.py stats [--game powerball|superlotto]
  python3 powerball_picker.py pick [-n 5] [--game powerball|superlotto]
"""

import argparse
import csv
import math
import random
from collections import Counter
from pathlib import Path

GAMES = {
    "powerball": {
        "csv": "official_results.csv",
        "white_max": 69, "special_max": 26, "special_name": "PB",
        "special_col": "powerball",
        "matrix_start": "2015-10-07",  # current 69/26 matrix
    },
    "superlotto": {
        "csv": "superlotto_results.csv",
        "white_max": 47, "special_max": 27, "special_name": "Mega",
        "special_col": "mega",
        "matrix_start": "2000-06-01",  # 47/27 format since June 2000
    },
}

rng = random.SystemRandom()


def jackpot_odds(game):
    return math.comb(game["white_max"], 5) * game["special_max"]


def load_draws(game):
    """-> list of (date, [5 whites sorted], special) for the current matrix only."""
    draws = []
    with open(game["csv"], newline="") as f:
        for row in csv.DictReader(f):
            if row["draw_date"] < game["matrix_start"]:
                continue
            whites = sorted(int(row[f"n{i}"]) for i in range(1, 6))
            draws.append((row["draw_date"], whites, int(row[game["special_col"]])))
    return draws


# ---------------- stats ----------------

def chi_square_uniform(counts, n_categories, n_observations):
    """Chi-square goodness-of-fit statistic + df against a uniform distribution."""
    expected = n_observations / n_categories
    stat = sum((counts.get(i, 0) - expected) ** 2 / expected
               for i in range(1, n_categories + 1))
    return stat, n_categories - 1


def chi_square_pvalue(stat, df):
    """Survival function of the chi-square distribution (regularized gamma)."""
    # Uses scipy when available, else a series/continued-fraction fallback.
    try:
        from scipy.stats import chi2
        return float(chi2.sf(stat, df))
    except ImportError:
        return 1.0 - _gammainc_lower_reg(df / 2.0, stat / 2.0)


def _gammainc_lower_reg(s, x):
    if x <= 0:
        return 0.0
    if x < s + 1:  # series expansion
        term = 1.0 / s
        total = term
        k = s
        while True:
            k += 1
            term *= x / k
            total += term
            if term < total * 1e-12:
                break
        return total * math.exp(-x + s * math.log(x) - math.lgamma(s))
    return 1.0 - _gammainc_upper_reg(s, x)


def _gammainc_upper_reg(s, x):
    # Lentz continued fraction for the upper incomplete gamma
    tiny = 1e-300
    b, c, d = x + 1 - s, 1 / tiny, 1 / (x + 1 - s)
    h = d
    for i in range(1, 200):
        an = -i * (i - s)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > tiny else tiny)
        c = b + an / (c if abs(c) > tiny else tiny)
        h *= d * c
    return h * math.exp(-x + s * math.log(x) - math.lgamma(s))


def run_stats(draws, game):
    n = len(draws)
    wmax, smax, sname = game["white_max"], game["special_max"], game["special_name"]
    print(f"Official draws under the current {wmax}/{smax} matrix: {n} "
          f"({draws[0][0]} to {draws[-1][0]})\n")

    white_counts = Counter(w for _, whites, _ in draws for w in whites)
    special_counts = Counter(s for _, _, s in draws)

    def show(label, counts, top=8):
        most = counts.most_common(top)
        least = counts.most_common()[: -top - 1 : -1]
        print(f"{label}")
        print("  most drawn:  " + ", ".join(f"{k} ({v}x)" for k, v in most))
        print("  least drawn: " + ", ".join(f"{k} ({v}x)" for k, v in least))

    show(f"White balls (1-{wmax}, {5 * n} balls drawn):", white_counts)
    show(f"\n{sname} (1-{smax}, {n} drawn):", special_counts)

    for label, counts, cats, obs in (
        ("white balls", white_counts, wmax, 5 * n),
        (sname, special_counts, smax, n),
    ):
        stat, df = chi_square_uniform(counts, cats, obs)
        p = chi_square_pvalue(stat, df)
        verdict = ("consistent with a fair, uniform draw"
                   if p > 0.05 else "unusual — but check data quality before excitement")
        print(f"\nChi-square test, {label}: stat={stat:.1f}, df={df}, p={p:.3f}")
        print(f"  -> {verdict}")

    print(f"""
What this means for picking:
  Every draw is independent. The spread between "hot" and "cold" numbers
  above is exactly what fair dice produce over {n} draws. No frequency,
  density, gap, or overlap analysis of past draws changes the odds of any
  future ticket: every combination is 1 in {jackpot_odds(game):,}.
  The only choice that matters is avoiding combinations OTHER PEOPLE play,
  so a jackpot is less likely to be split. That's what `pick` does.""")


# ---------------- picking ----------------

def is_popular_pattern(whites, special, past_winners):
    """True if humans disproportionately play this ticket shape."""
    # Birthday tickets: every number playable as a calendar date
    if all(w <= 31 for w in whites):
        return True
    # Arithmetic sequences (1-2-3-4-5, 5-10-15-20-25, ...)
    diffs = {b - a for a, b in zip(whites, whites[1:])}
    if len(diffs) == 1:
        return True
    # Past jackpot combinations get replayed
    if (frozenset(whites), special) in past_winners:
        return True
    # All same last digit reads as a "system" and is over-played
    if len({w % 10 for w in whites}) == 1:
        return True
    return False


def make_ticket(game, past_winners):
    while True:
        whites = sorted(rng.sample(range(1, game["white_max"] + 1), 5))
        special = rng.randint(1, game["special_max"])
        if not is_popular_pattern(whites, special, past_winners):
            return whites, special


def run_pick(draws, game, n_tickets):
    past_winners = {(frozenset(w), s) for _, w, s in draws}
    print(f"{n_tickets} ticket(s) — uniform random, screened against "
          f"{len(past_winners)} past winning combos and popular patterns:\n")
    for i in range(1, n_tickets + 1):
        whites, special = make_ticket(game, past_winners)
        print(f"  {i}. {' '.join(f'{w:2d}' for w in whites)}  "
              f"{game['special_name']} {special:2d}")
    print(f"\nOdds per ticket: 1 in {jackpot_odds(game):,} for the jackpot "
          f"(unchanged by any strategy).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["stats", "pick"])
    ap.add_argument("-n", "--tickets", type=int, default=5,
                    help="Number of tickets to generate (pick mode, default 5)")
    ap.add_argument("--game", choices=sorted(GAMES), default="powerball")
    ap.add_argument("--csv", default=None,
                    help="Override results CSV (default: per-game file)")
    args = ap.parse_args()

    game = dict(GAMES[args.game])
    if args.csv:
        game["csv"] = args.csv
    if not Path(game["csv"]).exists():
        raise SystemExit(f"{game['csv']} not found — run the fetch script first")
    draws = load_draws(game)
    if not draws:
        raise SystemExit("No current-matrix draws found in CSV")

    if args.command == "stats":
        run_stats(draws, game)
    else:
        run_pick(draws, game, args.tickets)


if __name__ == "__main__":
    main()

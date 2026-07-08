#!/usr/bin/env python3
"""
Server-side refresh for the lotto tracker.

1. Fetches fresh official results (Powerball sorted + drawn order, SuperLotto).
2. Recomputes all analysis: frequency stats, chi-square vs uniform, the
   within-draw order-dependence permutation test, and screened quick picks.
3. Writes data.json for index.html and the dashboard card.

Run from anywhere: paths resolve relative to the repo root.
"""

import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DATA = ROOT / "data"

sys.path.insert(0, str(SCRIPTS))
import picker  # noqa: E402
import order_dependence_test as odt  # noqa: E402
import cluster_picker as cp  # noqa: E402

ORDER_TEST_PERMS = 5000

# Cluster-method window, tuned 2026-07-08 by walk-forward sweep
# (50/75/100/150/200/300) on the first 70% of history, validated on the
# held-out 30%. w=200 had the best mean-match ratio on BOTH slices
# (train 1.034, holdout 1.096) but zero 3+ hits in holdout (3.8 expected) —
# metrics disagree, so treat as no demonstrated edge. See data.json tuning.
CLUSTER_WINDOW = 200
CLUSTER_TUNING = {
    "tuned": "2026-07-08", "windows_swept": [50, 75, 100, 150, 200, 300],
    "train_mean_ratio": 1.034, "holdout_mean_ratio": 1.096,
    "holdout_hits3": 0, "holdout_hits3_expected": 3.77,
    "verdict": "metrics disagree across slices — consistent with chance",
}


def run(script, *args):
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    print(f"$ {' '.join(cmd[1:])}", flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def freq_block(draws, game):
    """Frequency + chi-square summary for one game."""
    n = len(draws)
    whites = Counter(w for _, ws, _ in draws for w in ws)
    specials = Counter(s for _, _, s in draws)

    def chi(counts, cats, obs):
        stat, df = picker.chi_square_uniform(counts, cats, obs)
        p = picker.chi_square_pvalue(stat, df)
        return {"stat": round(stat, 1), "df": df, "p": round(p, 4),
                "uniform": p > 0.05}

    return {
        "draws": n,
        "first_date": draws[0][0],
        "last_date": draws[-1][0],
        "white_hot": whites.most_common(10),
        "white_cold": whites.most_common()[:-11:-1],
        "special_hot": specials.most_common(5),
        "special_cold": specials.most_common()[:-6:-1],
        "chi2_white": chi(whites, game["white_max"], 5 * n),
        "chi2_special": chi(specials, game["special_max"], n),
    }


def picks_block(draws, game, n_tickets=5):
    past = {(frozenset(w), s) for _, w, s in draws}
    out = []
    for _ in range(n_tickets):
        whites, special = picker.make_ticket(game, past)
        out.append({"whites": whites, "special": special})
    return out


def order_test_block():
    x = odt.load(DATA / "drawn_order.csv", "current")
    import numpy as np
    rng = np.random.default_rng(0)
    observed = [fn(x) for _, fn, _ in odt.STATS]
    null = np.empty((ORDER_TEST_PERMS, len(odt.STATS)))
    for p in range(ORDER_TEST_PERMS):
        xp = odt.permute_within_rows(x, rng)
        for j, (_, fn, _) in enumerate(odt.STATS):
            null[p, j] = fn(xp)

    results = []
    for j, (name, _, sided) in enumerate(odt.STATS):
        obs, ns = observed[j], null[:, j]
        if sided == "two":
            centered = np.abs(ns - ns.mean())
            pv = (1 + np.sum(centered >= abs(obs - ns.mean()))) / (ORDER_TEST_PERMS + 1)
        else:
            pv = (1 + np.sum(ns >= obs)) / (ORDER_TEST_PERMS + 1)
        results.append({"name": name, "observed": round(obs, 4),
                        "null_mean": round(float(ns.mean()), 4),
                        "p": round(float(pv), 4)})
    significant = any(r["p"] < 0.05 / len(results) for r in results)
    return {"draws": int(len(x)), "perms": ORDER_TEST_PERMS,
            "results": results, "dependence_detected": significant}


def latest_block(game_key):
    """Most recent draw, with drawn order for Powerball."""
    if game_key == "powerball":
        import csv
        with open(DATA / "drawn_order.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        last = rows[-1]
        drawn = [int(last[f"d{i}"]) for i in range(1, 6)]
        return {"date": last["draw_date"], "whites": sorted(drawn),
                "drawn_order": drawn, "special": int(last["powerball"]),
                "power_play": last["power_play"]}
    import csv
    with open(DATA / "superlotto_results.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return {"date": last["draw_date"],
            "whites": [int(last[f"n{i}"]) for i in range(1, 6)],
            "special": int(last["mega"])}


def cluster_block():
    """f5.jsl-style pairwise-KDE cluster picks + rolling walk-forward backtest.
       Powerball only: the method requires drawn order, which CA does not
       publish for SuperLotto."""
    game = dict(cp.GAMES["powerball"])
    game["csv"] = str(DATA / "drawn_order.csv")
    dates, whites, specials = cp.load(game)
    W = CLUSTER_WINDOW

    tix, pools = cp.suggest_whites(whites[-W:], 0.94, 3, 5)
    pred, active, members = cp.suggest_special(specials[-W:], game["special_max"])
    hist_sets = [frozenset(int(v) for v in w) for w in whites]
    tickets = []
    for s, combo in tix:
        best = max(len(hs & set(combo)) for hs in hist_sets)
        tickets.append({"whites": sorted(set(combo)), "positional": list(combo),
                        "score": round(s, 1), "best_past_overlap": best})

    bt = cp.run_backtest(whites, specials, game, W, 0.94, 3, 5)
    return {
        "window": W,
        "window_dates": [dates[-W], dates[-1]],
        "pools": pools,
        "tickets": tickets,
        "special_pick": pred,
        "special_cluster": [members[0], members[-1]],
        "backtest": bt,
        "tuning": CLUSTER_TUNING,
    }


def main():
    run("fetch_official_results.py", "--out", str(DATA / "official_results.csv"))
    run("fetch_drawn_order.py", "--out", str(DATA / "drawn_order.csv"),
        "--official", str(DATA / "official_results.csv"))
    run("fetch_superlotto.py", "--out", str(DATA / "superlotto_results.csv"))

    out = {"updated": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    for key, csv_name, special_label in (
            ("powerball", "official_results.csv", "Powerball"),
            ("superlotto", "superlotto_results.csv", "Mega")):
        game = dict(picker.GAMES[key])
        game["csv"] = str(DATA / csv_name)
        draws = picker.load_draws(game)
        out[key] = {
            "special_label": special_label,
            "white_max": game["white_max"],
            "special_max": game["special_max"],
            "odds": picker.jackpot_odds(game),
            "latest": latest_block(key),
            "stats": freq_block(draws, game),
            "picks": picks_block(draws, game),
        }

    out["powerball"]["order_test"] = order_test_block()
    out["powerball"]["cluster"] = cluster_block()
    out["superlotto"]["order_note"] = (
        "CA Lottery publishes SuperLotto numbers sorted; drawn order is not "
        "available from any official feed.")
    out["superlotto"]["cluster"] = {
        "available": False,
        "reason": ("Cluster method requires drawn order. CA conducts mechanical "
                   "draws but publishes sorted numbers only — no stream archive "
                   "or drawing videos exist. A CPRA records request for the "
                   "official draw sequence is the remaining path."),
    }

    with open(ROOT / "data.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"Wrote data.json (updated {out['updated']})")


if __name__ == "__main__":
    main()

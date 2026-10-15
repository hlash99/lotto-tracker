#!/usr/bin/env python3
"""
One recommended ticket set for Powerball, combining the two analyses on the
page — each one only gets a say when its evidence clears a fixed bar:

  * Hot-pixel check (stacked_frame.py): confirmed hot balls are included,
    dead balls excluded. Applies whenever a pixel is confirmed.
  * Cluster method (cluster_picker.py): its tickets are used only once its
    OUT-OF-SAMPLE record — draws after the window/density were tuned — beats
    chance at p < GATE_ALPHA over at least GATE_MIN_DRAWS draws. Earlier
    draws were used to choose the settings, so they can't vouch for them.

The test is exact and accounts for ticket overlap: for each draw, the five
tickets were fixed before the draw, so the only randomness under a fair draw
is which 5 balls come out. Total white matches across the tickets then has a
known mean and variance (hypergeometric indicators weighted by how many
tickets carry each ball), summed over draws into one z-score.

Otherwise the set is screened random tickets with a different special ball on
each — same expected value as any other choice, but the highest chance of
some prize per draw and the lowest chance of a shared jackpot.
"""

import math

import numpy as np

GATE_MIN_DRAWS = 100
GATE_ALPHA = 0.01


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def match_sum_test(records, whites, white_max):
    """records: (draw index, [ticket combos], special pick). Exact
    overlap-aware test of total white matches vs a fair draw."""
    p = 5 / white_max
    var1 = p * (1 - p)
    cov = (5 * 4) / (white_max * (white_max - 1)) - p * p
    obs = exp = var = 0.0
    for t, combos, _ in records:
        m = np.zeros(white_max + 1)
        for c in combos:
            for v in c:
                m[v] += 1
        actual = set(int(v) for v in whites[t])
        obs += sum(m[v] for v in actual)
        exp += p * m.sum()
        var += (var1 - cov) * (m * m).sum() + cov * m.sum() ** 2
    if not records:
        return {"draws": 0}
    z = (obs - exp) / math.sqrt(var)
    return {"draws": len(records), "obs": int(obs), "exp": round(float(exp), 1),
            "ratio": round(float(obs / exp), 3), "z": round(float(z), 2),
            "p": round(_norm_sf(float(z)), 4)}


def special_test(records, specials, special_max):
    """One-sided binomial test of the cluster special-ball pick."""
    preds = [(t, s) for t, _, s in records if s is not None]
    n = len(preds)
    if not n:
        return {"draws": 0}
    k = sum(1 for t, s in preds if int(specials[t]) == s)
    q = 1 / special_max
    logpmf = lambda i: (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                        + i * math.log(q) + (n - i) * math.log(1 - q))
    p = min(1.0, sum(math.exp(logpmf(i)) for i in range(k, n + 1)))
    return {"draws": n, "hits": k, "exp": round(n * q, 1), "p": round(p, 4)}


def prize_profile(tickets, white_max, special_max, sims=200_000, seed=0):
    """Chance per draw that at least one ticket wins anything (special-ball
    match, or 3+ whites), estimated over simulated fair draws."""
    rng = np.random.default_rng(seed)
    draws = np.argpartition(rng.random((sims, white_max)), 5, axis=1)[:, :5] + 1
    sp = rng.integers(1, special_max + 1, sims)
    drawn = np.zeros((sims, white_max + 1), dtype=bool)
    np.put_along_axis(drawn, draws, True, axis=1)
    any_prize = np.zeros(sims, dtype=bool)
    for t in tickets:
        k = drawn[:, t["whites"]].sum(axis=1)
        any_prize |= (k >= 3) | (sp == t["special"])
    return {
        "any_prize": round(float(any_prize.mean()), 4),
        "distinct_whites": len({w for t in tickets for w in t["whites"]}),
        "distinct_specials": len({t["special"] for t in tickets}),
    }


def consolidate(game, frame, cluster_tickets, cluster_special, records, whites,
                specials, dates, tuned_through):
    """-> the single recommended set + the evidence behind the choice."""
    oos = [r for r in records if dates[r[0]] > tuned_through]
    insample = match_sum_test(records, whites, game["white_max"])
    gate_w = match_sum_test(oos, whites, game["white_max"])
    gate_s = special_test(oos, specials, game["special_max"])
    open_w = gate_w["draws"] >= GATE_MIN_DRAWS and gate_w["p"] < GATE_ALPHA
    open_s = gate_s["draws"] >= GATE_MIN_DRAWS and gate_s["p"] < GATE_ALPHA

    dead_w, dead_s = set(frame["white"]["dead"]), set(frame["special"]["dead"])
    hot_s = frame["special"]["hot"]
    screened = frame["tickets"]           # already honours hot/dead pixels
    if open_w:
        usable = [c for c in cluster_tickets if not dead_w & set(c)]
        tickets = [{"whites": sorted(c), "special": s["special"]}
                   for c, s in zip(usable, screened)]
        tickets += screened[len(tickets):]
        source = "cluster"
    else:
        tickets = [dict(t) for t in screened]
        source = "screened"
    if open_s and not hot_s and cluster_special not in dead_s:
        tickets[0]["special"] = cluster_special

    sm = game["special_max"]
    cluster_raw = [{"whites": sorted(c), "special": cluster_special} for c in cluster_tickets]
    return {
        "source": source,
        "tickets": tickets,
        "gate": {"tuned_through": tuned_through, "min_draws": GATE_MIN_DRAWS,
                 "alpha": GATE_ALPHA, "whites": gate_w, "special": gate_s,
                 "open_whites": open_w, "open_special": open_s},
        "insample": insample,
        "pixels": {"hot": frame["white"]["hot"], "dead": sorted(dead_w),
                   "hot_special": hot_s, "dead_special": sorted(dead_s)},
        "compare": {
            "recommended": prize_profile(tickets, game["white_max"], sm),
            "cluster_raw": prize_profile(cluster_raw, game["white_max"], sm),
        },
    }

#!/usr/bin/env python3
"""
Crowd model — predict the PLAYERS, not the balls.

Every test on this site says the draw itself is unpredictable. What is
predictable is which numbers other people hold, and in California that
matters for every prize: all tiers are pari-mutuel, so a winning ticket's
payout is the tier's pool divided by everyone who matched the same balls.
Unpopular numbers win exactly as often and pay more when they do.

California publishes the number of winners in every tier for every draw
(data/ca_prizes_<game>.csv, see fetch_ca_prizes.py). Two quantities fall out
of those counts with no need to know ticket sales:

  WHITE BALLS.  Among tickets that matched the special ball, the tiers give
  the full distribution of white matches (0,1,2,3,4,5). Its mean is
      M_d = sum over the 5 drawn balls of pi_n,
  where pi_n is the share of tickets carrying ball n (5/N if everyone played
  uniformly). That identity is exact, so pi is a linear regression of M_d on
  which balls were drawn. Fitted by ridge regression shrunk toward uniform,
  with a small "birthday" basis (n <= 12, n <= 31, linear trend) left
  unpenalised; the ridge strength is chosen by leave-one-out CV.

  SPECIAL BALL.  Among tickets with 3+ whites, the fraction that also hold
  the special is the share of tickets carrying that special number (1/S if
  uniform). Pooled per special and shrunk by empirical Bayes.

Like every other analysis here, the model only gets a say once it proves
itself OUT OF SAMPLE:
  whites   walk-forward - before each draw, fit on earlier draws only and
           predict M_d; gate opens at one-sided p < ALPHA on >= MIN_OOS draws.
  special  leave-one-draw-out prediction, p from permuting the special labels.

When a gate is open the recommended tickets are drawn from the least-crowded
slice of screened random tickets. That changes nothing about the odds of
winning - only the expected size of the prize.
"""

import csv
import math
from pathlib import Path

import numpy as np

import picker

ALPHA = 0.01
MIN_OOS = 50
MIN_HOLDERS = 2000          # special-matching tickets needed to use a draw
LAMBDAS = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1e6)
TIERS = ["5s", "5", "4s", "4", "3s", "3", "2s", "1s", "0s"]
RATIO_CLIP = (0.5, 2.0)


# ------------------------------------------------------------------ data

def load_prizes(path):
    """-> list of draws (oldest first) with whites, special, counts, amounts."""
    out = []
    if not Path(path).exists():
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                c = {t: int(r[f"c{t}"]) for t in TIERS}
            except (KeyError, ValueError):
                continue
            a = {}
            for t in TIERS:
                try:
                    a[t] = float(r[f"a{t}"])
                except (KeyError, ValueError):
                    a[t] = None
            out.append({"date": r["draw_date"], "number": int(r["draw_number"]),
                        "whites": [int(r[f"n{i}"]) for i in range(1, 6)],
                        "special": int(r["special"]), "c": c, "a": a})
    out.sort(key=lambda d: (d["date"], d["number"]))
    return out


def observe(draws, white_max, special_max):
    """Per-draw observables. Drops draws with too few winners to measure."""
    obs = []
    for d in draws:
        c = d["c"]
        holders = sum(c[f"{k}s"] for k in range(6))
        if holders < MIN_HOLDERS:
            continue
        mean_match = sum(k * c[f"{k}s"] for k in range(6)) / holders
        with_sp = c["3s"] + c["4s"] + c["5s"]
        three_plus = with_sp + c["3"] + c["4"] + c["5"]
        obs.append({**d, "M": mean_match,
                    "index": mean_match / (25 / white_max),
                    "sp_hits": with_sp, "sp_n": three_plus})
    return obs


# ------------------------------------------------------------ white model

def _basis(N):
    n = np.arange(1, N + 1)
    B = np.column_stack([(n <= 12).astype(float), (n <= 31).astype(float),
                         (n - n.mean()) / N])
    return B - B.mean(axis=0)            # each column sums to zero


def _design(obs, N):
    X = np.zeros((len(obs), N))
    for i, d in enumerate(obs):
        X[i, [w - 1 for w in d["whites"]]] = 1.0
    y = np.array([d["M"] for d in obs]) - 25 / N
    return X, y


def _solve(X, y, N, lam):
    """Generalised ridge: delta = B beta + eps, eps penalised and sum-zero.
    -> (delta, leave-one-out residuals)."""
    B = _basis(N)
    Z = np.hstack([X @ B, X])
    p = B.shape[1]
    P = np.diag([1e-8] * p + [lam] * N)
    v = np.concatenate([np.zeros(p), np.ones(N)])
    A = Z.T @ Z + P + 1e6 * np.outer(v, v)
    Ainv = np.linalg.inv(A)
    theta = Ainv @ Z.T @ y
    h = np.einsum("ij,jk,ik->i", Z, Ainv, Z)
    loo = (y - Z @ theta) / np.clip(1 - h, 1e-6, None)
    return B @ theta[:p] + theta[p:], loo


def fit_white(obs, N):
    X, y = _design(obs, N)
    best = None
    for lam in LAMBDAS:
        delta, loo = _solve(X, y, N, lam)
        sse = float(loo @ loo)
        if best is None or sse < best[0]:
            best = (sse, lam, delta)
    return best[2], best[1]


def _one_sided_r_p(r, n):
    if n < 4 or not np.isfinite(r):
        return 1.0
    from scipy.stats import t as tdist
    r = max(min(r, 0.999999), -0.999999)
    return float(tdist.sf(r * math.sqrt((n - 2) / (1 - r * r)), n - 2))


def walk_forward_white(obs, N):
    """Predict each draw's crowding from earlier draws only."""
    start = max(40, len(obs) // 3)
    pred, act, idx = [], [], []
    for t in range(start, len(obs)):
        delta, _ = fit_white(obs[:t], N)
        pred.append(sum(delta[w - 1] for w in obs[t]["whites"]))
        act.append(obs[t]["M"] - 25 / N)
        idx.append(t)
    n = len(pred)
    if n < 4:
        return {"draws": n, "r": None, "p": 1.0, "skill": None}, [], []
    pred, act = np.array(pred), np.array(act)
    r = float(np.corrcoef(pred, act)[0, 1]) if pred.std() > 0 else 0.0
    skill = 1 - float(((act - pred) ** 2).sum() / (act ** 2).sum())
    return ({"draws": n, "r": round(r, 3), "p": round(_one_sided_r_p(r, n), 5),
             "skill": round(skill, 3)}, list(pred), idx)


# ---------------------------------------------------------- special model

def _special_shrunk(hits, tot, S):
    """Empirical-Bayes ratio (share / uniform share) per special number."""
    q0 = 1 / S
    raw = np.where(tot > 0, hits / np.maximum(tot * q0, 1e-9), 1.0)
    samp = np.where(tot > 0, (1 - q0) / np.maximum(tot * q0, 1e-9), np.inf)
    seen = tot > 0
    if seen.sum() < 3:
        return np.ones(S), 0.0
    tau2 = max(0.0, float(np.var(raw[seen], ddof=1) - np.mean(samp[seen])))
    shrink = np.where(seen, tau2 / (tau2 + samp), 0.0)
    return 1 + shrink * (raw - 1), tau2


def fit_special(obs, S, labels=None):
    hits, tot = np.zeros(S), np.zeros(S)
    for i, d in enumerate(obs):
        s = (labels[i] if labels is not None else d["special"]) - 1
        hits[s] += d["sp_hits"]
        tot[s] += d["sp_n"]
    ratio, tau2 = _special_shrunk(hits, tot, S)
    return ratio, hits, tot, tau2


def _loo_special_r(obs, S, labels):
    """Correlation between each draw's observed special share and the share
    predicted from the OTHER draws that had the same special."""
    _, hits, tot, tau2 = fit_special(obs, S, labels)
    q0 = 1 / S
    pred, act = [], []
    for i, d in enumerate(obs):
        s = labels[i] - 1
        h, n = hits[s] - d["sp_hits"], tot[s] - d["sp_n"]
        if n <= 0 or d["sp_n"] <= 0:
            continue
        samp = (1 - q0) / (n * q0)
        pred.append(1 + tau2 / (tau2 + samp) * (h / (n * q0) - 1))
        act.append(d["sp_hits"] / (d["sp_n"] * q0))
    if len(pred) < 4 or np.std(pred) == 0:
        return 0.0, len(pred)
    return float(np.corrcoef(pred, act)[0, 1]), len(pred)


def validate_special(obs, S, perms=2000, seed=0):
    labels = [d["special"] for d in obs]
    r, n = _loo_special_r(obs, S, labels)
    if n < 4:
        return {"draws": n, "r": None, "p": 1.0}
    rng = np.random.default_rng(seed)
    ge = sum(_loo_special_r(obs, S, list(rng.permutation(labels)))[0] >= r
             for _ in range(perms))
    return {"draws": n, "r": round(r, 3), "p": round((1 + ge) / (perms + 1), 5)}


# ------------------------------------------------------------- payoff view

def _terciles(pairs, label):
    """pairs: (crowd index, prize). -> mean prize in each third by index."""
    pairs = sorted(p for p in pairs if p[1])
    if len(pairs) < 9:
        return None
    k = len(pairs) // 3
    parts = [("least crowded third", pairs[:k]), ("middle third", pairs[k:-k]),
             ("most crowded third", pairs[-k:])]
    return {"basis": label, "rows": [
        {"label": name, "draws": len(ps),
         "index": round(float(np.mean([p[0] for p in ps])), 3),
         "prize": round(float(np.mean([p[1] for p in ps])), 1)} for name, ps in parts]}


# --------------------------------------------------------------- the model

def crowd_block(prize_csv, game):
    """Everything the page shows + what the ticket builder needs."""
    N, S = game["white_max"], game["special_max"]
    obs = observe(load_prizes(prize_csv), N, S)
    if len(obs) < 20:
        return {"available": False, "draws": len(obs),
                "reason": "Not enough California prize tables collected yet "
                          f"({len(obs)} draws; needs 20+)."}

    delta, lam = fit_white(obs, N)
    pi0 = 5 / N
    w_ratio = np.clip(1 + delta / pi0, *RATIO_CLIP)
    oos, pred, idx = walk_forward_white(obs, N)
    w_open = oos["draws"] >= MIN_OOS and oos["p"] < ALPHA

    s_ratio, hits, tot, _ = fit_special(obs, S)
    s_ratio = np.clip(s_ratio, *RATIO_CLIP)
    sval = validate_special(obs, S)
    s_open = sval["draws"] >= MIN_OOS and sval["p"] < ALPHA

    tier = "4"                            # 4 whites, no special: precise amounts
    payoff = {
        "tier": "4 of 5",
        "observed": _terciles([(d["index"], d["a"][tier]) for d in obs],
                              "how crowded the drawn balls turned out to be"),
        "predicted": _terciles(
            [(1 + pred[j] / (25 / N), obs[t]["a"][tier]) for j, t in enumerate(idx)],
            "the model's forecast made before each draw"),
    }
    order = np.argsort(w_ratio)
    return {
        "available": True,
        "draws": len(obs), "first_date": obs[0]["date"], "last_date": obs[-1]["date"],
        "alpha": ALPHA, "min_oos": MIN_OOS,
        "white": {
            "ratio": [round(float(v), 3) for v in w_ratio],
            "ridge": lam, "oos": oos, "gate_open": bool(w_open),
            "least": [int(i) + 1 for i in order[:8]],
            "most": [int(i) + 1 for i in order[::-1][:8]],
            "index_sd": round(float(np.std([d["index"] for d in obs])), 4),
        },
        "special": {
            "ratio": [round(float(v), 3) for v in s_ratio],
            "seen": [int(t > 0) for t in tot],
            "oos": sval, "gate_open": bool(s_open),
        },
        "payoff": payoff,
    }


# ------------------------------------------------------------ ticket builder

def build_tickets(game, draws, frame_white, frame_special, crowd,
                  n_tickets=5, candidates=20000, keep=0.10):
    """Screened random tickets with no white ball repeated across the set and
    a different special on each; confirmed hot/dead pixels are honoured. When
    a crowd gate is open, tickets come from the least-crowded slice."""
    rng = picker.rng
    N, S = game["white_max"], game["special_max"]
    past = {(frozenset(w), s) for _, w, s in draws}
    ok = bool(crowd and crowd.get("available"))
    use_w = ok and crowd["white"]["gate_open"]
    use_s = ok and crowd["special"]["gate_open"]
    lw = np.log(crowd["white"]["ratio"]) if ok else np.zeros(N)
    ls = np.log(crowd["special"]["ratio"]) if ok else np.zeros(S)

    must = sorted(frame_white["hot"],
                  key=lambda n: -frame_white["balls"][n - 1]["z"])[:5]
    pool_w = [v for v in range(1, N + 1)
              if v not in set(frame_white["dead"]) and v not in must]
    cands = []
    for _ in range(candidates if use_w else 4000):
        whites = sorted(must + rng.sample(pool_w, 5 - len(must)))
        if not picker.is_popular_pattern(whites, None, set()):
            cands.append((float(sum(lw[w - 1] for w in whites)), whites))
    cands.sort(key=lambda c: c[0])

    hot_s = sorted(frame_special["hot"],
                   key=lambda n: -frame_special["balls"][n - 1]["z"])
    pool_s = [v for v in range(1, S + 1) if v not in set(frame_special["dead"])]
    if hot_s:
        specials = [hot_s[0]] * n_tickets
    else:
        if use_s:                          # least-held third of the specials
            pool_s = sorted(pool_s, key=lambda v: ls[v - 1])[:max(n_tickets, S // 3)]
        specials = rng.sample(pool_s, min(n_tickets, len(pool_s)))

    def crowd_of(whites, sp):
        # additive (conservative) estimate of other holders vs an average ticket
        return max(0.1, 1 + float(sum(math.expm1(lw[w - 1]) for w in whites))
                   + math.expm1(float(ls[sp - 1])))

    def pick(slice_, want_disjoint):
        slice_ = list(slice_)
        rng.shuffle(slice_)
        tickets, used = [], set()
        for _, whites in slice_:
            if len(tickets) == n_tickets:
                break
            if want_disjoint and used & (set(whites) - set(must)):
                continue
            sp = specials[len(tickets) % len(specials)]
            if (frozenset(whites), sp) in past or any(t["whites"] == whites for t in tickets):
                continue
            used |= set(whites)
            t = {"whites": whites, "special": sp}
            if ok:
                t["crowd"] = round(crowd_of(whites, sp), 3)
            tickets.append(t)
        return tickets, used

    # Least-crowded slice first; widen it until five tickets with no shared
    # white ball fit (small matrices need a wider slice), then relax disjoint.
    fracs = (keep, 0.2, 0.35, 0.5, 1.0) if use_w else (1.0,)
    tickets, used = [], set()
    for want_disjoint in (True, False):
        for frac in fracs:
            for _ in range(20):
                tickets, used = pick(cands[:max(50, int(len(cands) * frac))], want_disjoint)
                if len(tickets) == n_tickets:
                    break
            if len(tickets) == n_tickets:
                break
        if len(tickets) == n_tickets:
            break
    summary = {"white_used": bool(use_w), "special_used": bool(use_s),
               "disjoint": len(used) == len(must) + n_tickets * (5 - len(must))}
    if ok:
        summary["crowd_mean"] = round(float(np.mean([t["crowd"] for t in tickets])), 3)
    return tickets, summary

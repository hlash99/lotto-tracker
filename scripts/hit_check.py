#!/usr/bin/env python3
"""
Score the dashboard's picks against the draw they were made for.

Every refresh regenerates `recommended.tickets` (and, for Powerball,
`cluster.tickets`) for the NEXT draw after `latest`. When a later refresh sees
a newer `latest`, the previous run's picks are scored against the first draw
after the previous `latest` and recorded as `last_result`, and prepended to a
rolling `ledger` (newest first). The hub page reads `last_result.significant`
to decide whether to show a banner.

"Significant" = a real prize, not the $4-$7 consolation tiers:
  Powerball:       4+ whites, or 3 whites + Powerball  ($100+)
  SuperLotto Plus: 4+ whites, or 3 whites + Mega

CLI (backfill the ledger from git history of data.json, writes nothing unless
--write):  python scripts/hit_check.py backfill [--write]
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
DATA = ROOT / "data"
GIT = "/usr/bin/git" if Path("/usr/bin/git").exists() else "git"

LEDGER_MAX = 100
SOURCES = ("recommended", "cluster")
SPECIAL_TAG = {"powerball": "PB", "superlotto": "M"}
RESULTS_CSV = {"powerball": "official_results.csv",
               "superlotto": "superlotto_results.csv"}


# ---------------------------------------------------------------- scoring

def tier_label(game_key, white_hits, special_hit):
    """Prize tier string ("3+PB", "4", "5+PB (JACKPOT)", "3+M", "PB" ...),
    or None when the ticket wins nothing."""
    tag = SPECIAL_TAG[game_key]
    if white_hits == 5 and special_hit:
        return f"5+{tag} (JACKPOT)"
    if special_hit:
        return f"{white_hits}+{tag}" if white_hits else tag
    if white_hits >= 3:
        return str(white_hits)
    return None


def is_significant(white_hits, special_hit):
    """Same bar for both games: 4+ whites, or 3 whites + special."""
    return white_hits >= 4 or (white_hits == 3 and bool(special_hit))


def tier_rank(white_hits, special_hit):
    """Ordering for picking the best ticket (higher = better prize). Follows
    the published prize ladder of both games: specials break ties within a
    white count, and 3 whites outranks 2+special."""
    if tier_label("powerball", white_hits, special_hit) is None:
        return 0
    return 1 + 2 * white_hits + (1 if special_hit else 0)


def score_ticket(game_key, ticket, whites, special):
    th = {int(v) for v in ticket["whites"]}
    wh = len(th & {int(v) for v in whites})
    sh = ticket.get("special") is not None and int(ticket["special"]) == int(special)
    return {"whites": sorted(th), "special": ticket.get("special"),
            "white_hits": wh, "special_hit": sh,
            "tier": tier_label(game_key, wh, sh),
            "significant": is_significant(wh, sh)}


def pick_sets(game_block):
    """[(source, [tickets])] present in one game's data.json block."""
    out = []
    for src in SOURCES:
        blk = (game_block or {}).get(src) or {}
        tix = blk.get("tickets")
        if isinstance(tix, list) and tix:
            # Cluster tickets carry whites only; their special ball is the
            # block-level `special_pick` (what the hub/dashboard display).
            sp = blk.get("special_pick")
            tix = [t if t.get("special") is not None or sp is None
                   else {**t, "special": sp} for t in tix]
            out.append((src, tix))
    return out


def score_picks(game_key, game_block, draw, picks_from):
    """Score every ticket in `game_block` against `draw` ({date, whites,
    special}). Returns a last_result dict, or None if there were no picks."""
    sets = pick_sets(game_block)
    if not sets:
        return None
    tickets, best, best_key = [], None, None
    for src, tix in sets:
        for i, t in enumerate(tix):
            s = score_ticket(game_key, t, draw["whites"], draw["special"])
            tickets.append({"source": src, "index": i, **s})
            key = (tier_rank(s["white_hits"], s["special_hit"]), s["white_hits"])
            if best_key is None or key > best_key:
                best_key = key
                best = {"tier": s["tier"], "source": src, "ticket": i,
                        "white_hits": s["white_hits"], "special_hit": s["special_hit"]}
    return {"draw_date": draw["date"],
            "whites": sorted(int(v) for v in draw["whites"]),
            "special": int(draw["special"]),
            "scored_against_picks_from": picks_from,
            "tickets": tickets,
            "best": best,
            "significant": any(t["significant"] for t in tickets)}


# ---------------------------------------------------------------- draws

def load_draws(game_key, data_dir=DATA):
    """[(date, [whites], special)] sorted by date, from the repo's CSVs."""
    path = Path(data_dir) / RESULTS_CSV[game_key]
    sp_col = "powerball" if game_key == "powerball" else "mega"
    out = []
    try:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    out.append((r["draw_date"], [int(r[f"n{i}"]) for i in range(1, 6)],
                                int(r[sp_col])))
                except (KeyError, TypeError, ValueError):
                    continue
    except FileNotFoundError:
        pass
    out.sort(key=lambda d: d[0])
    return out


def first_draw_after(date, draws, fallback=None):
    """The first official draw strictly after `date`; `fallback` (a
    {date, whites, special} dict) if the CSV doesn't have one."""
    for d, ws, sp in draws:
        if d > date:
            return {"date": d, "whites": ws, "special": sp}
    return fallback


# ---------------------------------------------------------------- update hook

def merge_ledger(entry, ledger):
    """Prepend `entry`, drop any older entry for the same draw, newest first."""
    rest = [e for e in (ledger or []) if e.get("draw_date") != entry["draw_date"]]
    out = [entry] + rest
    out.sort(key=lambda e: e.get("draw_date", ""), reverse=True)
    return out[:LEDGER_MAX]


def load_previous(path):
    """Previous data.json, or None if missing / unparseable (e.g. a committed
    merge conflict). Falls back to the last parseable committed version when
    git history is available (CI checkouts are shallow, so usually not)."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        pass
    for sha, _ in _history(path.parent, limit=20):
        d = _show(path.parent, sha)
        if d is not None:
            return d
    return None


def attach_results(out, prev, data_dir=DATA):
    """Fill out[game]['last_result'] / ['ledger'] for both games, given the
    new data.json dict `out` and the previous one `prev` (or None)."""
    prev = prev or {}
    for key in ("powerball", "superlotto"):
        new_g, old_g = out.get(key) or {}, prev.get(key) or {}
        last, ledger = old_g.get("last_result"), list(old_g.get("ledger") or [])
        new_date = (new_g.get("latest") or {}).get("date")
        old_date = (old_g.get("latest") or {}).get("date")
        if new_date and old_date and new_date > old_date:
            draw = first_draw_after(old_date, load_draws(key, data_dir),
                                    fallback=new_g["latest"])
            if draw is None or draw["date"] > new_date:
                draw = new_g["latest"]
            res = score_picks(key, old_g, draw, prev.get("updated"))
            if res is not None:
                last = res
                ledger = merge_ledger(res, ledger)
                flag = "SIGNIFICANT" if res["significant"] else "no big hit"
                print(f"[hit_check] {key} {res['draw_date']}: best "
                      f"{res['best']['tier']} ({flag})", flush=True)
        if last is not None:
            new_g["last_result"] = last
        new_g["ledger"] = ledger[:LEDGER_MAX]
    return out


# ---------------------------------------------------------------- backfill

def _history(repo, limit=None):
    """[(sha, commit ISO time)] for data.json, newest first."""
    cmd = [GIT, "-C", str(repo), "log", "--format=%H %cI"]
    if limit:
        cmd.append(f"-n{limit}")
    cmd += ["--", "data.json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return []
    return [tuple(l.split()) for l in r.stdout.splitlines() if l.strip()]


def _show(repo, sha):
    try:
        r = subprocess.run([GIT, "-C", str(repo), "show", f"{sha}:data.json"],
                           capture_output=True, text=True, check=True)
        return json.loads(r.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


def backfill(repo=ROOT, data_dir=DATA):
    """Rebuild each game's ledger from git history: for each draw, score the
    LAST pick set committed whose latest.date precedes it (i.e. the final
    picks shown for that draw). Returns ({game: ledger}, stats)."""
    hist = list(reversed(_history(repo)))           # oldest first
    stats = {"commits": len(hist), "unparseable": 0}
    snaps = []
    for sha, when in hist:
        d = _show(repo, sha)
        if d is None:
            stats["unparseable"] += 1
            continue
        snaps.append((sha, when, d))
    ledgers = {}
    for key in ("powerball", "superlotto"):
        draws = load_draws(key, data_dir)
        by_draw = {}                                  # target date -> entry
        for sha, when, d in snaps:                    # oldest -> newest
            g = d.get(key) or {}
            ld = (g.get("latest") or {}).get("date")
            if not ld or not pick_sets(g):
                continue
            draw = first_draw_after(ld, draws)
            if draw is None:
                continue                              # not drawn yet
            res = score_picks(key, g, draw, d.get("updated") or when)
            res["_sha"] = sha[:7]
            by_draw[draw["date"]] = res               # later commit wins
        led = sorted(by_draw.values(), key=lambda e: e["draw_date"], reverse=True)
        ledgers[key] = led
        stats[key] = {"draws_scored": len(led),
                      "significant": [(e["draw_date"], e["best"]["tier"],
                                       e["best"]["source"], e["_sha"])
                                      for e in led if e["significant"]],
                      "tier_counts": _tier_counts(led)}
    return ledgers, stats


def _tier_counts(ledger):
    from collections import Counter
    return dict(Counter((e["best"]["tier"] or "none") for e in ledger))


def main(argv):
    if not argv or argv[0] != "backfill":
        print(__doc__)
        return 1
    ledgers, stats = backfill()
    print(json.dumps(stats, indent=1))
    if "--write" in argv:
        path = ROOT / "data.json"
        with open(path) as f:
            cur = json.load(f)
        for key, led in ledgers.items():
            g = cur.get(key) or {}
            merged = {e["draw_date"]: e for e in led}
            for e in g.get("ledger") or []:          # live entries win
                merged[e["draw_date"]] = e
            clean = [{k: v for k, v in e.items() if not k.startswith("_")}
                     for e in merged.values()]
            clean.sort(key=lambda e: e["draw_date"], reverse=True)
            g["ledger"] = clean[:LEDGER_MAX]
            if clean and "last_result" not in g:
                g["last_result"] = clean[0]
        with open(path, "w") as f:
            json.dump(cur, f, indent=1)
        print(f"wrote ledgers into {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

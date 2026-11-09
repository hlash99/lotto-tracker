#!/usr/bin/env python3
"""
Fetch California's per-draw prize table (winner COUNT and prize AMOUNT for
every tier) for Powerball and SuperLotto Plus, and accumulate it in
data/ca_prizes_<game>.csv.

Why: California pays every prize tier pari-mutuel, so the number of winners
in each tier is published for every draw. Those counts reveal which numbers
OTHER PLAYERS hold (see crowd.py) — the one thing about a lottery that is
actually predictable.

The CA API only exposes ~12 months, so the CSV accumulates across runs
(existing rows are kept; API rows win on collision).

Columns: draw_date, draw_number, n1..n5, special,
         c5s,c5,c4s,c4,c3s,c3,c2s,c1s,c0s   winning tickets per tier
         a5s,a5,a4s,a4,a3s,a3,a2s,a1s,a0s   prize paid per winner ($)
(tier "3s" = 3 whites + special, "3" = 3 whites only, "0s" = special only)

Usage: python3 fetch_ca_prizes.py [--out-dir data] [--game powerball|superlotto|all]
"""

import argparse
import csv
import json
import re
import ssl
import sys
import urllib.request
from pathlib import Path

API = "https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults/{gid}/{page}/{size}"
PAGE_SIZE = 50  # API returns null for page sizes above 50
GAMES = {
    "powerball": {"id": 12, "white_max": 69, "special_max": 26},
    "superlotto": {"id": 8, "white_max": 47, "special_max": 27},
}
TIERS = ["5s", "5", "4s", "4", "3s", "3", "2s", "1s", "0s"]
HEADER = (["draw_date", "draw_number"] + [f"n{i}" for i in range(1, 6)] + ["special"]
          + [f"c{t}" for t in TIERS] + [f"a{t}" for t in TIERS])


def ssl_context():
    ctx = ssl.create_default_context()
    try:
        ctx.load_default_certs()
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def get_json(url, ctx):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return json.load(resp)


def tier_key(description):
    """'5 + Mega' -> '5s', '4' -> '4', 'Powerball' -> '0s'; None if unknown."""
    d = (description or "").strip().lower()
    m = re.match(r"^(\d)\s*(\+\s*\w+)?$", d)
    if m:
        return m.group(1) + ("s" if m.group(2) else "")
    if d in ("mega", "powerball", "mega number"):
        return "0s"
    return None


def parse_draw(draw, game):
    """-> CSV row dict, or None when the draw or its prize table is unusable."""
    wn = draw.get("WinningNumbers") or {}
    whites, special = [], None
    for k in sorted(wn, key=lambda s: int(s)):
        num = int(wn[k]["Number"])
        if wn[k].get("IsSpecial"):
            special = num
        else:
            whites.append(num)
    if (len(whites) != 5 or len(set(whites)) != 5 or special is None
            or not all(1 <= w <= game["white_max"] for w in whites)
            or not 1 <= special <= game["special_max"]):
        return None
    prizes = draw.get("Prizes") or {}
    prizes = prizes.values() if isinstance(prizes, dict) else prizes
    counts, amounts = {}, {}
    for p in prizes:
        key = tier_key(p.get("PrizeTypeDescription"))
        if key is None or p.get("Count") is None:
            continue
        counts[key] = int(p["Count"])
        amounts[key] = p.get("Amount")
    if any(t not in counts for t in TIERS):
        return None
    row = {"draw_date": (draw.get("DrawDate") or "")[:10],
           "draw_number": draw.get("DrawNumber"), "special": special}
    for i, w in enumerate(sorted(whites), 1):
        row[f"n{i}"] = w
    for t in TIERS:
        row[f"c{t}"] = counts[t]
        a = amounts.get(t)
        row[f"a{t}"] = "" if a is None else int(round(float(a)))
    return row


def fetch_game(key, out_dir, ctx):
    game = GAMES[key]
    first = get_json(API.format(gid=game["id"], page=1, size=PAGE_SIZE), ctx)
    if not first:
        raise RuntimeError("CA Lottery API returned no data")
    pages = (first["TotalPreviousDraws"] + PAGE_SIZE - 1) // PAGE_SIZE
    rows, skipped = {}, 0
    for page in range(1, pages + 1):
        data = first if page == 1 else get_json(
            API.format(gid=game["id"], page=page, size=PAGE_SIZE), ctx)
        if not data:
            break
        for draw in data.get("PreviousDraws") or []:
            row = parse_draw(draw, game)
            if row is None:
                skipped += 1
            else:
                rows[int(row["draw_number"])] = row
    fetched = len(rows)

    path = Path(out_dir) / f"ca_prizes_{key}.csv"
    if path.exists():
        with open(path, newline="") as f:
            for old in csv.DictReader(f):
                rows.setdefault(int(old["draw_number"]), old)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER)
        w.writeheader()
        for dn in sorted(rows):
            w.writerow({k: rows[dn].get(k, "") for k in HEADER})
    print(f"{key}: {len(rows)} draws with prize tables in {path.name} "
          f"({fetched} from API this run"
          + (f", {skipped} skipped" if skipped else "") + ")")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--game", choices=sorted(GAMES) + ["all"], default="all")
    args = ap.parse_args()
    ctx = ssl_context()
    failed = False
    for key in (GAMES if args.game == "all" else [args.game]):
        try:
            fetch_game(key, args.out_dir, ctx)
        except Exception as e:      # keep the other game + existing CSV usable
            failed = True
            print(f"{key}: prize fetch FAILED ({e})", file=sys.stderr)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

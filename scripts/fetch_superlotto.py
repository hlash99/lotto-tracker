#!/usr/bin/env python3
"""
Fetch the complete SuperLotto Plus draw history from the California Lottery
API and write it to a clean CSV.

SuperLotto Plus: 5 white balls from 1-47 + Mega number from 1-27
(format in effect since June 2000).

Output columns: draw_date, draw_number, n1..n5, mega
NOTE: the CA API returns white balls in ascending order (verified across the
full history), so drawn order is NOT available from this source. The CA
Lottery YouTube channel is the only known drawn-order source for SuperLotto.

Usage:
  python3 fetch_superlotto.py [--out superlotto_results.csv]
"""

import argparse
import csv
import json
import ssl
import sys
import urllib.request
from pathlib import Path

API = "https://www.calottery.com/api/DrawGameApi/DrawGamePastDrawResults/8/{page}/{size}"
PAGE_SIZE = 50  # API returns null for page sizes above 50
WHITE_MAX, MEGA_MAX = 47, 27


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


def parse_draw(draw):
    """-> (date, draw_number, [5 whites], mega) or None if malformed."""
    date = draw.get("DrawDate", "")[:10]
    wn = draw.get("WinningNumbers") or {}
    whites, mega = [], None
    for k in sorted(wn, key=lambda s: int(s)):
        entry = wn[k]
        num = int(entry["Number"])
        if entry.get("IsSpecial"):
            mega = num
        else:
            whites.append(num)
    if (len(whites) != 5 or mega is None or len(set(whites)) != 5
            or not all(1 <= w <= WHITE_MAX for w in whites)
            or not 1 <= mega <= MEGA_MAX):
        return None
    return date, draw.get("DrawNumber"), whites, mega


def main():
    ap = argparse.ArgumentParser(description="Download SuperLotto Plus history to CSV")
    ap.add_argument("--out", default="superlotto_results.csv")
    args = ap.parse_args()

    ctx = ssl_context()
    print("Downloading SuperLotto Plus history from CA Lottery...", file=sys.stderr)

    first = get_json(API.format(page=1, size=PAGE_SIZE), ctx)
    if not first:
        raise SystemExit("CA Lottery API returned no data (null response)")
    total = first["TotalPreviousDraws"]
    pages = (total + PAGE_SIZE - 1) // PAGE_SIZE

    rows, skipped = [], 0
    ascending = 0
    for page in range(1, pages + 1):
        data = first if page == 1 else get_json(API.format(page=page, size=PAGE_SIZE), ctx)
        if not data:
            print(f"  page {page}: null response, stopping early", file=sys.stderr)
            break
        for draw in data.get("PreviousDraws") or []:
            parsed = parse_draw(draw)
            if parsed is None:
                skipped += 1
                continue
            date, dn, whites, mega = parsed
            if whites == sorted(whites):
                ascending += 1
            rows.append([date, dn] + sorted(whites) + [mega])
        if page % 5 == 0 or page == pages:
            print(f"  fetched page {page}/{pages} ({len(rows)} draws)", file=sys.stderr)

    fetched_count = len(rows)

    # Merge with any existing CSV: the API only exposes ~12 months, so the
    # local file accumulates history across runs.
    if Path(args.out).exists():
        with open(args.out, newline="") as f:
            for old in csv.DictReader(f):
                rows.append([old["draw_date"], int(old["draw_number"])]
                            + [int(old[f"n{i}"]) for i in range(1, 6)]
                            + [int(old["mega"])])

    # de-dupe by draw number, oldest first
    seen = set()
    unique = []
    for r in sorted(rows, key=lambda r: (r[0], r[1] or 0)):
        if r[1] in seen:
            continue
        seen.add(r[1])
        unique.append(r)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_date", "draw_number", "n1", "n2", "n3", "n4", "n5", "mega"])
        w.writerows(unique)

    print(f"Wrote {len(unique)} draws to {args.out}"
          + (f" ({skipped} malformed skipped)" if skipped else ""))
    if unique:
        print(f"Date range: {unique[0][0]} to {unique[-1][0]}")
    print(f"API rows already in ascending order: {ascending}/{fetched_count} "
          f"({100 * ascending / max(1, fetched_count):.1f}%) — "
          + ("sorted feed, drawn order NOT preserved"
             if ascending > 0.5 * fetched_count else "looks like drawn order!"))


if __name__ == "__main__":
    main()

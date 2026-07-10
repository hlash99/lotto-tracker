#!/usr/bin/env python3
"""
Backfill deep SuperLotto Plus / Fantasy 5 history from lottery.net year pages.

The CA Lottery API only exposes ~12 months, capping the signature analysis at
~106 draws (SLP) / ~182 (F5). This scraper pulls the year archives, validates
every row against game rules, CROSS-CHECKS the overlap against the official
CA-API rows already in the CSV (must agree 100%, else abort), and merges.

Usage:
  python3 backfill_ca_history.py superlotto [--from-year 2000]
  python3 backfill_ca_history.py fantasy5  [--from-year 2011]
"""

import argparse
import csv
import re
import ssl
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

GAMES = {
    "superlotto": {
        "slug": "superlotto-plus", "csv": "superlotto_results.csv",
        "white_max": 47, "special_max": 27, "special_col": "mega",
        "start": "2000-06-04",  # 5/47+27 matrix begins
        "default_from": 2000,
    },
    "fantasy5": {
        "slug": "fantasy-5", "csv": "fantasy5_results.csv",
        "white_max": 39, "special_max": None, "special_col": None,
        "start": "1992-01-01",
        "default_from": 2011,
    },
}

ROW_RE = re.compile(
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December)\s+(\d{1,2}),\s+(\d{4})\s*</a>\s*</td>\s*"
    r"<td[^>]*>(\d+)</td>.*?<ul[^>]*>(.*?)</ul>", re.S)
BALL_RE = re.compile(r'<li class="(ball|mega-ball)">\s*(\d+)', re.S)
MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}


def ssl_context():
    ctx = ssl.create_default_context()
    try:
        ctx.load_default_certs()
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def fetch_year(slug, year, ctx):
    url = f"https://www.lottery.net/california/{slug}/numbers/{year}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_year(html, game):
    rows = []
    for m in ROW_RE.finditer(html):
        mon, day, year, drawno, ul = m.groups()
        date = f"{int(year):04d}-{MONTHS[mon]:02d}-{int(day):02d}"
        whites, special = [], None
        for cls, num in BALL_RE.findall(ul):
            if cls == "ball":
                whites.append(int(num))
            else:
                special = int(num)
        ok = (len(whites) == 5 and len(set(whites)) == 5
              and all(1 <= w <= game["white_max"] for w in whites)
              and date >= game["start"])
        if game["special_max"]:
            ok = ok and special is not None and 1 <= special <= game["special_max"]
        if ok:
            rows.append((date, int(drawno), sorted(whites), special))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("game", choices=sorted(GAMES))
    ap.add_argument("--from-year", type=int, default=None)
    args = ap.parse_args()
    game = GAMES[args.game]
    from_year = args.from_year or game["default_from"]
    ctx = ssl_context()

    scraped = []
    for year in range(from_year, datetime.now().year + 1):
        try:
            html = fetch_year(game["slug"], year, ctx)
            rows = parse_year(html, game)
            scraped.extend(rows)
            print(f"  {year}: {len(rows)} draws", file=sys.stderr)
        except Exception as e:
            print(f"  {year}: FAILED ({e})", file=sys.stderr)
        time.sleep(1.2)

    if not scraped:
        raise SystemExit("Nothing scraped — page structure may have changed")

    # load existing official-API rows for cross-validation + merge
    existing = {}
    csv_path = Path(game["csv"])
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                whites = sorted(int(row[f"n{i}"]) for i in range(1, 6))
                sp = int(row[game["special_col"]]) if game["special_col"] else None
                existing[int(row["draw_number"])] = (row["draw_date"], whites, sp)

    # cross-check overlap: scraped rows must agree with official API rows
    overlap = agree = 0
    for date, dn, whites, sp in scraped:
        if dn in existing:
            overlap += 1
            od, ow, osp = existing[dn]
            if ow == whites and osp == sp and od == date:
                agree += 1
            else:
                print(f"  MISMATCH draw {dn}: scraped {date} {whites}+{sp} "
                      f"vs official {od} {ow}+{osp}", file=sys.stderr)
    if overlap:
        print(f"Cross-validation vs official CA API: {agree}/{overlap} agree")
        if agree < overlap:
            raise SystemExit("ABORT: scraped data disagrees with official rows — not merging")
    else:
        print("WARNING: no overlap with official rows to validate against")

    # merge (official rows win on collision) and rewrite
    merged = {dn: (date, whites, sp) for date, dn, whites, sp in scraped}
    merged.update(existing)
    header = ["draw_date", "draw_number", "n1", "n2", "n3", "n4", "n5"]
    if game["special_col"]:
        header.append(game["special_col"])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for dn in sorted(merged):
            date, whites, sp = merged[dn]
            row = [date, dn] + whites
            if game["special_col"]:
                row.append(sp)
            w.writerow(row)
    dates = [v[0] for v in merged.values()]
    print(f"Merged archive: {len(merged)} draws ({min(dates)} to {max(dates)}) -> {csv_path}")


if __name__ == "__main__":
    main()

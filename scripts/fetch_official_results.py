#!/usr/bin/env python3
"""
Fetch the complete official Powerball draw history from the NY Open Data API
(https://data.ny.gov/resource/d6yy-54nr.json) and write it to a clean CSV.

This replaces the YouTube -> frame grab -> crop -> OCR pipeline for data
collection: same numbers, zero OCR errors, includes Double Play, and covers
the full history back to 2010.

Usage:
  python3 fetch_official_results.py [--out official_results.csv]

No dependencies beyond the standard library.
"""

import argparse
import csv
import json
import ssl
import sys
import urllib.request
from pathlib import Path

API_URL = "https://data.ny.gov/resource/d6yy-54nr.json"
PAGE_SIZE = 5000


def ssl_context():
    """python.org Python on macOS often lacks root CAs; fall back to certifi."""
    ctx = ssl.create_default_context()
    try:
        ctx.load_default_certs()
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


# Game matrix changed over the years; validate against the rules in effect.
# (start_date, white_max, pb_max)
MATRIX_ERAS = [
    ("2015-10-07", 69, 26),  # current matrix
    ("2012-01-15", 59, 35),
    ("1900-01-01", 59, 39),
]
CURRENT_MATRIX_START = MATRIX_ERAS[0][0]


def rules_for(date):
    for start, wmax, pmax in MATRIX_ERAS:
        if date >= start:
            return wmax, pmax
    return MATRIX_ERAS[-1][1:]


def parse_numbers(s, date="2100-01-01"):
    """'17 44 63 66 67 04' -> ([17, 44, 63, 66, 67], 4). Last number is the Powerball."""
    wmax, pmax = rules_for(date)
    parts = [int(p) for p in s.split()]
    if len(parts) != 6:
        raise ValueError(f"expected 6 numbers, got {len(parts)}: {s!r}")
    whites, pb = sorted(parts[:5]), parts[5]
    if not all(1 <= n <= wmax for n in whites) or len(set(whites)) != 5:
        raise ValueError(f"invalid white balls: {whites}")
    if not 1 <= pb <= pmax:
        raise ValueError(f"invalid powerball: {pb}")
    return whites, pb


def fetch_all():
    ctx = ssl_context()
    rows = []
    offset = 0
    while True:
        url = (f"{API_URL}?$order=draw_date%20ASC&$limit={PAGE_SIZE}"
               f"&$offset={offset}")
        with urllib.request.urlopen(url, timeout=60, context=ctx) as resp:
            page = json.load(resp)
        if not page:
            break
        rows.extend(page)
        offset += len(page)
        print(f"  fetched {len(rows)} draws...", file=sys.stderr)
        if len(page) < PAGE_SIZE:
            break
    return rows


def main():
    ap = argparse.ArgumentParser(description="Download official Powerball history to CSV")
    ap.add_argument("--out", default="official_results.csv", help="Output CSV path")
    args = ap.parse_args()

    print("Downloading official Powerball history...", file=sys.stderr)
    raw = fetch_all()

    out_path = Path(args.out)
    written = skipped = 0
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_date", "n1", "n2", "n3", "n4", "n5", "powerball",
                    "multiplier", "dp1", "dp2", "dp3", "dp4", "dp5", "dp_powerball"])
        for rec in raw:
            date = rec["draw_date"][:10]  # YYYY-MM-DD
            try:
                whites, pb = parse_numbers(rec["winning_numbers"], date)
            except (ValueError, KeyError) as e:
                print(f"  skipping {date}: {e}", file=sys.stderr)
                skipped += 1
                continue
            dp = [""] * 6
            if rec.get("double_play_winning_numbers"):
                try:
                    dpw, dppb = parse_numbers(rec["double_play_winning_numbers"], date)
                    dp = dpw + [dppb]
                except ValueError:
                    pass
            w.writerow([date] + whites + [pb, rec.get("multiplier", "")] + dp)
            written += 1

    print(f"Wrote {written} draws to {out_path}"
          + (f" ({skipped} skipped)" if skipped else ""))
    if written:
        print(f"Date range: {raw[0]['draw_date'][:10]} to {raw[-1]['draw_date'][:10]}")


if __name__ == "__main__":
    main()

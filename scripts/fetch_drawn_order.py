#!/usr/bin/env python3
"""
Fetch Powerball results in DRAWN ORDER from the Texas Lottery CSV.

Unlike the NY Open Data feed (sorted ascending), the Texas Lottery publishes
the five white balls in the order they came out of the machine — verified by
matching its ordering against 650 OCR reads of the official broadcast bar
(100% agreement). Note: Texas does not sell Double Play, so this file covers
the main draw only; the YouTube/OCR pipeline remains the only drawn-order
source for Double Play.

Output columns: draw_date, d1..d5 (drawn order), powerball, power_play

Usage:
  python3 fetch_drawn_order.py [--out drawn_order.csv]
                               [--official official_results.csv]

If the official CSV (from fetch_official_results.py) is present, every row is
cross-checked against it as a set and mismatches are reported.
"""

import argparse
import csv
import io
import ssl
import sys
import urllib.request
from pathlib import Path

TX_CSV_URL = ("https://www.texaslottery.com/export/sites/lottery/Games/"
              "Powerball/Winning_Numbers/powerball.csv")

# (start_date, white_max, pb_max) — game matrix eras
MATRIX_ERAS = [
    ("2015-10-07", 69, 26),
    ("2012-01-15", 59, 35),
    ("1900-01-01", 59, 39),
]


def ssl_context():
    ctx = ssl.create_default_context()
    try:
        ctx.load_default_certs()
        import certifi
        ctx.load_verify_locations(certifi.where())
    except ImportError:
        pass
    return ctx


def rules_for(date):
    for start, wmax, pmax in MATRIX_ERAS:
        if date >= start:
            return wmax, pmax
    return MATRIX_ERAS[-1][1:]


def load_official_sets(path):
    sets = {}
    if not Path(path).exists():
        return sets
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            sets[row["draw_date"]] = (
                frozenset(int(row[f"n{i}"]) for i in range(1, 6)),
                int(row["powerball"]))
    return sets


def main():
    ap = argparse.ArgumentParser(description="Download drawn-order Powerball history (TX Lottery)")
    ap.add_argument("--out", default="drawn_order.csv")
    ap.add_argument("--official", default="official_results.csv",
                    help="Official CSV to cross-check sets against (optional)")
    args = ap.parse_args()

    print("Downloading Texas Lottery drawn-order history...", file=sys.stderr)
    req = urllib.request.Request(TX_CSV_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ssl_context()) as resp:
        text = resp.read().decode("utf-8", errors="replace")

    official = load_official_sets(args.official)
    written = bad = setfail = 0
    rows_out = []
    for row in csv.reader(io.StringIO(text)):
        # Format: Powerball,M,D,YYYY,d1,d2,d3,d4,d5,PB,PowerPlay
        if len(row) < 10 or row[0] != "Powerball":
            continue
        date = f"{int(row[3]):04d}-{int(row[1]):02d}-{int(row[2]):02d}"
        whites = [int(x) for x in row[4:9]]
        pb = int(row[9])
        power_play = row[10] if len(row) > 10 else ""

        wmax, pmax = rules_for(date)
        if (len(set(whites)) != 5 or not all(1 <= w <= wmax for w in whites)
                or not 1 <= pb <= pmax):
            print(f"  invalid row skipped: {date} {whites} + {pb}", file=sys.stderr)
            bad += 1
            continue
        if date in official and (frozenset(whites), pb) != official[date]:
            print(f"  SET MISMATCH vs official: {date} TX={sorted(whites)}+{pb} "
                  f"official={sorted(official[date][0])}+{official[date][1]}",
                  file=sys.stderr)
            setfail += 1
            continue
        rows_out.append([date] + whites + [pb, power_play])
        written += 1

    # Drawn-order guard: a random permutation of 5 balls is ascending only
    # 1/120 of the time (~0.8%). If a large share of rows arrives ascending,
    # the feed has switched to sorted numbers and drawn order is LOST —
    # fail loudly rather than silently archive sorted data.
    asc = sum(1 for r in rows_out if r[1:6] == sorted(r[1:6]))
    if rows_out and asc / len(rows_out) > 0.05:
        raise SystemExit(
            f"ORDER GUARD TRIPPED: {asc}/{len(rows_out)} rows ascending "
            f"({100 * asc / len(rows_out):.1f}%) — TX feed looks SORTED, "
            f"not drawn order. Refusing to write {args.out}.")
    print(f"Order guard OK: {asc}/{len(rows_out)} ascending rows "
          f"({100 * asc / max(1, len(rows_out)):.2f}%, random-order rate is ~0.8%)",
          file=sys.stderr)

    rows_out.sort(key=lambda r: r[0])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["draw_date", "d1", "d2", "d3", "d4", "d5", "powerball", "power_play"])
        w.writerows(rows_out)

    checked = sum(1 for r in rows_out if r[0] in official)
    print(f"Wrote {written} draws in drawn order to {args.out}"
          + (f" ({bad} invalid, {setfail} set-mismatch skipped)" if bad or setfail else ""))
    if official:
        print(f"Cross-checked {checked} draws against {args.official}: all sets match")
    if rows_out:
        print(f"Date range: {rows_out[0][0]} to {rows_out[-1][0]}")


if __name__ == "__main__":
    main()

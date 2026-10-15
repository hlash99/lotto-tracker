# 🎱 lotto-tracker

Live at **https://hlash99.github.io/lotto-tracker/**

Auto-updating Powerball + SuperLotto Plus tracker with server-side analysis.
GitHub Actions refreshes twice daily (10pm PT & 6am PT — after each game's
draw nights), commits fresh data, and GitHub Pages serves the result.

## What it tracks

- **Powerball** — full official history (NY Open Data) *plus drawn order*
  (Texas Lottery feed, which preserves the sequence the balls came out;
  verified against 650 OCR reads of the broadcast bar, 100% agreement).
- **SuperLotto Plus** — CA Lottery API. The API only exposes ~12 months, so
  `data/superlotto_results.csv` accumulates history across runs. CA publishes
  numbers sorted; drawn order is not available from any official feed.

## Analysis (recomputed every refresh)

- Hot/cold frequency + chi-square goodness-of-fit vs uniform, per game.
- **Order-dependence permutation test** (Powerball): does ball *i* influence
  ball *i+1* within a draw? Order is shuffled within each draw 5,000× to build
  the null, so number frequencies and without-replacement effects cancel out.
- **Stacked-frame hot-pixel check** (`scripts/stacked_frame.py`, both games):
  the astrophotography dark-frame idea — stack every draw and look for balls
  that stick out. A ball counts as a hot (or dead) pixel only if it clears a
  family-wise bar (max |z| over 2,000 simulated fair histories) *and* is off
  in the same direction in both independent halves of history. Also reports a
  fixed-pattern test (do halves / odd-vs-even draws agree on which balls run
  hot?), the smallest bias detectable at the current depth, and a
  recommendation + tickets built only from confirmed pixels.
- **One recommended ticket set per game** (`scripts/consolidated.py`):
  confirmed hot/dead pixels are always applied; the cluster method's tickets
  are used only once its *out-of-sample* record (draws after the 2026-10-11
  tuning) beats chance at p < 0.01 over ≥ 100 draws, on an exact
  overlap-aware test (total white matches across the 5 tickets, with the
  null mean/variance computed from how many tickets share each ball).
  Otherwise: screened random tickets with a distinct special ball each —
  ~20% chance of some prize per draw vs ~4% for the cluster tickets, which
  share most balls and one PB.
- Screened quick picks: uniform random tickets, re-drawn if they match
  over-played human patterns (all-birthday, arithmetic sequences, past
  jackpots). Doesn't change the odds — nothing can — only reduces expected
  jackpot splitting.
- **Cluster-method picks** (`scripts/cluster_picker.py`, a port of the owner's
  JMP `f5.jsl`): adjacent-position pair KDEs (num1·num2 … num4·num5 — the
  physically plausible pairs; non-adjacent pairs excluded) on recent
  drawn-order draws, points selected at normalized density ≥ 0.77, tickets
  ranked by summed log pair-density; PB from the most-recently-active 1D
  density cluster (separate chamber). Window (270) and density (0.77) chosen by
  forward selection over a window × density grid (weak, fading signal); a
  rolling backtest is published next to the picks —
  currently consistent with chance, and the page says so. Powerball only:
  the method requires drawn order, which CA does not publish for SuperLotto
  (CPRA records request is the open path).

## Layout

```
scripts/   fetchers + analysis + update.py (CI entry point)
data/      accumulated CSVs (committed so history grows)
data.json  published snapshot consumed by index.html + the dashboard card
```

Local run: `pip install numpy certifi && python scripts/update.py`

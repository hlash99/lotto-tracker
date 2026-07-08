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
- Screened quick picks: uniform random tickets, re-drawn if they match
  over-played human patterns (all-birthday, arithmetic sequences, past
  jackpots). Doesn't change the odds — nothing can — only reduces expected
  jackpot splitting.
- **Cluster-method picks** (`scripts/cluster_picker.py`, a port of the owner's
  JMP `f5.jsl`): all 10 position-pair KDEs on recent drawn-order draws,
  points selected at normalized density ≥ 0.94, tickets ranked by summed log
  pair-density; PB from the most-recently-active 1D density cluster. Window
  (200) tuned by a train/holdout walk-forward sweep; a rolling backtest is
  published next to the picks — currently consistent with chance, and the
  page says so. Powerball only: the method requires drawn order, which CA
  does not publish for SuperLotto (CPRA records request is the open path).

## Layout

```
scripts/   fetchers + analysis + update.py (CI entry point)
data/      accumulated CSVs (committed so history grows)
data.json  published snapshot consumed by index.html + the dashboard card
```

Local run: `pip install numpy certifi && python scripts/update.py`

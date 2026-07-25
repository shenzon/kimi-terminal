# Kimi L1 — 4H Swing Terminal

Auto-updating dashboard for the **Kimi L1** low-lag trend filter on
**Gold (XAUUSD), Silver (XAGUSD)** and **EUR/USD**, 4-hour swing timeframe.

Each panel shows a plain-English verdict — **TRADE / EXIT / NO TRADE** — plus
a bias gauge, vol-target position size, and a price × L1-trend sparkline.
Times in **Malaysia / Singapore (MYT/SGT, UTC+8)**.

- **Long-only edge** validated on gold & silver (short side loses); EUR/USD is context-only.
- **Vol-target sizing**: `size = median_vol / current_realized_vol`, capped — lifts Sharpe, tightens drawdown.

A GitHub Action rebuilds `index.html` from live data every 4H bar
(00/04/08/12/16/20 UTC) and deploys it to GitHub Pages.

Not financial advice.

## TradingView indicator

`pine/silver_kimi_l1_4h.pine` — a Pine v5 port of the dual-lens model for
**XAGUSD 4H**: SLOW lens (N=12) drives the bias / trend label / exit, FAST lens
(N=4) drives entry timing, with a volume-confirmation gate and the asymmetric
long-only ENTER/HOLD/EXIT rule. Paste into the TradingView Pine Editor on a 4H
XAGUSD chart. Core math matches `swing4h.py` (validated to 1e-9), including the
**frozen λ anchor** (default = `LAM_ANCHOR["xagusd"]`) so the read is stable
day-over-day; toggle *Freeze lambda anchor* off for the old rolling median.
Signals still won't reproduce the Python tick-for-tick because TradingView's
XAGUSD (spot) differs from Yahoo `SI=F` (futures).

## Local build
```
pip install -r requirements.txt
python build_dashboard.py            # writes index.html
```

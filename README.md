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

## Local build
```
pip install -r requirements.txt
python build_dashboard.py            # writes index.html
```

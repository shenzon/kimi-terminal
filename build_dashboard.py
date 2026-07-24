#!/usr/bin/env python3
"""
Build the Kimi L1 dashboard (index.html) from live 4H data.

Pulls gold / silver / EUR-USD / USD-JPY via swing4h.py, computes the current state
(bias, direction, vol-target size, TRADE/EXIT/NO-TRADE verdict, price×trend
sparkline), and injects it as JSON into dashboard_template.html. Times in
Malaysia/Singapore (MYT/SGT, UTC+8).

Usage:  python3 build_dashboard.py [out.html]   # default: index.html
"""
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import swing4h as S

HERE = Path(__file__).resolve().parent
MYT = timedelta(hours=8)


def series_state(key):
    m = S.INSTRUMENTS[key]
    df = fetch_with_retry(m["ticker"])
    close = df["Close"]
    mintick = m.get("mintick", 0.0001)

    # DUAL LENS (matches live swing4h): SLOW = trend regime (bias/label/trend/
    # exit), FAST = early turns (entry timing). Run slow manually to keep the
    # trend sparkline; fast only needs its final slope.
    slow = S.KimiL1(lam=S.auto_lambda(close, S.SLOW_SWING_N), mintick=mintick)
    trend_hist = []
    for px in close.to_numpy(dtype=float):
        slow.update(px)
        trend_hist.append(slow.trend)
    fast = S.run_lens(close, S.FAST_SWING_N, mintick)

    last = float(close.iloc[-1])
    a = S.atr(df)
    dist = slow.trend_dist_pct(last)
    tradeable = key in S.TRADEABLE
    vt_size, vt_ctx = S.vol_target_size(close) if tradeable else (1.0, "")
    stop = last - 1.5 * a          # long-only: stop below / target above
    tgt = last + 3.0 * a

    # Asymmetric entry gate (stateless snapshot — no frozen-position store in
    # CI, so the page shows the ENTRY decision "what to do now").
    trend_ok = slow.direction != "BEAR"
    vol_ok, vol_ctx = (S.volume_confirms(df["Volume"]) if tradeable
                       else (True, ""))
    want_long = fast.slope > 0 and trend_ok and vol_ok

    if tradeable:
        if want_long:
            action = "TRADE"
            sub = "go long — fast turned up inside a non-bearish trend"
            lvl = {"entry": last, "stop": stop, "target": tgt}
        else:
            action, lvl = "NO TRADE", None
            if not trend_ok:
                sub = "slow trend is down — stay flat"
            elif fast.slope <= 0:
                sub = "waiting for fast lens to turn up"
            else:
                sub = f"fast up but volume light ({vol_ctx}) — wait"
    else:
        # Context-only: L1 still reads a trend, but no backtested edge.
        if slow.direction == "BULL":
            sub = "trend up, but no backtested edge"
        elif slow.direction == "BEAR":
            sub = "trend down, but no backtested edge"
        else:
            sub = "context only — no edge"
        action, lvl = "NO TRADE", None

    N = 90
    idx = df.index[-N:]
    px_h = close.to_numpy()[-N:]
    tr_h = np.array(trend_hist)[-N:]
    spark = [{"t": (ts + MYT).strftime("%d %b %H:%M"),
              "p": round(float(p), 5), "tr": round(float(tr), 5)}
             for ts, p, tr in zip(idx, px_h, tr_h)]
    return {
        "key": key, "label": m["label"], "sym": m["label"].split()[0],
        "tradeable": tradeable, "bias": slow.bias_score, "direction": slow.direction,
        "slope": round(slow.slope, 6),
        "slope_sign": "UP" if slow.slope > 0 else "DOWN" if slow.slope < 0 else "FLAT",
        "trend": round(slow.trend, 5), "price": round(last, 5), "dist": round(dist, 2),
        "vt_size": round(vt_size, 2), "vt_ctx": vt_ctx,
        "decimals": int(m["px"].strip(".f")),
        "action": action, "sub": sub, "levels": lvl,
        # index labels are bin LEFT edges (bar open) — show the CLOSE, or the
        # page reads a full 4H stale now that the forming bar is dropped
        "bar_myt": (df.index[-1] + S.BAR + MYT).strftime("%a %d %b %Y · %H:%M"),
        "bars": len(df), "lam": round(slow.lam, 4), "spark": spark,
    }


def fetch_with_retry(ticker, tries=4):
    """Yahoo occasionally rate-limits CI IPs — retry with backoff."""
    last_err = None
    for i in range(tries):
        try:
            df = S.fetch_4h(ticker)
            if len(df) > 200:
                return df
            last_err = RuntimeError(f"thin data ({len(df)} bars)")
        except Exception as e:      # noqa: BLE001
            last_err = e
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"{ticker}: {last_err}")


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "index.html"
    state = {
        "generated_myt": (pd.Timestamp.now(tz="UTC") + MYT).strftime("%a %d %b %Y · %H:%M"),
        "instruments": [series_state(k) for k in ("gold", "xagusd", "eurusd", "usdjpy")],
    }
    template = (HERE / "dashboard_template.html").read_text()
    html = template.replace("__KIMI_DATA__", json.dumps(state))
    out.write_text(html)
    for i in state["instruments"]:
        print(f"{i['sym']:<7} {i['action']:<9} bias {i['bias']:>3} "
              f"{i['direction']:<5} price {i['price']} vt {i['vt_size']}")
    print(f"wrote {out}  ({len(html)} bytes)  bar {state['instruments'][0]['bar_myt']} MYT")


if __name__ == "__main__":
    main()

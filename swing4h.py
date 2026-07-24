#!/usr/bin/env python3
"""
4H swing analysis for Gold (XAUUSD) and EUR/USD, driven by a faithful Python
port of the *Kimi L1 Trend Filter v1.3.15* (Pine) — a streaming CUSUM-style
change-point slope tracker with g-h level correction.

WHY THIS AND NOT EMA/RSI:
  Moving averages/RSI lag by design (they average a window). Kimi L1's slope
  only moves at sparse breakpoints — no window, near-zero lag — so a genuine
  4H trend shift shows up on the bar it happens, not N bars later.

This is NOT Boyd's batch convex L1 solve. It replicates the streaming state
machine exactly: predict -> residual -> g-h level correct -> accumulate dual
-> soft-threshold(lambda) -> gap-clip -> slope update -> anti-windup reset,
plus the safe_amp peak tracker, 0-100 bias score, materiality gate, warmup,
and FIRED persistence.

Feed: Yahoo 1H bars resampled to 4H (~15 min delayed — fine for 4H swing).
Lambda: AUTO — N x median(|dsrc|) swing anchor, so it self-scales gold vs FX.

Usage:
    python3 swing4h.py            # both
    python3 swing4h.py gold       # gold only
    python3 swing4h.py eurusd     # EUR/USD only
"""
import sys
import os
import json
import subprocess
import warnings
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import yfinance as yf

warnings.simplefilter("ignore")

# frozen open-position store (survives restarts) — locks TP/stop at entry
POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".swing4h_pos.json")


def load_positions():
    try:
        with open(POS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_positions(pos):
    with open(POS_FILE, "w") as f:
        json.dump(pos, f, indent=2)


def notify(title, body):
    """Desktop notification (Linux notify-send); silent no-op if unavailable."""
    try:
        subprocess.Popen(["notify-send", "-a", "Kimi L1", "-u", "normal", title, body],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, OSError):
        pass

INSTRUMENTS = {
    "gold":   {"ticker": "GC=F",     "label": "GOLD  (XAUUSD / GC futures)",   "px": ".2f", "slp": "0.3f", "mintick": 0.01},
    "xagusd": {"ticker": "SI=F",     "label": "SILVER (XAGUSD / SI futures)",  "px": ".3f", "slp": "0.4f", "mintick": 0.005},
    "eurusd": {"ticker": "EURUSD=X", "label": "EUR/USD",                       "px": ".5f", "slp": "0.6f", "mintick": 0.00001},
    "usdjpy": {"ticker": "USDJPY=X", "label": "USD/JPY",                       "px": ".3f", "slp": "0.4f", "mintick": 0.001},
}

# Instruments with a backtest-validated LONG-only edge → get the TRADE/EXIT
# system. Others are context-only (NO TRADE). Gold validated; silver pending
# the backtest below — promote it here only if it shows a real edge.
TRADEABLE = {"gold", "xagusd"}

# ── Kimi L1 params (match the Pine defaults) ───────────────────────────────
ALPHA          = 0.15   # slope adjustment magnitude
LEVEL_GAIN     = 0.10   # g-h level re-anchor (g)
GAP_CLIP       = 3.0    # max |delta| per bar, in units of lambda
MIN_BP_FRAC    = 0.10   # materiality gate: |slope_change| >= frac * safe_amp
AMP_LEN        = 50     # slope-amplitude decay length
AMP_FLOOR_MULT = 1.5    # bar-change-MAD seed multiplier
WARMUP_BPS     = 2      # warmup path A: breakpoints
WARMUP_BARS    = 30     # warmup path B: bars since first BP
BULL_SCORE     = 65     # score >= -> BULL
BEAR_SCORE     = 35     # score <= -> BEAR
DUAL_PCT_ALERT = 50.0   # dual >= this % of lambda -> "building"
FIRED_SHOW     = 2      # keep FIRED visible for N bars
AUTOLAM_LEN    = 100    # median lookback for auto-lambda
# N x median bar move. The Pine ballparks are DAILY; on 4H bars a daily-tuned
# N (12) is too sluggish and lags reversals by ~6 bars. N=4 makes the slope
# track 4H momentum so the read matches the chart instead of holding a stale
# trend through a fresh selloff. (See systematic-debugging session 2026-07-23.)
AUTOLAM_SWING_N = 4.0   # swing-style multiplier, 4H-corrected
PEAK_EXPAND    = 1.05
# Vol-target sizing (kimi_qte.py, 2026-07-23): robust Sharpe lift ~0.97->1.25,
# MaxDD ~-21%->-13% on gold Rule D. size = median_vol / realized_vol(W), capped.
VOL_W          = 30     # realized-vol window (~5 trading days on 4H)
VOL_CAP        = 3.0    # max size multiplier


def soft(x, thresh):
    return np.sign(x) * max(0.0, abs(x) - thresh)


@dataclass
class KimiL1:
    lam: float
    # persistent state
    trend: float = None
    slope: float = 0.0
    dual: float = 0.0
    bp_count: int = 0
    first_bp_bar: int = None
    bar_change_mad: float = None
    slope_amplitude: float = None
    last_fire_dir: int = 0
    last_fire_bar: int = None
    bar: int = -1
    mintick: float = 0.0001
    # last-bar diagnostics (filled by update)
    slope_change: float = 0.0
    breakpoint: bool = False
    bp_material: bool = False
    safe_amp: float = 0.0
    bias_score: int = 50
    warmed_up: bool = False
    prev_src: float = None
    history: list = field(default_factory=list)

    def update(self, src):
        """One confirmed bar. Mirrors the Pine is_new_bar block + derived layer."""
        self.bar += 1
        i = self.bar

        # ── init on first valid src ──
        if self.trend is None:
            self.trend = src
            self.slope = 0.0
            self.dual = 0.0

        prev_slope = self.slope
        old_slope = self.slope

        # 1-3. predict, residual, g-h LEVEL CORRECTION
        predicted = self.trend + old_slope
        resid = src - predicted
        self.trend = predicted + LEVEL_GAIN * resid

        # 4. accumulate POST-correction residual into dual
        resid_dual = src - self.trend
        self.dual += resid_dual

        # 5. soft-threshold vs lambda
        delta = soft(self.dual, self.lam)

        # 6. gap clip
        cap = self.lam * GAP_CLIP
        if abs(delta) > cap:
            delta = np.sign(delta) * cap

        # 7. slope update
        self.slope = old_slope + delta * ALPHA

        # 8. anti-windup full reset on breakpoint
        if delta != 0.0:
            self.dual = 0.0

        self.slope_change = self.slope - prev_slope
        self.breakpoint = delta != 0.0
        if self.breakpoint:
            self.bp_count += 1
            if self.first_bp_bar is None:
                self.first_bp_bar = i

        # ── amp floor: EMA of |bar change| (MAD-ish) ──
        if self.prev_src is not None:
            change = src - self.prev_src
            if self.bar_change_mad is None:
                self.bar_change_mad = abs(change)
            else:
                a = 2.0 / (AMP_LEN + 1.0)
                self.bar_change_mad = self.bar_change_mad * (1 - a) + abs(change) * a
        amp_floor = (self.bar_change_mad if self.bar_change_mad is not None
                     else self.mintick * 100.0) * AMP_FLOOR_MULT

        # ── slope amplitude: peak tracker w/ EMA decay-down ──
        cur_amp = abs(self.slope)
        if self.slope_amplitude is None:
            if cur_amp > 0.0:
                self.slope_amplitude = max(cur_amp * PEAK_EXPAND, amp_floor)
        elif cur_amp > self.slope_amplitude:
            self.slope_amplitude = max(cur_amp * PEAK_EXPAND, amp_floor)
        else:
            d = 2.0 / (AMP_LEN + 1.0)
            self.slope_amplitude = self.slope_amplitude * (1 - d) + cur_amp * d

        self.safe_amp = max(self.slope_amplitude if self.slope_amplitude is not None
                            else amp_floor, self.mintick)

        # ── materiality gate ──
        min_bp_delta = MIN_BP_FRAC * self.safe_amp
        self.bp_material = self.breakpoint and abs(self.slope_change) >= min_bp_delta
        if self.bp_material:
            self.last_fire_dir = 1 if self.slope_change > 0 else -1
            self.last_fire_bar = i

        # ── warmup + bias score ──
        bars_since_first = 0 if self.first_bp_bar is None else i - self.first_bp_bar
        self.warmed_up = (self.slope_amplitude is not None and
                          (self.bp_count >= WARMUP_BPS or bars_since_first >= WARMUP_BARS))
        slope_norm = max(0.0, min(100.0, 50.0 + (self.slope / self.safe_amp) * 50.0))
        self.bias_score = round(slope_norm) if self.warmed_up else 50

        self.prev_src = src
        self.history.append(self.slope)

    # ── read-outs ──
    @property
    def dual_pct(self):
        return abs(self.dual) / self.lam * 100.0 if self.lam else 0.0

    @property
    def direction(self):
        if not self.warmed_up:
            return "WARMING"
        if self.bias_score >= BULL_SCORE:
            return "BULL"
        if self.bias_score <= BEAR_SCORE:
            return "BEAR"
        return "FLAT"

    def trend_dist_pct(self, src):
        return 0.0 if not self.trend or src == 0 else (src - self.trend) / src * 100.0


BAR = pd.Timedelta("4h")


def fetch_4h(ticker, closed_only=True, with_forming=False):
    """Resampled 4H bars. Index labels are bin LEFT edges (bar open); the bar
    labelled t covers [t, t+4h). With closed_only the still-forming bar is
    dropped, so the read only ever moves when a bar actually closes.

    Returns the closed bars, or (closed_bars, forming_bar_or_None) if
    with_forming.
    """
    df = yf.download(ticker, period="720d", interval="1h",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Pin the index to UTC before resampling. Yahoo's tz varies by ticker and
    # environment, and resample("4h") bins on the index's own wall clock — a
    # non-UTC index silently shifts every bin edge (different bar count, wrong
    # bar labels) and breaks any downstream "+ offset" display arithmetic.
    df.index = (df.index.tz_localize("UTC") if df.index.tz is None
                else df.index.tz_convert("UTC"))
    o = df["Open"].resample("4h").first()
    h = df["High"].resample("4h").max()
    l = df["Low"].resample("4h").min()
    c = df["Close"].resample("4h").last()
    out = pd.DataFrame({"Open": o, "High": h, "Low": l,
                        "Close": c}).dropna(subset=["Close"])
    forming = None
    if closed_only and len(out):
        now = pd.Timestamp.now(tz=out.index.tz)
        if out.index[-1] + BAR > now:
            forming = out.iloc[-1]
            out = out.iloc[:-1]
    return (out, forming) if with_forming else out


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean().iloc[-1]


def auto_lambda(close: pd.Series) -> float:
    """N x median(|dsrc|) swing anchor over the last AUTOLAM_LEN bars."""
    move = close.diff().abs().tail(AUTOLAM_LEN)
    med = float(np.nanmedian(move))
    return med * AUTOLAM_SWING_N


def vol_target_size(close: pd.Series, w=VOL_W, cap=VOL_CAP):
    """Vol-target position multiplier: median_vol / current_vol, clipped [0, cap].
    >1 in calm regimes (size up), <1 in turbulence (size down). Returns
    (size, context_str). Median is the whole-window scale reference."""
    ret = close.pct_change().to_numpy()
    vol = pd.Series(ret).rolling(w).std().to_numpy()
    cur = vol[-1]
    finite = vol[np.isfinite(vol) & (vol > 0)]
    if not (np.isfinite(cur) and cur > 0 and finite.size):
        return 1.0, "insufficient history"
    ref = float(np.median(finite))
    size = float(np.clip(ref / cur, 0.0, cap))
    pct = (cur / ref - 1.0) * 100.0
    tag = "calm" if size > 1.05 else "turbulent" if size < 0.95 else "normal vol"
    where = "below" if pct < 0 else "above"
    return size, f"realized vol {abs(pct):.0f}% {where} median — {tag}"


def signal_line(k: KimiL1):
    """FIRED / BUILDING SIGNAL row (v1.3.14 persistence)."""
    bars_since = 10**9 if k.last_fire_bar is None else k.bar - k.last_fire_bar
    if k.bp_material and k.slope_change > 0:
        return "\U0001F525 FIRED ▲"
    if k.bp_material and k.slope_change < 0:
        return "\U0001F525 FIRED ▼"
    if k.last_fire_bar is not None and 0 < bars_since <= FIRED_SHOW:
        arrow = "▲" if k.last_fire_dir > 0 else "▼"
        return f"\U0001F525 FIRED {arrow} ({bars_since} bar{'s' if bars_since > 1 else ''} ago)"
    if k.dual_pct >= DUAL_PCT_ALERT and not k.breakpoint:
        d = "▲" if k.dual > 0 else "▼"
        return f"⏳ BUILDING {d} {k.dual_pct:.0f}% (unfired)"
    return "— no signal"


# Instruments with a backtest-validated SHORT edge. Empty by design:
# gold short-only = -29% / Sharpe -0.87, EURUSD no edge either side
# (backtest_kimi.py, 2026-07-23). Down-fires are EXIT/NO-LONG, never "sell".
SHORTABLE = frozenset()


def guidance(k: KimiL1, key=None):
    """Priority-ordered trade guidance (v1.3.13): building = watch, fired = act.

    Long-only guard: for instruments not in SHORTABLE, a slope-down fire is an
    exit / stand-aside signal, NOT a sell — the short side has no validated edge.
    """
    shortable = key in SHORTABLE
    if not k.warmed_up:
        return "Wait — collecting first breakpoints"
    if k.bp_material and k.slope_change > 0:
        return "▲ FIRED — slope-up breakpoint confirmed, actionable"
    if k.bp_material and k.slope_change < 0:
        if shortable:
            return "▼ FIRED — slope-down breakpoint confirmed, actionable (short)"
        return "▼ FIRED — slope-down breakpoint — EXIT / NO-LONG (long-only: NOT a sell)"
    if k.dual_pct >= DUAL_PCT_ALERT and not k.breakpoint:
        return f"WATCH — pressure {k.dual_pct:.0f}% (UNFIRED), do NOT trade until it fires"
    if k.direction == "BULL":
        return "Ride confirmed up-trend — pullbacks = add"
    if k.direction == "BEAR":
        if shortable:
            return "Fade confirmed down-trend — bounces = short"
        return "Down-trend — stand aside / stay FLAT (long-only, no short)"
    return "Wait — no directional slope"


def analyze(key):
    m = INSTRUMENTS[key]
    df, forming = fetch_4h(m["ticker"], with_forming=True)
    close = df["Close"]
    lam = auto_lambda(close)

    k = KimiL1(lam=lam, mintick=m.get("mintick", 0.0001))
    for px in close.to_numpy(dtype=float):
        k.update(px)

    last = float(close.iloc[-1])
    a = atr(df)
    pxf, slp = m["px"], m["slp"]
    dist = k.trend_dist_pct(last)
    vt_size, vt_ctx = vol_target_size(close) if key in TRADEABLE else (1.0, "")
    dir_sym = {"BULL": "▲ BULL", "BEAR": "▼ BEAR",
               "FLAT": "— FLAT", "WARMING": "⏳ WARMING"}[k.direction]

    # ATR stop/target aligned to the slope-sign bias (1.5 / 3.0 ATR ~ 1:2R)
    long_side = k.slope > 0
    stop = last - 1.5 * a if long_side else last + 1.5 * a
    tgt = last + 3.0 * a if long_side else last - 3.0 * a

    bar = "".join("█" if i < round(k.bias_score / 10) else "░" for i in range(10))

    print(f"\n{'='*60}\n \U0001F4D0 KIMI L1  —  {m['label']}")
    b_open = df.index[-1]
    b_close = b_open + BAR
    print(f" bar {b_open:%Y-%m-%d %H:%M}→{b_close:%H:%M} UTC closed   "
          f"({len(df)} 4H bars, λ={lam:{slp}} auto)")
    if forming is not None:
        print(f" forming  {forming.name:%H:%M}→{forming.name + BAR:%H:%M} UTC   "
              f"last {float(forming['Close']):{pxf}}  (not in the read)")
    print("-" * 60)
    print(f"  BIAS        {bar}  {k.bias_score}/100   ({dir_sym})")
    print(f"  Slope       {k.slope:+{slp}} /bar   (sign: {'UP' if k.slope>0 else 'DOWN' if k.slope<0 else 'FLAT'})")
    print(f"  Trend level {k.trend:{pxf}}   price {last:{pxf}}  ({dist:+.2f}% vs trend)")
    print(f"  Norm scale  ±{k.safe_amp:{slp}} /bar   ({k.bp_count} breakpoints)")
    dd = "▲" if k.dual > 0 else "▼" if k.dual < 0 else "—"
    print(f"  Dual press. {dd} {k.dual_pct:.1f}% of λ")
    if key in TRADEABLE:
        print(f"  Vol-target  {vt_size:.2f}×  ({vt_ctx})")
    print("-" * 60)

    # ── ONE plain-English bottom line: TRADE / NO TRADE / EXIT ──
    event = None  # "entry" / "exit" for watch-loop notification
    sym = m["label"].split()[0]   # "GOLD" / "SILVER"
    if key in TRADEABLE:
        pos = load_positions()
        held = pos.get(key)
        if k.slope > 0:
            if held is None:
                # NEW entry — FREEZE entry/stop/tp/size here and remember them
                held = {"entry": last, "stop": stop, "tp": tgt, "size": round(vt_size, 2),
                        "entry_bar": f"{df.index[-1]:%Y-%m-%d %H:%M} UTC"}
                pos[key] = held
                save_positions(pos)
                event = "entry"
                notify(f"🟢 {sym} — TRADE: go long",
                       f"size {vt_size:.2f}×  entry {last:{pxf}}  SL {stop:{pxf}}  TP {tgt:{pxf}}")
                print(f"  ➤ ACTION     🟢 TRADE — GO LONG now  (size {vt_size:.2f}×)")
            else:
                sz = held.get("size", 1.0)   # back-compat: positions saved before sizing
                print(f"  ➤ ACTION     🟢 TRADE — HOLD your long  (size {sz:.2f}×)")
            e, s, t = held["entry"], held["stop"], held["tp"]
            r_now = (last - e) / (e - s) if e != s else 0.0
            prog = (last - e) / (t - e) * 100 if t != e else 0.0
            print(f"     entry {e:{pxf}}  ·  stop {s:{pxf}}  ·  target {t:{pxf}}")
            print(f"     now {last:{pxf}}  →  {r_now:+.2f}R  ({prog:+.0f}% to target)")
        else:
            if held is not None:
                e, s, t = held["entry"], held["stop"], held["tp"]
                res = ("target hit ✅" if last >= t else "stop hit ⛔" if last <= s
                       else f"{((last-e)/(e-s) if e!=s else 0):+.2f}R")
                pos.pop(key, None)
                save_positions(pos)
                event = "exit"
                notify(f"⚪ {sym} — EXIT: close long", f"{res}   entry {e:{pxf}} → now {last:{pxf}}")
                print(f"  ➤ ACTION     🔴 EXIT — CLOSE your long now  ({res})")
                print(f"     trend rolled over — this is an exit, NOT a sell/short")
            else:
                print(f"  ➤ ACTION     ⚪ NO TRADE — stay flat")
                print(f"     {sym} trend is not up — do nothing (long-only: never short here)")
    else:
        print(f"  ➤ ACTION     ⚪ NO TRADE — context only")
        print(f"     no tradeable edge on {m['label']} — use as background only")
    print("=" * 60)

    return {"bar": df.index[-1], "fire_bar": k.last_fire_bar,
            "fire_dir": k.last_fire_dir, "material": k.bp_material, "event": event}


def watch(keys, interval_min):
    """Auto-refresh loop: re-pull, re-run, ring the bell only on a NEW fire."""
    import time
    from datetime import datetime, timezone
    print(f"⏱  WATCH mode — every {interval_min} min, desktop alerts on entry/exit. Ctrl-C to stop.")
    notify("Kimi L1 watch started", f"gold/eurusd 4H, refresh {interval_min} min")
    try:
        while True:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            print(f"\n\n{'#'*60}\n#  refresh {stamp}\n{'#'*60}")
            for key in keys:
                try:
                    st = analyze(key)   # fires notify() itself on entry/exit
                except Exception as e:
                    print(f"[{key}] error: {e}")
                    continue
                if st.get("event"):
                    print(f"  🔔 desktop alert sent: gold {st['event'].upper()}")
            time.sleep(interval_min * 60)
    except KeyboardInterrupt:
        print("\n⏹  watch stopped.")


def main():
    argv = [a.lower() for a in sys.argv[1:]]
    watch_mode = "--watch" in argv
    interval = 30
    if watch_mode:
        argv.remove("--watch")
        # optional numeric arg = refresh minutes
        for a in list(argv):
            if a.isdigit():
                interval = int(a)
                argv.remove(a)
    keys = argv or list(INSTRUMENTS)
    keys = [k for k in keys if k in INSTRUMENTS] or list(INSTRUMENTS)

    if watch_mode:
        watch(keys, interval)
        return
    for a in keys:
        try:
            analyze(a)
        except Exception as e:
            print(f"[{a}] error: {e}")


if __name__ == "__main__":
    main()

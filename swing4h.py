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
# DUAL-LENS λ (2026-07-24). One slope can't both turn fast AND stay stable
# through pullbacks, so we run TWO on the same 4H bars:
#   FAST (small λ, N=4)  — early turns; drives FIRED/BUILDING + entry timing.
#   SLOW (large λ, N=12) — trend regime; drives the BULL/FLAT/BEAR label + exit.
# N is the swing multiplier on median(|Δ4H|). Small N = low threshold = fires on
# small moves (fast). Large N = daily-tuned, matches the on-chart "Kimi Smart
# Money Engine v1.4.3 [Daily Forex]" the user actually trades. See design doc
# docs/superpowers/specs/2026-07-24-swing4h-dual-lens-design.md.
FAST_SWING_N = 4.0
SLOW_SWING_N = 12.0
# FROZEN λ ANCHOR (2026-07-25). λ = ANCHOR x N. The anchor was the 100-bar
# median(|Δ4H|), RE-COMPUTED every run — but on a rolling 720d fetch that median
# swings ~70% in a month, silently rewriting breakpoint history: the live read
# (dir + in-trade state) differed day-over-day on 12/30 gold days purely from λ
# drift, and the backtest Sharpe moved 1.63->1.06 on one day of new data. The
# fixed-λ surface is jagged (gold 0.86-1.35, silver 0.49-1.46 across the grid),
# so no λ is knowably "optimal" — freezing just makes the model MEAN THE SAME
# THING every day. Re-anchor manually only if a symbol's median bar-move shifts
# ~±50% sustained. Investigation: scratchpad/lambda_stability.py, 2026-07-25.
LAM_ANCHOR = {
    "gold":   10.9,        # slow 130.8 / fast 43.6
    "xagusd": 0.357502,    # slow 4.29  / fast 1.43
    "eurusd": 0.000654638, # slow 0.00786 / fast 0.00262
    "usdjpy": 0.098999,    # slow 1.188 / fast 0.396
}
PEAK_EXPAND    = 1.05
# Vol-target sizing (kimi_qte.py, 2026-07-23): robust Sharpe lift ~0.97->1.25,
# MaxDD ~-21%->-13% on gold Rule D. size = median_vol / realized_vol(W), capped.
VOL_W          = 30     # realized-vol window (~5 trading days on 4H)
VOL_CAP        = 3.0    # max size multiplier
# Volume-confirmation gate on the fast-lens ENTRY (gold/silver only; FX has no
# Yahoo volume). Entry bar must trade >= median volume of the recent window, so
# a thin low-participation up-bar can't trip an entry. Fails OPEN on missing
# data. See docs/superpowers/specs/2026-07-24-swing4h-volume-gate-design.md.
VOL_CONFIRM_LOOKBACK = 20   # bars for the median-volume baseline (~3.3 days)
VOL_CONFIRM_MULT     = 1.0  # entry vol must be >= MULT x median


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
    # Volume: sum the constituent 1H bars. Present for futures (gold/silver),
    # all-zero for Yahoo FX — the volume gate fails open on that.
    v = (df["Volume"].resample("4h").sum() if "Volume" in df.columns
         else pd.Series(0.0, index=c.index))
    out = pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c,
                        "Volume": v}).dropna(subset=["Close"])
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


def auto_lambda(close: pd.Series, n: float, anchor: float = None) -> float:
    """n x swing-anchor, where the anchor is median(|dsrc|). Pass a FROZEN
    `anchor` (from LAM_ANCHOR) for a stable day-over-day λ; anchor=None falls
    back to re-computing the 100-bar median (unstable — see LAM_ANCHOR note).
    n = FAST_SWING_N (twitchy) or SLOW_SWING_N (trend-stable)."""
    if anchor is None:
        anchor = float(np.nanmedian(close.diff().abs().tail(AUTOLAM_LEN)))
    return anchor * n


def run_lens(close: pd.Series, n: float, mintick: float, anchor: float = None) -> KimiL1:
    """Run a full KimiL1 pass over `close` at swing multiplier `n`."""
    k = KimiL1(lam=auto_lambda(close, n, anchor), mintick=mintick)
    for px in close.to_numpy(dtype=float):
        k.update(px)
    return k


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


def volume_confirms(vol: pd.Series):
    """Fast-lens ENTRY volume gate. True when the latest closed bar traded
    >= VOL_CONFIRM_MULT x median volume over the recent window. Fails OPEN
    (returns True) when volume is 0/NaN or no positive history exists — never
    block an entry on missing data. Returns (ok, context_str)."""
    if vol is None or len(vol) == 0:
        return True, "no volume data — gate skipped"
    last = float(vol.iloc[-1])
    hist = vol.tail(VOL_CONFIRM_LOOKBACK)
    pos = hist[hist > 0]
    if not np.isfinite(last) or last <= 0 or pos.empty:
        return True, "no volume data — gate skipped"
    med = float(np.median(pos.to_numpy()))
    ratio = last / med if med > 0 else 0.0
    ok = last >= VOL_CONFIRM_MULT * med
    tag = "✓" if ok else "— light"
    return ok, f"vol {ratio:.1f}× median {tag}"


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


# ── terminal styling (256-color; auto-off when piped or NO_COLOR set) ──────
_USE_COLOR = sys.stdout.isatty() and "NO_COLOR" not in os.environ

def _c(s, *codes):
    """Wrap s in ANSI SGR codes, or return it untouched when color is off."""
    if not _USE_COLOR or not codes:
        return s
    return "\033[" + ";".join(codes) + "m" + s + "\033[0m"

_DIM, _B = "2", "1"
_GOLD, _SILV, _BLUE, _CYAN = "38;5;178", "38;5;250", "38;5;111", "38;5;80"
_GRN, _RED, _AMB, _WHT = "38;5;114", "38;5;203", "38;5;214", "38;5;255"
_ACCENT = {"gold": _GOLD, "xagusd": _SILV, "eurusd": _BLUE, "usdjpy": _CYAN}
_DIR = {"BULL": ("▲ BULL", _GRN), "BEAR": ("▼ BEAR", _RED),
        "FLAT": ("— FLAT", _AMB), "WARMING": ("⏳ WARM", _DIM)}


def _meter(score, width=22):
    """Bias track: red/amber/green zones at the real BEAR/BULL thresholds,
    with a bright marker sitting at the current score."""
    pos = round(score / 100 * (width - 1))
    cells = []
    for i in range(width):
        v = i / (width - 1) * 100
        zone = _RED if v < BEAR_SCORE else _GRN if v >= BULL_SCORE else _AMB
        cells.append(_c("◆", _B, _WHT) if i == pos else _c("━", zone))
    return "".join(cells)


def analyze(key):
    m = INSTRUMENTS[key]
    df, forming = fetch_4h(m["ticker"], with_forming=True)
    close = df["Close"]
    mintick = m.get("mintick", 0.0001)

    # DUAL LENS on the same 4H bars: slow = trend regime (label + exit),
    # fast = early turns (fired/building signal + entry timing). Frozen λ anchor
    # (LAM_ANCHOR) for a stable day-over-day read; None-fallback stays adaptive.
    anchor = LAM_ANCHOR.get(key)
    slow = run_lens(close, SLOW_SWING_N, mintick, anchor)
    fast = run_lens(close, FAST_SWING_N, mintick, anchor)

    last = float(close.iloc[-1])
    a = atr(df)
    pxf, slp = m["px"], m["slp"]
    dist = slow.trend_dist_pct(last)
    vt_size, vt_ctx = vol_target_size(close) if key in TRADEABLE else (1.0, "")
    dtxt, dcol = _DIR[slow.direction]

    # Asymmetric long-only rule: ENTER when fast turns up within a non-bearish
    # slow regime; HOLD while slow trend holds; EXIT only when slow goes BEAR.
    # Gate on slow.direction (the displayed label) not raw slope sign, so a
    # slope grazing zero (still labelled FLAT) doesn't chatter us in/out.
    trend_ok = slow.direction != "BEAR"              # hold gate: BULL/FLAT/WARMING
    # Volume-confirmation gate on ENTRY only (gold/silver; FX fails open).
    vol_ok, vol_ctx = (volume_confirms(df["Volume"]) if key in TRADEABLE
                       else (True, ""))
    want_long = fast.slope > 0 and trend_ok and vol_ok   # early-entry trigger

    # ATR stop/target for the long side (1.5 / 3.0 ATR ~ 1:2R). Long-only here.
    stop = last - 1.5 * a
    tgt = last + 3.0 * a

    # ── card: left accent spine + dim labels / bright values ──
    acc = _ACCENT.get(key, _CYAN)
    sp = _c("▌", acc)
    sym = m["label"].split()[0]                 # "GOLD" / "EUR/USD"
    desc = m["label"].split("(", 1)[-1].rstrip(")") if "(" in m["label"] else ""

    def row(lbl, val):
        print(f"{sp}  {_c(lbl.ljust(9), _DIM)} {val}")

    b_open, b_close = df.index[-1], df.index[-1] + BAR
    scol = _GRN if slow.slope > 0 else _RED if slow.slope < 0 else _AMB
    fcol = _GRN if fast.slope > 0 else _RED if fast.slope < 0 else _AMB
    distc = _GRN if dist >= 0 else _RED
    dd = "▲" if fast.dual > 0 else "▼" if fast.dual < 0 else "—"

    print()
    print(f"{sp} {_c('📐 KIMI L1', _DIM)}  {_c(sym, _B, acc)}"
          f"{('  ' + _c(desc, _DIM)) if desc else ''}   {_c('[ ' + dtxt + ' ]', _B, dcol)}")
    print(f"{sp} {_c(f'{b_open:%Y-%m-%d %H:%M}→{b_close:%H:%M} UTC · {len(df)} bars · λ slow {slow.lam:{slp}} / fast {fast.lam:{slp}}', _DIM)}")
    if forming is not None:
        f_last = float(forming["Close"])
        print(f"{sp} {_c(f'forming {forming.name:%H:%M}→{forming.name + BAR:%H:%M} · last {f_last:{pxf}} (excl)', _DIM)}")
    print(sp)
    row("bias", f"{_meter(slow.bias_score)}  {_c(f'{slow.bias_score}/100', _B)}  {_c(dtxt, dcol)}")
    row("slope", f"{_c(f'{slow.slope:+{slp}}', _B, scol)} {_c('/bar trend', _DIM)}   "
                 f"{_c('fast', _DIM)} {_c(f'{fast.slope:+{slp}}', fcol)}")
    row("trend", f"{slow.trend:{pxf}}   {_c('price', _DIM)} {_c(f'{last:{pxf}}', _B)}  "
                 f"{_c(f'{dist:+.2f}%', distc)}")
    row("norm ±", f"{slow.safe_amp:{slp}} {_c(f'/bar · {slow.bp_count} breakpoints', _DIM)}")
    row("signal", _c(signal_line(fast), _DIM))
    row("dual", _c(f"{dd} {fast.dual_pct:.1f}% of λ (fast)", _DIM))
    if key in TRADEABLE:
        row("size", f"{_c(f'{vt_size:.2f}×', _B)}  {_c(vt_ctx, _DIM)}")
        vcol = _GRN if vol_ok else _AMB
        row("vol", _c(vol_ctx, vcol))
    print(sp)

    def action(dot, dotcol, head, headcol):
        print(f"{sp}  {_c('▶', _DIM)} {_c(dot, dotcol)} {_c(head, _B, headcol)}")

    def note(txt):
        print(f"{sp}    {_c(txt, _DIM)}")

    # ── ONE plain-English bottom line: TRADE / NO TRADE / EXIT ──
    event = None  # "entry" / "exit" for watch-loop notification
    if key in TRADEABLE:
        pos = load_positions()
        held = pos.get(key)
        # HOLD/EXIT is governed by the SLOW lens (trend_ok); ENTRY needs the
        # FAST lens to turn up inside that non-bearish regime (want_long).
        if held is not None and trend_ok:
            # still in the trend — HOLD (fast pullbacks don't kick us out).
            # REBALANCE size to the current vol-target each refresh (backtested
            # F+size rebal): entry/stop/tp stay frozen — only the size multiplier
            # tracks live volatility, sizing down as turbulence builds mid-trade.
            entry_sz = held.get("entry_size", held.get("size", 1.0))
            cur_sz = round(vt_size, 2)
            if held.get("size") != cur_sz:      # persist the rebalanced size
                held["size"] = cur_sz
                pos[key] = held
                save_positions(pos)
            action("●", _GRN,
                   f"TRADE — HOLD your long  (size {cur_sz:.2f}× · entry {entry_sz:.2f}×)", _GRN)
            note("holding through the pullback — slow trend still up (fast may dip)"
                 if fast.slope <= 0 else "slow trend up, fast confirms")
            if abs(cur_sz - entry_sz) >= 0.05:
                verb = "down" if cur_sz < entry_sz else "up"
                note(f"size rebalanced {verb} to {cur_sz:.2f}× ({vt_ctx})")
            e, s, t = held["entry"], held["stop"], held["tp"]
            r_now = (last - e) / (e - s) if e != s else 0.0
            prog = (last - e) / (t - e) * 100 if t != e else 0.0
            rc = _GRN if r_now >= 0 else _RED
            note(f"entry {e:{pxf}}  ·  stop {s:{pxf}}  ·  target {t:{pxf}}")
            print(f"{sp}    {_c('now', _DIM)} {_c(f'{last:{pxf}}', _B)} → "
                  f"{_c(f'{r_now:+.2f}R', _B, rc)} {_c(f'({prog:+.0f}% to target)', _DIM)}")
        elif held is not None and not trend_ok:
            # slow trend rolled over — EXIT (single clean exit)
            e, s, t = held["entry"], held["stop"], held["tp"]
            res = ("target hit ✅" if last >= t else "stop hit ⛔" if last <= s
                   else f"{((last-e)/(e-s) if e!=s else 0):+.2f}R")
            pos.pop(key, None)
            save_positions(pos)
            event = "exit"
            notify(f"⚪ {sym} — EXIT: close long", f"{res}   entry {e:{pxf}} → now {last:{pxf}}")
            action("●", _RED, f"EXIT — CLOSE your long now  ({res})", _RED)
            note("slow trend rolled over — this is an exit, NOT a sell/short")
        elif want_long:
            # flat + fast turns up inside a non-bearish slow regime — ENTER
            held = {"entry": last, "stop": stop, "tp": tgt, "size": round(vt_size, 2),
                    "entry_size": round(vt_size, 2),   # frozen ref; size rebalances
                    "entry_bar": f"{df.index[-1]:%Y-%m-%d %H:%M} UTC"}
            pos[key] = held
            save_positions(pos)
            event = "entry"
            notify(f"🟢 {sym} — TRADE: go long",
                   f"size {vt_size:.2f}×  entry {last:{pxf}}  SL {stop:{pxf}}  TP {tgt:{pxf}}")
            action("●", _GRN, f"TRADE — GO LONG now  (size {vt_size:.2f}×)", _GRN)
            note("fast turned up inside a non-bearish trend — early entry")
            note(f"entry {last:{pxf}}  ·  stop {stop:{pxf}}  ·  target {tgt:{pxf}}")
        else:
            action("○", _AMB, "NO TRADE — stay flat", _AMB)
            if not trend_ok:
                reason = "slow trend is down"
            elif fast.slope <= 0:
                reason = "fast hasn't turned up yet"
            else:   # fast up + trend ok but volume gate blocked it
                reason = f"fast turned up but volume light ({vol_ctx}) — waiting for participation"
            note(f"{sym}: {reason} — do nothing (long-only: never short here)")
    else:
        action("○", _DIM, "NO TRADE — context only", _AMB)
        if slow.direction == "BULL":
            note(f"{sym} trend is up, but no backtested edge — background only")
        elif slow.direction == "BEAR":
            note(f"{sym} trend is down, but no backtested edge — background only")
        else:
            note(f"no tradeable edge on {m['label']} — use as background only")

    return {"bar": df.index[-1], "fire_bar": fast.last_fire_bar,
            "fire_dir": fast.last_fire_dir, "material": fast.bp_material, "event": event}


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

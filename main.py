#!/usr/bin/env python3
"""polymarket-nvidia-ceo-forecast-01 entry (Python 3.13, standard library only).

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at each of the 240 minutes after a checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%) per minute.

Method: the median is the last price (a shrink toward the trailing mean was
tried on the practice backtest and hurt, so the shrink is 0). The spreads
come from the market's own empirical distribution of h-minute price moves
over the trailing 18 hours of history, measured at a set of anchor horizons
and interpolated in sqrt(h) between them, then shrunk toward the pooled
cross-market distribution (thin markets borrow strength) and scaled by 0.5:
raw empirical spreads over-cover badly (rare past jumps widen every level).
A small floor keeps flat markets from claiming zero spread. Everything is
deterministic: no randomness, no clock, markets processed in input order,
files in name order.

Reads every *.in in $MAKEFUN_INPUT_DIR (else /input), writes <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output). Runs offline, stdlib only.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
# Anchor horizons (minutes) where empirical move distributions are measured.
ANCHORS = [1, 2, 4, 8, 16, 32, 64, 128, 240]
# How far the median is pulled from the last price toward the trailing mean.
SHRINK = 0.0
# Minutes of trailing history the median shrinks toward.
SHRINK_WINDOW = 120
# Minutes of trailing history used to measure move distributions (None = all).
# Recent volatility tracks the live regime better than a 72h window that
# includes long-past jumps. Tuned on the practice backtest (see README).
MOVE_WINDOW = 1080
# Global multiplier on the (recentred) spread offsets. Empirical move
# distributions overstate live spread (rare past jumps widen every level),
# so the offsets are shrunk toward the median. Tuned on the practice
# backtest: 0.5 with the 1080-minute window scored best.
SPREAD_SCALE = 0.5
# Shrinkage of a market's own move distribution toward the pooled one:
# own weight = n / (n + POOL_PRIOR), n = number of move samples.
POOL_PRIOR = 1500.0
# Minimum half-widths (price units per sqrt minute) so flat markets still
# carry some spread: (outer 5%/95% levels, inner 25%/75% levels).
FLOOR_OUTER = 0.0012
FLOOR_INNER = 0.0005
# Minimum own samples at an anchor before the market's own distribution is
# trusted at all; below this, the pooled distribution is used alone.
MIN_OWN_SAMPLES = 30


def quantile(sorted_samples, tau):
    """Linear-interpolation quantile of an ascending list (numpy 'linear')."""
    n = len(sorted_samples)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_samples[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac


def move_quantiles(prices, anchor):
    """Sorted h-minute moves of one market, and their quantiles at QUANTILES.

    Returns (quantile offsets list, sample count). Offsets are price moves,
    to be added to the median.
    """
    n = len(prices)
    count = n - anchor
    if count <= 0:
        return None, 0
    moves = sorted(prices[i + anchor] - prices[i] for i in range(count))
    return [quantile(moves, tau) for tau in QUANTILES], count


def pooled_quantiles(all_prices, anchor):
    """Move quantiles at one anchor, pooled over every market in the case."""
    moves = []
    for prices in all_prices:
        n = len(prices)
        if n > anchor:
            moves.extend(prices[i + anchor] - prices[i] for i in range(n - anchor))
    if not moves:
        return [0.0] * len(QUANTILES)
    moves.sort()
    return [quantile(moves, tau) for tau in QUANTILES]


def interp_offsets(anchor_offsets, h):
    """Interpolate per-level offsets at horizon h, linear in sqrt(h).

    anchor_offsets: {anchor: [offset per quantile level]}. h is within
    [ANCHORS[0], ANCHORS[-1]] by construction (1..240).
    """
    root = math.sqrt(h)
    if h <= ANCHORS[0]:
        return list(anchor_offsets[ANCHORS[0]])
    if h >= ANCHORS[-1]:
        return list(anchor_offsets[ANCHORS[-1]])
    for j in range(len(ANCHORS) - 1):
        a, b = ANCHORS[j], ANCHORS[j + 1]
        if a <= h <= b:
            ra, rb = math.sqrt(a), math.sqrt(b)
            w = (root - ra) / (rb - ra) if rb > ra else 0.0
            low, high = anchor_offsets[a], anchor_offsets[b]
            return [low[k] * (1.0 - w) + high[k] * w for k in range(len(QUANTILES))]
    return list(anchor_offsets[ANCHORS[-1]])


def price6(x):
    """x clamped to [0, 1] and rounded to 6 decimals (0, never -0)."""
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0


def forecast_market(prices, horizon, pooled):
    """240 (horizon) rows of five quantiles for one market.

    pooled: {anchor: [pooled quantile offsets]} precomputed for the case.
    """
    n = len(prices)
    if n == 0:
        raise ValueError("market has no prices")
    last = prices[-1]
    window = prices[-SHRINK_WINDOW:] if n > 1 else prices
    mean_recent = sum(window) / len(window)
    median = last + SHRINK * (mean_recent - last)
    hist = prices[-MOVE_WINDOW:] if MOVE_WINDOW else prices

    # Per-anchor offsets: own empirical distribution shrunk toward pooled.
    anchor_offsets = {}
    for a in ANCHORS:
        own, count = move_quantiles(hist, a)
        pool = pooled[a]
        if own is None or count < MIN_OWN_SAMPLES:
            anchor_offsets[a] = list(pool)
            continue
        w = count / (count + POOL_PRIOR)
        anchor_offsets[a] = [own[k] * w + pool[k] * (1.0 - w)
                             for k in range(len(QUANTILES))]

    rows = []
    for h in range(1, horizon + 1):
        offs = interp_offsets(anchor_offsets, h)
        # Recentre on the median: the empirical median move is estimation
        # noise, so the 50% level sits exactly on our median, and the other
        # levels are scaled toward it.
        med_off = offs[2]
        shift = median - last
        root_h = math.sqrt(h)
        vals = []
        for k, tau in enumerate(QUANTILES):
            off = (offs[k] - med_off) * SPREAD_SCALE + shift
            # Floor the spread so flat markets still carry some width.
            floor = FLOOR_OUTER if tau in (0.05, 0.95) else FLOOR_INNER
            if tau < 0.5 and off > -floor * root_h:
                off = -floor * root_h
            elif tau > 0.5 and off < floor * root_h:
                off = floor * root_h
            vals.append(price6(last + off))
        # Enforce non-decreasing order after rounding.
        for k in range(1, len(vals)):
            if vals[k] < vals[k - 1]:
                vals[k] = vals[k - 1]
        rows.append(vals)
    return rows


def forecast_case(case):
    """Forecast every market of one parsed case object."""
    horizon = case["horizon_minutes"]
    levels = case["quantiles"]
    if len(levels) != len(QUANTILES) or any(
            levels[j] != QUANTILES[j] for j in range(len(QUANTILES))):
        raise ValueError("case asks for quantiles %r, not %r" % (levels, QUANTILES))
    markets = case["markets"]
    all_prices = [m["prices"][-MOVE_WINDOW:] if MOVE_WINDOW else m["prices"]
                  for m in markets]
    pooled = {a: pooled_quantiles(all_prices, a) for a in ANCHORS}
    forecasts = []
    for m in markets:
        forecasts.append({"id": m["id"],
                          "q": forecast_market(m["prices"], horizon, pooled)})
    return {"forecasts": forecasts}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inputs = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    if not inputs:
        raise SystemExit("no .in files in %s" % INPUT_DIR)
    for fname in inputs:
        case_id = fname[:-len(".in")]
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":")) + "\n"
        if len(text.encode("utf-8")) > 1048576:
            raise SystemExit("output for %s exceeds 1 MiB" % case_id)
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"),
                  "w", encoding="utf-8") as fh:
            fh.write(text)


if __name__ == "__main__":
    main()

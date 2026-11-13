# polymarket-nvidia-ceo-forecast-01 entry

Forecasts the Yes price of every market of Polymarket's "Next CEO of NVIDIA?"
event at each of the 240 minutes after a checkpoint, as five quantiles
(5%, 25%, 50%, 75%, 95%) per minute. An entry for the make.fun bounty
`polymarket-nvidia-ceo-forecast-01`, scored by mean pinball loss.

## How it works

`main.py` (Python 3.13, standard library only, no network) reads every
`<id>.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes `<id>.out`
to `$MAKEFUN_OUTPUT_DIR` (default `/output`).

Per market, the median is the last price (shrinking it toward the trailing
mean was tried on the practice backtest and hurt, so the shrink is 0). The
spreads come from the market's own empirical distribution of h-minute price
moves over the trailing 18 hours, measured at anchor horizons (1, 2, 4, 8,
16, 32, 64, 128, 240 minutes) and interpolated in sqrt(h), shrunk toward
the pooled cross-market distribution so thin markets borrow strength, and
scaled by 0.5 (raw empirical spreads over-cover badly). A small floor keeps
flat markets from claiming zero spread. All quantiles are clamped to [0, 1],
ordered, and rounded to 6 decimals. Fully deterministic: no randomness,
no clock.

Backtested on the practice cases p00–p04, scored against the next case's
prices as ground truth: mean pinball loss 0.3460 (house baseline 0.3985),
90% coverage 0.933, median error 0.9010 — about 13% below the baseline.

## Run it locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

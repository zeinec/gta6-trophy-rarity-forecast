# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeinec/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/zeinec/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun bounty gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01): forecast how many GTA VI players earn each trophy in week one.

Forecasts, for every PS5 trophy of Grand Theft Auto VI, the percent of
Exophase-tracked players who will hold it at the case's target time, as
five quantiles per trophy. Scored by pinball loss on the logit scale.

## How it works

`rate(t) = Holders(t) / Players(t)` — the two sides are forecast
separately, then divided. That is the whole trick: launch week is a
dilution story, not a level story. Every refresh floods in new players
who hold almost nothing yet, so every rate falls even while holders
only grow.

- **Refreshes only.** Exophase's page is frozen between its own
  refreshes, so consecutive identical reads carry no information. Only
  one observation per refresh is used.
- **Players.** Log players is fit against log hours since the NZ unlock
  (quadratic with 3+ refresh points, linear with 2, a launch-calibrated
  prior curve with fewer). Beyond the observed range it continues as a
  power law with the slope at the last point — a raw quadratic collapses
  at a 7-day horizon, so it is never extrapolated as one.
- **Holders.** Per trophy, across refreshes:
  `dH = a * dN + b * dt * N_prev`, solved through the origin with ridge
  shrinkage toward a prior. With fewer than 2 refresh points the prior
  is used directly: `a = last_rate * earliness(trophy)`, where earliness
  is read off the trophy text (chapter number, activity keywords) and
  calibrated to the measured launch-to-now cohort profile — new players
  hold ~1.04 trophies each, almost all of it the intro trophy.
- **Fan.** Laplace quantiles on the logit scale with a fat lower tail
  (dilution risk is one-sided), widened by N-forecast uncertainty and
  the holder regression's residual.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, offline, deterministic, well under the 8 s
per-case cap (~0.1 s).

## Validation

Local checks (see `work/validate.py` in the build workspace, not part
of the entry): practice cases valid with 55 rows, byte-identical
across runs; launch-to-now backtest against the live Exophase snapshot
of 19 Nov ~15:18 UTC scores far below the house persistence baseline;
a synthetic multi-refresh smoke test recovers the injected dynamics.

Built for [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) — referral `ha8nyspb`.

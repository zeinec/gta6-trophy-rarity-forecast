# gta6-trophy-rarity-forecast-01

**Version:** 1.0.0

**About:** Refresh-aware earners/players trophy forecaster

## Description

A forecast program for the make.fun bounty
**gta6-trophy-rarity-forecast-01** — "Forecast how many GTA VI players
earn each trophy in week one" ($250, top 20 paid). It reads the house's
recorded Exophase reads for Grand Theft Auto VI (PS5) and forecasts every
trophy's earn rate at the case target as five quantiles (5/25/50/75/95%).

**Method:** a trophy's rate is holders / players_tracked, and the two
sides are forecast separately. Exophase's page is frozen between its
refreshes, so only refresh points carry signal. Players N(t) is fit in
log-log time since the New Zealand unlock (quadratic inside the observed
range, damped power law beyond it). Holders H(t) follow
dH = a·dN + b·dt·N_prev across refreshes, fit through the origin and
shrunk toward an earliness prior read from the trophy text (chapter
number, activity type). The median rate is H_target / N_target, with a
Laplace fan on the logit scale that has a fatter lower tail, because a
bigger refresh can only dilute a rate further.

**Validation (local, 21 Nov 2026):**

- Practice cases p00/p01: valid output, byte-identical across runs,
  ~0.05 s per case.
- Launch → now backtest (15.5 h horizon, answer = the live Exophase
  page): pinball loss **0.0975** vs house persistence baseline **2.3370**
  (oracle median + house spread: 0.0793). Typical median miss 0.004 pp,
  90% interval coverage 1.00.
- Synthetic 4-refresh smoke test at the final horizon: loss 0.1799,
  players forecast +6% vs injected truth.

## Installation

No dependencies beyond a Python 3 interpreter (stdlib only, offline,
deterministic). Clone the repo and run `main.py` directly:

```bash
git clone https://github.com/zeinec/gta6-trophy-rarity-forecast.git
cd gta6-trophy-rarity-forecast
python3 main.py
```

## Usage

The program reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default
`/input`) and writes `<id>.out` to `$MAKEFUN_OUTPUT_DIR` (default
`/output`) as `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Contributing

Issues and pull requests are welcome. Please keep the entry stdlib-only
and deterministic: no network access, no randomness, no wall-clock
dependence.

## License

MIT — see [LICENSE](LICENSE).

## Referral

- Referral code: `ha8nyspb`

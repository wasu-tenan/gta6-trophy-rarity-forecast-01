# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wasu-tenan/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/wasu-tenan/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase one week after launch (2026-11-26 00:00 UTC), as five quantiles
per trophy (5%, 25%, 50%, 75%, 95%), from the house's recorded reads of
the game's trophy page. Scored by pinball loss on the logit scale.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one
`<id>.out` per case to the output directory. Python 3.13, standard
library only, no network, deterministic (two runs write byte-identical
output).

## How it works

Exophase prints one rate per trophy — earners over players tracked —
and refreshes its counts every few hours, so reads arrive as flat steps
with jumps. The program picks one of two regimes from the case history:

- **Flat history** (launch night: no read differs from another): a
  two-branch mixture — with probability *p* a refresh lands before the
  target and dilutes every rate (newly tracked players hold far fewer
  trophies), otherwise rates sit where they are. *p* comes from a
  lognormal refresh-interval prior (median 4 h) conditioned on the
  time already spent without a refresh.
- **Moving history** (the final case: 2.5 days of reads): per trophy,
  two recency-weighted least-squares fits against log-time — logit(rate)
  directly, and log(earners) vs log(players tracked) — averaged, then
  blended with persistence by fit quality and capped by an archetype
  ceiling (platinum / 100%-completion trophies stay in the low single
  digits in week one; story trophies can reach the 90s).

Quantiles are normal quantiles on the logit scale around the median.
A linear-time fit was tried and rejected: it overshoots badly on
synthetic finals (loss 0.75–0.84 vs 0.38–0.54 for the log-time pair).

## Validation (local)

- Practice cases p00/p01 against reconstructed 04:30Z answers (the kit
  records 3,868 players tracked, 4,276 trophies earned at the refresh):
  loss **0.61–0.65** vs the persistence baseline's **1.40** on the same
  reconstruction (kit: baseline ≈ 1.51).
- Synthetic final cases (a cohort simulation with refresh dilution,
  a different mechanism from this model): loss **0.38–0.54** vs the
  baseline's **0.85–0.97**, 90% coverage 0.89–0.98.
- Outputs valid (55 rows, monotone quantiles, ≤ 65,536 bytes),
  byte-identical across runs, ~0.2 s per case (limit 8 s), identical
  under UTC, America/New_York and Asia/Jakarta timezones.

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>

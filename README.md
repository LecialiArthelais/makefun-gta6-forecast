# GTA VI Twitch CCU Forecaster — gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/LecialiArthelais/makefun-gta6-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/LecialiArthelais/makefun-gta6-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A forecaster for the make.fun GTA VI Twitch viewers bounty: one hour into the
release day (2026-11-19 UTC) it forecasts, for each of hours 2–24, five
quantiles (5%, 25%, 50%, 75%, 95%) of the concurrent viewers in Twitch's
Grand Theft Auto VI category. Scored by mean pinball loss on the log scale;
a prize needs a loss strictly below the best house entry.

## Strategy

Log-ratio shape transfer from comparable launches. For every past big launch
in the case file, the launch-day shape is the log-ratio curve
`r(h) = ln(viewers(h) / viewers(1))` for hours 2–24. The forecaster fits, per
hour, a kernel-weighted local-linear regression of that curve on the launch's
log hour-1 size, then takes the spread from the residuals — so the median
follows how big launches' days actually bend (early dip, US-unlock wave,
evening prime-time peak) instead of assuming every hour stays at hour 1.

- **Anchor:** the case's hour-1 print — the only hard fact about GTA VI's day.
- **Donors:** the case's comparables (38 in practice, 42 in the final),
  weighted by similarity in log hour-1 size (Gaussian kernel, bandwidth 1.0).
- **Median:** per-hour kernel-weighted ridge regression of the log-ratio on
  three features — log hour-1 size, the eve ratio (viewers the hour before
  launch, over hour 1) and the 7-day pre-release mean ratio. The eve/pre7
  features are the big win: they let the model tell a hyped big launch
  (big evening peak) from an ordinary big launch.
- **Spread:** per-hour residual spread (x0.85), widened by a
  prediction-interval leverage term (capped at sqrt(2)) — a target at or
  beyond the edge of the donor cloud honestly gets a wider 5–95% band.
- **Deterministic:** no clock, no randomness — every quantity is a pure
  function of the case file, so two runs are byte-identical.

## Validation

Practice answers were reconstructed from SullyGnome's public hourly
averages — the same source the house scored the practice on — and the
scorer was verified to reproduce the house baseline's practice loss of
0.3871 exactly. Mean pinball loss (lower is better):

| Method | Practice (4 launches) | LOO all donors | LOO hour-1 ≥ 50k |
|---|---:|---:|---:|
| House baseline (persistence) | 0.3871 | — | — |
| Size-only local-linear (v1) | 0.1894 | 0.390 | 0.202 |
| **This entry (size + eve + pre7, leverage spread)** | **0.1538** | **0.361** | **0.200** |
| Dev leaderboard top-1 / top-20 line / median | 0.1140 / 0.135 / 0.211 | — | — |

Per practice launch: Wolverine 0.167, Red Dead Redemption 2 0.152, God of
War Ragnarök 0.111, Spider-Man 2 0.185. Median miss 25.8%, 90% coverage
0.935 (nominal 0.90 — the band is honest). Runtime ~0.1s per case
(limit 8s). Two runs byte-identical. Edge cases checked: no eve/pre7
history, hour-1 of 500 and 5,000,000 viewers, only 5 donors — all produce
23 finite, monotone quantile rows.

Honest read: this beats the house by ~2.5x and sits just above the dev
top-20 line on practice. The final is a single case (GTA VI itself), so
the prize is a draw, not a guarantee — but the method is validated
out-of-sample (leave-one-out) rather than fitted to the four practice
cases: adding the eve/pre7 features improves practice AND leave-one-out,
and the per-hour bias correction and the own-day pseudo-donor ideas were
tested and rejected (they helped practice but hurt leave-one-out, or did
nothing).

## Run it

```sh
node main.ts   # the grader runs `node /bot/main.ts` with this directory at /bot
```

Node 22, standard library only, no network. Reads `$MAKEFUN_INPUT_DIR`
(default `/input`), writes `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## License

MIT — see [LICENSE](LICENSE).

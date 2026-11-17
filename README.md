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
  weighted by similarity in log hour-1 size (Gaussian kernel).
- **Median:** per-hour weighted local-linear fit of the log-ratio on log size.
- **Spread:** per-hour residual spread, calibrated on leave-one-out residuals
  of the donors, so the 5–95% band is honest about how much launch days vary.
- **Deterministic:** no clock, no randomness — every quantity is a pure
  function of the case file, so two runs are byte-identical.

## Validation

(Numbers below are filled in once the final validation run lands.)

## Run it

```sh
node main.ts   # the grader runs `node /bot/main.ts` with this directory at /bot
```

Node 22, standard library only, no network. Reads `$MAKEFUN_INPUT_DIR`
(default `/input`), writes `$MAKEFUN_OUTPUT_DIR` (default `/output`).

## License

MIT — see [LICENSE](LICENSE).

# Project Map

Rayn Strategy Research is a Python project for offline crypto-market strategy
experiments. It is not connected to an exchange account.

## Core Code

- `src/rayn_strategy/backtest.py` - candle loading, backtest primitives, equity records, and trade records.
- `src/rayn_strategy/grid.py` - bounded grid research engine.
- `src/rayn_strategy/momentum.py` - momentum engine for long and short research.
- `src/rayn_strategy/exchange_filters.py` - pure offline helpers for minimum-size feasibility checks.
- `src/rayn_strategy/signal_check.py` - read-only signal evaluation over provided candle data.
- `src/rayn_strategy/paper_account.py` - local simulated account snapshots.

## Main Tools

- `run_backtest.py` - baseline strategy backtest runner.
- `run_grid_backtest.py` - bounded-grid backtest runner.
- `tools/walk_forward.py` - monthly validation for baseline profiles.
- `tools/walk_forward_grid.py` - monthly validation for bounded-grid profiles.
- `tools/combined_momentum_backtest.py` - combined long/short comparison runner.
- `tools/sim_review_report.py` - simulated-account review renderer.
- `tools/fetch_binance_klines.py` - public kline fetcher.
- `tools/fetch_binance_funding.py` - public funding-rate fetcher.

## Profiles

- `configs/active/` - current comparison profiles.
- `configs/research/` - research and stress-test profiles.
- `configs/baselines/` - examples and baseline profiles.

## Data And Artifacts

- `data*/` directories contain local market CSVs and are mostly ignored.
- `reports/evidence/` contains generated experiment outputs.
- `reports/runtime/` contains local scratch state and caches.
- `reports/README.md` explains the artifact policy.

## Verification

Run the project test suite from the repository root:

```bash
python3 -m pytest -q
```

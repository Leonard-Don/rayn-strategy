# Rayn Strategy Research

[![CI](https://github.com/Leonard-Don/rayn-strategy/actions/workflows/ci.yml/badge.svg)](https://github.com/Leonard-Don/rayn-strategy/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-blue)

Rayn Strategy Research is a Python scaffold for crypto-market strategy research,
offline backtesting, stress testing, and simulated monitoring.

This repository is intentionally account-disconnected:

- no exchange-account integration
- no signed exchange requests
- no automated trade placement
- no generated runtime evidence committed as source

It is research software only, not investment advice.

## What Is Included

- bounded grid and momentum research engines
- risk checks, market filters, and indicator helpers
- offline backtest runners
- public market-data fetchers for klines and funding rates
- TOML strategy profiles
- unit and behavior tests

## Layout

- `src/rayn_strategy/` - core engines, indicators, risk logic, and config loading.
- `configs/active/` - current comparison profiles.
- `configs/research/` - research and stress-test profiles.
- `configs/baselines/` - examples and baseline profiles.
- `tools/` - data fetchers, analysis tools, and report helpers.
- `data/README.md` - expected market-data CSV format.
- `reports/README.md` - artifact policy for generated reports.
- `tests/` - regression tests.

## Quick Start

Install test dependencies in your preferred Python environment, then run:

```bash
python3 -m pytest -q
```

Put CSV files in `data/`, one file per symbol:

```text
BTCUSDT.csv
ETHUSDT.csv
SOLUSDT.csv
```

Each CSV should contain:

```text
timestamp,open,high,low,close,volume
```

Run a baseline backtest:

```bash
python3 run_backtest.py \
  --data data \
  --config configs/baselines/config.optimized.toml \
  --symbols BTCUSDT,ETHUSDT,ZECUSDT \
  --trades-out reports/evidence/example-trades.csv
```

Fetch public Binance USD-M kline data:

```bash
python3 tools/fetch_binance_klines.py \
  --symbols BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT \
  --interval 1h \
  --start 2026-02-01 \
  --end 2026-05-31 \
  --out data \
  --skip-errors
```

Run a bounded-grid research backtest:

```bash
python3 run_grid_backtest.py \
  --data data \
  --config configs/research/config.bounded-grid-research.toml \
  --symbols BTCUSDT,ETHUSDT,ZECUSDT \
  --trades-out reports/evidence/grid-example-trades.csv
```

Run monthly forward-window validation:

```bash
python3 tools/walk_forward.py \
  --data data \
  --config configs/baselines/config.optimized.toml \
  --symbols BTCUSDT,ETHUSDT,ZECUSDT \
  --start 2026-02-01 \
  --first-test 2026-03-01 \
  --end 2026-05-31 \
  --out reports/evidence/walk-forward-example.csv
```

## Research Notes

Generated CSV, PNG, JSON, and Markdown outputs belong under `reports/evidence/`
or `reports/runtime/` and are ignored by default. Keep durable methodology in
source-controlled docs only after removing local account state and generated
cache data.

## Safety Boundary

The project should remain an offline research repository. Any future account
connector, signing client, or automated trading path should stay outside this
public codebase unless it is reviewed and intentionally re-scoped.

## License

MIT License. See [LICENSE](LICENSE).

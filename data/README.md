# Data Format

Place one CSV file per symbol in this folder.

Required filename format:

```text
BTCUSDT.csv
ETHUSDT.csv
```

Required columns:

```text
timestamp,open,high,low,close,volume
```

Accepted timestamp formats:

- Unix seconds
- Unix milliseconds
- ISO strings, for example `2026-05-31T12:00:00Z`

Optional future columns:

```text
quote_volume,spread_bps
```

The current backtester ignores optional columns, but the risk layer is designed
to add liquidity and spread filters later.


#!/usr/bin/env python3
"""Record a local paper-account mark-to-market snapshot.

This reads local paper state and market data only. It never places orders.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.rayn_strategy.backtest import load_candles
from src.rayn_strategy.paper_account import append_snapshot_csv, build_paper_account_snapshot
from src.rayn_strategy.paths import repo_output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--label", default="rayn_sim")
    parser.add_argument("--csv", default="reports/runtime/rayn-sim-account-equity.csv")
    parser.add_argument("--json-out", default="reports/runtime/rayn-sim-account-latest.json")
    args = parser.parse_args()

    state_path = ROOT / args.state if not Path(args.state).is_absolute() else Path(args.state)
    data_dir = ROOT / args.data if not Path(args.data).is_absolute() else Path(args.data)
    symbols = [item.strip().upper() for item in args.symbols.split(",") if item.strip()]

    state = json.loads(state_path.read_text())
    rows = {symbol: load_candles(data_dir / f"{symbol}.csv") for symbol in symbols}
    timestamp = max(rows[symbol][-1].timestamp for symbol in symbols)
    marks = {symbol: rows[symbol][-1].close for symbol in symbols}
    snapshot = build_paper_account_snapshot(state, marks, timestamp=timestamp, label=args.label)

    csv_path = repo_output_path(args.csv)
    json_path = repo_output_path(args.json_out)
    append_snapshot_csv(csv_path, snapshot)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True))
    print(json.dumps(snapshot, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fetch public Binance USD-M futures klines into the local CSV format.

This script uses only public market-data endpoints and does not need API keys.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.client import IncompleteRead, RemoteDisconnected
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


BASE_URL = "https://fapi.binance.com/fapi/v1/klines"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance futures klines.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--interval", default="1h", help="Kline interval, default 1h.")
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2024-01-01.")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2026-05-31.")
    parser.add_argument("--out", default="rayn/data", help="Output folder.")
    parser.add_argument(
        "--request-limit",
        type=int,
        default=500,
        help="Klines per request. Smaller chunks reduce proxy/Binance partial-read failures.",
    )
    parser.add_argument("--skip-errors", action="store_true", help="Continue if one symbol fails.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ms = to_ms(args.start)
    end_ms = to_ms(args.end)

    for symbol in [item.strip().upper() for item in args.symbols.split(",") if item.strip()]:
        try:
            rows = fetch_symbol(symbol, args.interval, start_ms, end_ms, args.request_limit)
        except RuntimeError as exc:
            if not args.skip_errors:
                raise
            print(f"{symbol}: skipped ({exc})")
            continue
        write_csv(out_dir / f"{symbol}.csv", rows)
        print(f"{symbol}: wrote {len(rows)} rows")
        time.sleep(0.2)
    return 0


def fetch_symbol(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    request_limit: int,
) -> list[list[object]]:
    rows: list[list[object]] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": request_limit,
        }
        url = f"{BASE_URL}?{urlencode(params)}"
        batch = read_json_url(url)
        if isinstance(batch, dict):
            code = batch.get("code", "unknown")
            message = batch.get("msg", "unknown error")
            raise RuntimeError(f"Binance error {code}: {message}")
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        next_cursor = last_open + interval_ms(interval)
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    return rows


def read_json_url(url: str) -> object:
    last_error: Exception | None = None
    for attempt in range(1, 9):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "rayn-research-fetcher/0.1",
            },
        )
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            try:
                body = exc.read().decode("utf-8")
            except Exception:
                body = str(exc)
            raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
        except (IncompleteRead, RemoteDisconnected, TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"request failed after retries: {last_error}")


def write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for row in rows:
            writer.writerow(
                [
                    datetime.fromtimestamp(int(row[0]) / 1000, tz=timezone.utc).isoformat(),
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                ]
            )


def to_ms(value: str) -> int:
    item = value.strip()
    if len(item) == 10:
        item = item + "T00:00:00+00:00"
    if item.endswith("Z"):
        item = item[:-1] + "+00:00"
    parsed = datetime.fromisoformat(item)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def interval_ms(interval: str) -> int:
    unit = interval[-1]
    amount = int(interval[:-1])
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
    }
    if unit not in multipliers:
        raise ValueError(f"unsupported interval: {interval}")
    return amount * multipliers[unit]


if __name__ == "__main__":
    raise SystemExit(main())

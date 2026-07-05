#!/usr/bin/env python3
"""Fetch public Binance USD-M funding rates into local CSV files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.client import IncompleteRead
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance funding rate history.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2026-02-01.")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2026-05-31.")
    parser.add_argument("--out", default="rayn/data/funding", help="Output folder.")
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
            rows = fetch_symbol(symbol, start_ms, end_ms)
        except RuntimeError as exc:
            if not args.skip_errors:
                raise
            print(f"{symbol}: skipped ({exc})")
            continue
        write_csv(out_dir / f"{symbol}.csv", rows)
        print(f"{symbol}: wrote {len(rows)} funding rows")
        time.sleep(0.2)
    return 0


def fetch_symbol(symbol: str, start_ms: int, end_ms: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cursor = start_ms
    while cursor <= end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        batch = read_json_url(f"{BASE_URL}?{urlencode(params)}")
        if isinstance(batch, dict):
            raise RuntimeError(f"Binance error {batch.get('code')}: {batch.get('msg')}")
        if not batch:
            break
        rows.extend(batch)
        last_time = int(batch[-1]["fundingTime"])
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    return rows


def read_json_url(url: str) -> object:
    last_error: Exception | None = None
    for attempt in range(1, 5):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "User-Agent": "rayn-funding-fetcher/0.1",
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
        except (IncompleteRead, TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"request failed after retries: {last_error}")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "funding_rate", "mark_price"])
        for row in rows:
            writer.writerow(
                [
                    datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=timezone.utc).isoformat(),
                    row["fundingRate"],
                    row.get("markPrice", "0"),
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


if __name__ == "__main__":
    raise SystemExit(main())

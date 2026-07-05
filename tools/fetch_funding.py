#!/usr/bin/env python3
"""Fetch public Binance USD-M futures funding-rate history into local CSVs.

This script uses only the public market-data endpoint and does not need API keys.
Funding settles every 8h (00:00/08:00/16:00 UTC). The fundingRate endpoint
returns at most 1000 records per request, so we paginate forward in time using
the last fundingTime + 1 as the next startTime until we reach endTime or get an
empty page.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.client import IncompleteRead
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import json


BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Binance futures funding rates.")
    parser.add_argument("--symbols", required=True, help="Comma-separated symbols.")
    parser.add_argument("--start", required=True, help="UTC start, e.g. 2021-01-01.")
    parser.add_argument("--end", required=True, help="UTC end, e.g. 2025-10-13.")
    parser.add_argument("--out", default="data_funding", help="Output folder.")
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
        if rows:
            first_ts = iso_utc(rows[0][0])
            last_ts = iso_utc(rows[-1][0])
            print(f"{symbol}: wrote {len(rows)} rows ({first_ts} -> {last_ts})")
        else:
            print(f"{symbol}: wrote 0 rows")
        time.sleep(0.2)
    return 0


def fetch_symbol(symbol: str, start_ms: int, end_ms: int) -> list[tuple[int, float]]:
    """Return sorted, de-duplicated (fundingTime_ms, fundingRate) tuples."""
    seen: dict[int, float] = {}
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        }
        url = f"{BASE_URL}?{urlencode(params)}"
        batch = read_json_url(url)
        if isinstance(batch, dict):
            code = batch.get("code", "unknown")
            message = batch.get("msg", "unknown error")
            raise RuntimeError(f"Binance error {code}: {message}")
        if not batch:
            break
        for entry in batch:
            ftime = int(entry["fundingTime"])
            if ftime > end_ms:
                continue
            seen[ftime] = float(entry["fundingRate"])
        last_time = int(batch[-1]["fundingTime"])
        next_cursor = last_time + 1
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)
    return sorted(seen.items())


def read_json_url(url: str) -> object:
    last_error: Exception | None = None
    for attempt in range(1, 5):
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
        except (IncompleteRead, TimeoutError, URLError) as exc:
            last_error = exc
            time.sleep(0.5 * attempt)
    raise RuntimeError(f"request failed after retries: {last_error}")


def write_csv(path: Path, rows: list[tuple[int, float]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "funding_rate"])
        for ftime, rate in rows:
            writer.writerow([iso_utc(ftime), rate])


def iso_utc(ftime_ms: int) -> str:
    return (
        datetime.fromtimestamp(ftime_ms / 1000, tz=timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ")
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

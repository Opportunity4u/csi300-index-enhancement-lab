from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import load_price_panel

KLINE_COLUMNS = [
    "Date",
    "Open",
    "Close",
    "High",
    "Low",
    "Volume",
    "Amount",
    "Amplitude",
    "PctChange",
    "Change",
    "Turnover",
]


@dataclass(frozen=True)
class MarketDataUpdate:
    prices: pd.DataFrame
    index_prices: pd.DataFrame
    log: pd.DataFrame
    latest_market_date: pd.Timestamp | None


def _stock_secid(ticker: str) -> str:
    return f"1.{ticker}" if ticker.startswith(("5", "6", "9")) else f"0.{ticker}"


def _yahoo_symbol(ticker: str) -> str:
    suffix = ".SS" if ticker.startswith(("5", "6", "9")) else ".SZ"
    return f"{ticker}{suffix}"


def _fetch_json(url: str, retries: int = 3, timeout: int = 30) -> dict:
    headers = {"User-Agent": "Mozilla/5.0 CSI300ForwardMonitor/1.0"}
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - network path
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"market-data request failed after {retries} attempts") from last_error


def download_eastmoney_kline(
    identifier: str,
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    index: bool = False,
) -> pd.DataFrame:
    """Download one adjusted daily K-line series without persisting raw responses."""
    ticker = str(identifier).zfill(6)
    secid = f"1.{ticker}" if index else _stock_secid(ticker)
    params = {
        "secid": secid,
        "klt": "101",
        # The existing research panel is based on Yahoo's unadjusted close.
        # Keep this fallback unadjusted as well so overlapping rows cannot
        # introduce artificial corporate-action jumps.
        "fqt": "0",
        "beg": pd.Timestamp(start).strftime("%Y%m%d"),
        "end": pd.Timestamp(end).strftime("%Y%m%d"),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url)
    rows = ((payload.get("data") or {}).get("klines") or [])
    if not rows:
        return pd.DataFrame(columns=["Ticker", *KLINE_COLUMNS])
    values = [row.split(",") for row in rows]
    frame = pd.DataFrame(values, columns=KLINE_COLUMNS)
    frame.insert(0, "Ticker", ticker)
    frame["Date"] = pd.to_datetime(frame["Date"])
    for column in KLINE_COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def download_yahoo_spark(
    identifiers: list[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    index: bool = False,
) -> pd.DataFrame:
    """Download recent daily closes from Yahoo's batch Spark endpoint."""
    tickers = [str(identifier).zfill(6) for identifier in identifiers]
    symbols = ["000300.SS"] if index else [_yahoo_symbol(ticker) for ticker in tickers]
    params = {
        "symbols": ",".join(symbols),
        "range": "1mo",
        "interval": "1d",
    }
    url = "https://query1.finance.yahoo.com/v7/finance/spark?" + urllib.parse.urlencode(params)
    payload = _fetch_json(url, retries=2, timeout=20)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    rows: list[dict] = []
    for item in (payload.get("spark") or {}).get("result") or []:
        ticker = str(item.get("symbol", "")).split(".")[0].zfill(6)
        responses = item.get("response") or []
        if not responses:
            continue
        response = responses[0]
        timestamps = response.get("timestamp") or []
        quote = ((response.get("indicators") or {}).get("quote") or [{}])[0]
        closes = quote.get("close") or []
        for timestamp, close in zip(timestamps, closes, strict=False):
            if close is None:
                continue
            date = (
                pd.to_datetime(timestamp, unit="s", utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None)
                .normalize()
            )
            if start_date <= date <= end_date:
                value = float(close)
                rows.append({
                    "Ticker": ticker,
                    "Date": date,
                    "Open": value,
                    "Close": value,
                    "High": value,
                    "Low": value,
                    "Volume": pd.NA,
                    "Amount": pd.NA,
                    "Amplitude": pd.NA,
                    "PctChange": pd.NA,
                    "Change": pd.NA,
                    "Turnover": pd.NA,
                })
    return pd.DataFrame(rows, columns=["Ticker", *KLINE_COLUMNS])


def update_price_panel(
    price_path: Path,
    tickers: list[str],
    as_of: pd.Timestamp | str,
    *,
    lookback_calendar_days: int = 14,
    max_workers: int = 8,
) -> MarketDataUpdate:
    """Incrementally refresh a normalized private price panel.

    The caller owns the data file. Only normalized rows are written; raw API
    payloads are neither retained nor copied into the public repository.
    """
    as_of = pd.Timestamp(as_of).normalize()
    existing = load_price_panel(price_path)
    latest = existing["Date"].max()
    start = min(latest - pd.Timedelta(days=lookback_calendar_days), as_of)
    normalized_tickers = sorted({str(ticker).zfill(6) for ticker in tickers})

    frames: list[pd.DataFrame] = []
    source_by_ticker: dict[str, str] = {}
    error_by_ticker: dict[str, str] = {}

    # Yahoo is the primary incremental source because the historical seed is
    # Yahoo close data. A failed batch is retried ticker-by-ticker through the
    # Eastmoney adapter; neither source is allowed to silently reuse stale data.
    batches = [
        normalized_tickers[offset : offset + 20]
        for offset in range(0, len(normalized_tickers), 20)
    ]

    def fetch_yahoo(batch: list[str]) -> tuple[list[str], pd.DataFrame, str | None]:
        try:
            return batch, download_yahoo_spark(batch, start, as_of), None
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - network path
            return batch, pd.DataFrame(), str(exc)

    with ThreadPoolExecutor(max_workers=min(max_workers, 4)) as pool:
        futures = {pool.submit(fetch_yahoo, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch, frame, error = future.result()
            if not frame.empty:
                frames.append(frame)
                for ticker in frame["Ticker"].unique():
                    source_by_ticker[str(ticker)] = "YahooSpark"
            if error:
                for ticker in batch:
                    error_by_ticker[ticker] = error

    fetched_tickers = set(source_by_ticker)
    fallback_tickers = [ticker for ticker in normalized_tickers if ticker not in fetched_tickers]

    def fetch(ticker: str) -> tuple[str, pd.DataFrame, str | None]:
        try:
            return ticker, download_eastmoney_kline(ticker, start, as_of), None
        except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - network path
            return ticker, pd.DataFrame(), str(exc)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch, ticker): ticker for ticker in fallback_tickers}
        for future in as_completed(futures):
            ticker, frame, error = future.result()
            if not frame.empty:
                frames.append(frame)
                source_by_ticker[ticker] = "Eastmoney"
                error_by_ticker.pop(ticker, None)
            elif error:
                error_by_ticker[ticker] = error

    downloaded = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    log_rows = []
    for ticker in normalized_tickers:
        ticker_rows = (
            downloaded[downloaded["Ticker"] == ticker]
            if not downloaded.empty
            else pd.DataFrame()
        )
        log_rows.append({
            "Ticker": ticker,
            "Source": source_by_ticker.get(ticker),
            "Rows": len(ticker_rows),
            "LatestDate": (
                ticker_rows["Date"].max() if not ticker_rows.empty else pd.NaT
            ),
            "Error": error_by_ticker.get(ticker),
        })

    if frames:
        update = pd.concat(frames, ignore_index=True)
        columns = list(dict.fromkeys([*existing.columns, *update.columns]))
        merged = pd.concat(
            [existing.reindex(columns=columns), update.reindex(columns=columns)],
            ignore_index=True,
        )
        merged = merged.drop_duplicates(["Date", "Ticker"], keep="last")
        merged = merged.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        temporary = price_path.with_name(f"{price_path.name}.tmp")
        merged.to_csv(temporary, index=False, compression="gzip")
        temporary.replace(price_path)
    else:
        merged = existing

    try:
        index_prices = download_yahoo_spark(["000300"], start, as_of, index=True)
    except (OSError, RuntimeError, ValueError):  # pragma: no cover - network path
        index_prices = download_eastmoney_kline("000300", start, as_of, index=True)
    latest_market_date = None if index_prices.empty else pd.Timestamp(index_prices["Date"].max())
    return MarketDataUpdate(
        prices=merged,
        index_prices=index_prices,
        log=pd.DataFrame(log_rows).sort_values("Ticker").reset_index(drop=True),
        latest_market_date=latest_market_date,
    )

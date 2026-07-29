# Data contract

Place user-provided files in `<root>/data/raw/`.

## `prices.csv.gz`

One row per trading date and security.

| Column | Required | Meaning |
|---|---|---|
| `Date` | yes | Trading date |
| `Ticker` | yes | Six-digit security identifier |
| `Close` | yes | Consistently adjusted close or total-return price |
| `Amount` | no | Daily traded value in CNY; enables liquidity constraints |
| `Open`, `Volume` | no | Preserved but not required by the default factors |

`Date x Ticker` must be unique and `Close` must be positive.

## `csi300_weights.csv`

| Column | Required | Meaning |
|---|---|---|
| `AsOfDate` | yes | Effective date of the snapshot |
| `Ticker` | yes | Six-digit security identifier |
| `BenchmarkWeight` | yes | Non-negative index weight; normalized on load |

The public adapter currently calibrates a fixed-share proxy from the latest
snapshot. For institutional research, replace this adapter with dated
point-in-time weights.

## `csi300_constituents.csv`

| Column | Required | Meaning |
|---|---|---|
| `AsOfDate` | yes | Effective membership date |
| `Ticker` | yes | Six-digit security identifier |

Never backfill present-day membership into the past without labeling the
result as a survivorship-biased approximation.

# Methodology contract

This project studies a long-only CSI 300 index-enhancement portfolio. Signal,
target portfolio, execution and realized portfolio are intentionally separate.

## Timing

Features at decision date `t` use information dated `t` or earlier. The
five-trading-day target ends at `t+5`. A training row is eligible only when that
target end date is strictly earlier than the current prediction date. A signal
formed at the close of `t` is executed at the next trading close; the resulting
weight applies to subsequent close-to-close returns.

## Benchmark approximation

The illustrative empirical snapshot uses current official constituents and
closing weights. Fixed proxy share units are calibrated on the snapshot date
and carried backward with prices. This is an explicit approximation, not
official point-in-time history. Production inputs should provide dated
`Date x Ticker x BenchmarkWeight` observations.

## Factors

All features are winsorized at the 1st/99th cross-sectional percentiles and
standardized each date. The fixed feature set is 20-5 momentum, 60-5 momentum,
5-day reversal, 20-day low volatility, 20-day low downside volatility and
60-day drawdown.

## Portfolio and costs

The active portfolio is funded by benchmark underweights and constrained to
long-only. Active weights are scaled using trailing covariance when ex-ante
tracking error exceeds its cap. Actual trades are computed from drifted
holdings, not stale prior targets. Small changes are ignored, turnover is
capped, and asymmetric transaction costs are deducted from NAV. ADV
participation and square-root impact are enabled only when reliable historical
traded-amount data are supplied.

## Interpretation

Only out-of-sample, after-cost results may be presented as the main finding.
Fixed-current-universe results are bias-disclosed research evidence, not live
performance or investment advice.

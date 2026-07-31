# Forward monitoring protocol

The forward monitor is deliberately separate from the historical backtest.
Historical results retain their monthly walk-forward refit schedule. The
forward process attempts a refit after every close while freezing the feature
set, five-session target, Ridge penalty and portfolio constraints.

## Timing

1. Refresh the private normalized panel after the market close.
2. Reject stale or incomplete data before fitting or trading.
3. Train only on weekly observations whose five-session labels ended strictly
   before the current knowledge date.
4. Save a daily diagnostic forecast. It cannot be evaluated using the same
   day's return.
5. Evaluate the forecast from five sessions earlier using its now-realized
   cross-section.
6. Freeze a formal target only after a Friday-ended week is complete. A partial
   Tuesday data cutoff is never treated as a Friday decision.
7. Execute the queued target on the next observed market close, after board-lot,
   no-trade-band, turnover and cost constraints.

## Health states

- `GREEN_WARMUP`: fewer than 20 forward observations or 10 matured forecasts.
- `GREEN`: the 20-observation mean Rank IC and cost-aware active performance are
  non-negative.
- `YELLOW`: three consecutive matured Rank IC readings are negative, or the
  20-observation mean Rank IC is below zero.
- `RED`: mean Rank IC is below -0.02 together with 20-session active return below
  -1%, or active drawdown is worse than -2%. New active risk is then frozen to
  the benchmark target.

Only coefficients can update automatically. Factor definitions, lookbacks,
model class and constraints require a separately reviewed champion-challenger
change after at least 60 forward market sessions.

## Public/private boundary

The private root contains prices, predictions, coefficients, target weights,
holdings, fills and failure diagnostics. The public record contains dates,
coverage, model version, training cutoff, matured Rank IC, decile spread,
paper-account active return, turnover, cost, tracking error, information ratio
and health state. No ticker-level signal, position or order is published.

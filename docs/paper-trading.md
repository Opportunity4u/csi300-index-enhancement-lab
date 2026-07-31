# Paper-trading bridge

The research engine does not submit live orders. Its paper-trading bridge
creates a reviewable order packet that can be entered into a simulator such as
TradingView Paper Trading.

## Why keep this layer separate

Model output is not an order. Between the two are:

- the latest actual positions and cash;
- board-lot rounding;
- a minimum trade size;
- signal and execution timestamps;
- user review and platform-specific symbol mapping.

The helper `target_weights_to_board_lot_orders` converts long-only target
weights to 100-share board lots, rounds toward zero and emits `BUY`/`SELL`
instructions. It deliberately performs no automatic network submission.

## Suggested shadow-account protocol

1. Freeze a weekly decision packet after Friday's close.
2. Record the model's latest permissible training-label end date.
3. Generate proposed orders from the actual paper-account positions.
4. Enter orders at the next supported session and record fills, rejects and
   slippage separately from model targets.
5. Reconcile paper-account positions against the local ledger.
6. Publish monthly tear sheets showing gross alpha, costs, turnover, tracking
   error, rejected orders and target-versus-actual drift.

This creates forward evidence without pretending that simulated fills are live
fund performance.

## Automated local ledger

The `monitor` command adds a fully local, broker-independent paper ledger. It
marks holdings at each close, executes a queued Friday target on the next
observed market session, applies board-lot rounding and modeled buy/sell costs,
and preserves target-versus-actual differences. Complete holdings and fills
remain outside the public repository. The public forward record contains only
aggregate performance and model-health diagnostics.

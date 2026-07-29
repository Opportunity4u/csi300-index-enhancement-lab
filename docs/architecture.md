# Architecture and invariants

## Information timing

At decision close `t`, every feature uses information dated `t` or earlier.
The five-day target is known only after its target date. A training row is
eligible only when:

```text
target_end_date < current_model_date
```

The target formed at `t` is mapped to the next available trading date. It
therefore cannot earn the return already observed at `t`.

## Portfolio state

The engine distinguishes:

1. benchmark weights;
2. model-implied target weights;
3. pre-trade weights after market drift;
4. executed weights after no-trade, turnover and liquidity constraints.

Returns are earned by actual holdings. Costs are charged only on executed
weight changes.

## Required invariants

- Portfolio weights sum to one.
- Long-only weights never fall below zero.
- Matched active trades have zero net cash flow before costs.
- No model training label extends into or beyond its prediction date.
- Turnover and costs cannot be negative.
- Decision dates precede execution dates.

# Contributing

Contributions that improve correctness, reproducibility or implementation
realism are welcome.

## Development

```bash
python -m pip install -e ".[dev]"
ruff check src tests
pytest
```

Keep changes focused and add a regression test for behavioral fixes.

## Research standards

- Never introduce a feature using information unavailable at the decision time.
- State the execution convention and delay explicitly.
- Evaluate strategy changes out of sample and after costs.
- Do not commit vendor data, credentials or proprietary index files.
- Clearly distinguish synthetic, historical simulation, paper trading and live
  performance.

Open an issue before proposing a major new data adapter or portfolio objective.

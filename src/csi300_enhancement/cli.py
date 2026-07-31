from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ResearchConfig
from .monitor import run_forward_monitor
from .pipeline import run_pipeline
from .synthetic import generate_synthetic_inputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="csi300-research",
        description="Run the benchmark-aware CSI 300 enhancement research pipeline.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run on normalized user-provided CSV files")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--oos-start", default="2021-01-01")

    demo = sub.add_parser("demo", help="generate deterministic synthetic data and run")
    demo.add_argument("--root", type=Path, default=Path("demo_run"))
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--tickers", type=int, default=120)

    monitor = sub.add_parser("monitor", help="run one forward daily monitoring cycle")
    monitor.add_argument("--root", type=Path, required=True, help="private data root")
    monitor.add_argument("--public-root", type=Path, default=Path.cwd())
    monitor.add_argument("--as-of", default=None, help="YYYY-MM-DD; defaults to today")
    monitor.add_argument("--no-update-data", action="store_true")
    monitor.add_argument("--publish", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "monitor":
        output = run_forward_monitor(
            root,
            args.public_root,
            as_of=args.as_of,
            update_data=not args.no_update_data,
            publish=args.publish,
        )
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    if args.command == "demo":
        generate_synthetic_inputs(root, n_tickers=args.tickers, seed=args.seed)
        config = ResearchConfig(end_date="2025-12-31", oos_start="2021-01-01")
    else:
        config = ResearchConfig(oos_start=args.oos_start)
    output = run_pipeline(root, config)
    print(output["metrics"].to_string(index=False))
    print(f"\nResults written to: {root / 'results'}")


if __name__ == "__main__":
    main()

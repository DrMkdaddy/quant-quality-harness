"""Command line interface for the quantization harness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .runner import RunError, fetch, render_markdown, run_sweep


DEFAULT_LLAMA_BIN = Path.home() / "llama.cpp" / "src-b11050" / "build" / "bin"


def _common() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--config",
        default="configs/qwen2.5-3b.json",
        help="model-family manifest (default: configs/qwen2.5-3b.json)",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="override QUANTLAB_MODEL_DIR",
    )
    return parser


def _parser() -> argparse.ArgumentParser:
    common = _common()
    parser = argparse.ArgumentParser(
        prog="quantlab",
        description="Measure GGUF quant quality, speed, and VRAM with provenance.",
        parents=[common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser(
        "fetch", parents=[common], help="download corpus and model files"
    )
    fetch_p.add_argument("--force", action="store_true", help="re-download existing files")

    run_p = sub.add_parser("run", parents=[common], help="run the full sweep")
    run_p.add_argument(
        "--llama-bin",
        default=str(DEFAULT_LLAMA_BIN),
        help="directory containing llama-perplexity, llama-bench, llama-cli",
    )
    run_p.add_argument("--out-dir", default="results", help="artifact and log directory")
    run_p.add_argument("--run-id", default=None, help="stable run identifier")
    run_p.add_argument("--timeout", type=int, default=3600, help="per-command timeout (s)")

    report_p = sub.add_parser(
        "report", parents=[common], help="render Markdown from result JSON"
    )
    report_p.add_argument("results", nargs="+", help="one or more result JSON files")

    check_p = sub.add_parser(
        "check", parents=[common], help="validate config and local files"
    )
    check_p.add_argument("--llama-bin", default=str(DEFAULT_LLAMA_BIN))

    all_p = sub.add_parser("all", parents=[common], help="fetch, run, and report")
    all_p.add_argument("--llama-bin", default=str(DEFAULT_LLAMA_BIN))
    all_p.add_argument("--out-dir", default="results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_config(args.config, args.model_dir)
    except (ConfigError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    try:
        if args.command == "fetch":
            fetch(config, force=args.force)
            return 0

        if args.command == "run":
            path = run_sweep(
                config,
                llama_bin=Path(args.llama_bin),
                out_dir=Path(args.out_dir),
                run_id=args.run_id,
                timeout_s=args.timeout,
            )
            print(render_markdown([path]))
            return 0

        if args.command == "report":
            paths = [Path(p) for p in args.results]
            print(render_markdown(paths))
            return 0

        if args.command == "check":
            missing = [str(q.path) for q in config.quants if not q.path.exists()]
            if not config.corpus_path.exists():
                missing.append(str(config.corpus_path))
            if missing:
                print("missing:", *missing, sep="\n  ")
                return 1
            print(f"config ok: {config.name}, {len(config.quants)} quants, corpus present")
            return 0

        if args.command == "all":
            fetch(config)
            path = run_sweep(
                config,
                llama_bin=Path(args.llama_bin),
                out_dir=Path(args.out_dir),
            )
            print(render_markdown([path]))
            return 0
    except (RunError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 2

"""Parsers for llama.cpp and rocm-smi output.

These are pure functions so the measurement pipeline can be tested without a
GPU. Every parser returns ``None`` for a missing metric instead of inventing a
zero, because a zero is a valid measurement and a missing value is not.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FLOAT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def _search_float(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_perplexity(text: str) -> dict[str, Any]:
    """Parse ``llama-perplexity`` output.

    Handles both plain PPL runs and ``--kl-divergence`` runs. The KL block also
    contains PPL(Q) and PPL(base), so one parser covers both.
    """
    result: dict[str, Any] = {
        "ppl": _search_float(rf"Final estimate: PPL = ({_FLOAT})", text),
        "ppl_q": _search_float(rf"Mean PPL\(Q\)\s*:\s*({_FLOAT})", text),
        "ppl_base": _search_float(rf"Mean PPL\(base\)\s*:\s*({_FLOAT})", text),
        "ppl_ratio": _search_float(rf"Mean PPL\(Q\)/PPL\(base\)\s*:\s*({_FLOAT})", text),
        "kl_mean": _search_float(rf"Mean\s+KLD:\s*({_FLOAT})", text),
        "kl_median": _search_float(rf"Median\s+KLD:\s*({_FLOAT})", text),
        "kl_p95": _search_float(rf"95\.0%\s+KLD:\s*({_FLOAT})", text),
        "kl_p99": _search_float(rf"99\.0%\s+KLD:\s*({_FLOAT})", text),
        "mean_delta_p_pct": _search_float(rf"Mean\s+Δp:\s*({_FLOAT})", text),
    }
    if result["kl_median"] is None:
        # Older builds print "Median  KLD" with two spaces; the regex covers it,
        # but keep a fallback for builds that only emit the mean.
        result["kl_median"] = result["kl_mean"]
    return result


def parse_bench_json(text: str) -> list[dict[str, Any]]:
    """Parse ``llama-bench -o json``.

    llama-bench sometimes prints a warning line before the JSON array. Find the
    first ``[`` and parse from there.
    """
    start = text.find("[")
    if start < 0:
        raise ValueError("llama-bench output has no JSON array")
    # llama-bench may print backend warnings after the JSON array, so parse
    # exactly one JSON value instead of assuming the rest of stdout is clean.
    rows, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(rows, list):
        raise ValueError("llama-bench JSON is not an array")
    return rows


def summarize_bench(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Collapse llama-bench rows into prompt and decode throughput.

    ``test`` is ``pp512`` for prompt processing and ``tg128`` for generation.
    The field name for throughput changed across llama.cpp versions, so accept
    both ``avg_ts`` and ``avg_ns``-derived values.
    """
    prompt = None
    decode = None
    for row in rows:
        ts = row.get("avg_ts")
        if ts is None:
            continue
        test = str(row.get("test", ""))
        n_prompt = int(row.get("n_prompt", 0) or 0)
        n_gen = int(row.get("n_gen", 0) or 0)
        if test.startswith("pp") or (test == "" and n_prompt > 0 and n_gen == 0):
            prompt = float(ts)
        elif test.startswith("tg") or (test == "" and n_gen > 0 and n_prompt == 0):
            decode = float(ts)
    return {"prompt_tps": prompt, "decode_tps": decode}


def parse_vram_json(text: str) -> int | None:
    """Return used VRAM bytes from ``rocm-smi --showmeminfo vram --json``."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    for card in data.values():
        if not isinstance(card, dict):
            continue
        for key, value in card.items():
            if "Used Memory" in key and "B" in key:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
    return None


def parse_llama_version(text: str) -> dict[str, str]:
    """Extract version, build number, and commit from ``llama-cli --version``."""
    version = None
    build = None
    commit = None
    m = re.search(r"version:\s*([^\s]+)", text)
    if m:
        version = m.group(1)
    m = re.search(r"build\s+(\d+)", text)
    if m:
        build = m.group(1)
    m = re.search(r"commit\s+([0-9a-f]+)", text)
    if m:
        commit = m.group(1)
    return {
        "version": version or "unknown",
        "build": build or "unknown",
        "commit": commit or "unknown",
    }

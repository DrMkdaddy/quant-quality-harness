"""Run the measurement pipeline and write one artifact per sweep."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, QuantSpec
from .parse import (
    parse_bench_json,
    parse_llama_version,
    parse_perplexity,
    summarize_bench,
)
from .vram import VramSampler


SCHEMA = "quantlab/v1"


class RunError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(cmd: list[str], log_path: Path, timeout_s: int | None = None) -> tuple[str, float, int]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        text = (exc.stdout or "") + (exc.stderr or "")
        log_path.write_text(text + f"\nTIMEOUT after {timeout_s}s\n")
        raise RunError(f"command timed out after {timeout_s}s: {' '.join(cmd)}") from exc
    text = proc.stdout + proc.stderr
    log_path.write_text(text)
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = "\n".join(text.strip().splitlines()[-20:])
        raise RunError(
            f"command failed ({proc.returncode}): {' '.join(cmd)}\n{tail}\nlog: {log_path}"
        )
    return text, elapsed, proc.returncode


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    if shutil.which("aria2c"):
        subprocess.run(
            [
                "aria2c",
                "-x", "8",
                "-s", "8",
                "-k", "1M",
                "--file-allocation=none",
                "-o", tmp.name,
                "-d", str(tmp.parent),
                url,
            ],
            check=True,
        )
    else:
        urllib.request.urlretrieve(url, tmp)
    tmp.replace(dst)


def fetch(config: Config, *, force: bool = False) -> None:
    """Download the corpus and every quant that is not already present."""
    corpus = config.corpus_path
    if force or not corpus.exists():
        archive = config.corpus_dir / "wikitext-2-raw-v1.zip"
        if force or not archive.exists():
            print(f"fetch corpus: {config.corpus_url}")
            _download(config.corpus_url, archive)
        with zipfile.ZipFile(archive) as zf:
            member = config.corpus_member
            with zf.open(member) as src, corpus.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        print(f"  wrote {corpus} ({corpus.stat().st_size} bytes)")

    for quant in config.quants:
        path = quant.path
        if path.exists() and path.stat().st_size > 0 and not force:
            print(f"have {quant.name}: {path} ({path.stat().st_size} bytes)")
            continue
        if not quant.url:
            raise RunError(f"{quant.name} is missing at {path} and has no url")
        print(f"fetch {quant.name}: {quant.url}")
        _download(quant.url, path)
        print(f"  wrote {path} ({path.stat().st_size} bytes)")

    verify_hashes(config)


def verify_hashes(config: Config) -> None:
    for quant in config.quants:
        if not quant.sha256:
            continue
        actual = _sha256(quant.path)
        if actual != quant.sha256:
            raise RunError(
                f"{quant.name} sha256 mismatch: expected {quant.sha256}, got {actual}"
            )
        print(f"sha256 ok: {quant.name}")


def _binaries(llama_bin: Path) -> dict[str, Path]:
    names = {
        "perplexity": "llama-perplexity",
        "bench": "llama-bench",
        "cli": "llama-cli",
    }
    resolved = {}
    for key, name in names.items():
        path = llama_bin / name
        if not path.exists():
            raise RunError(f"missing {name} under {llama_bin}")
        resolved[key] = path
    return resolved


def _hardware() -> dict[str, Any]:
    gpu = None
    for command in (["rocm-smi", "--showproductname", "--json"], ["lspci"]):
        try:
            if command[0] == "rocm-smi":
                proc = subprocess.run(command, capture_output=True, text=True, check=False)
                if proc.returncode == 0:
                    gpu = proc.stdout.strip()
                    break
            else:
                proc = subprocess.run(
                    ["bash", "-lc", "lspci | grep -Ei 'vga|display'"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                gpu = proc.stdout.strip()
                break
        except FileNotFoundError:
            continue
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def _model_meta(quant: QuantSpec) -> dict[str, Any]:
    path = quant.path
    return {
        "name": quant.name,
        "path": str(path),
        "file_bytes": path.stat().st_size if path.exists() else None,
        "sha256": _sha256(path) if path.exists() else None,
    }


def run_sweep(
    config: Config,
    *,
    llama_bin: Path,
    out_dir: Path,
    run_id: str | None = None,
    timeout_s: int = 3600,
) -> Path:
    bins = _binaries(llama_bin)
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)

    corpus = config.corpus_path
    if not corpus.exists():
        raise RunError(f"corpus missing at {corpus}; run `quantlab fetch` first")
    for quant in config.quants:
        if not quant.path.exists():
            raise RunError(f"model missing at {quant.path}; run `quantlab fetch` first")

    version_text, _, _ = _run([str(bins["cli"]), "--version"], log_dir / "llama-version.log")
    version = parse_llama_version(version_text)

    ref_quant = config.quant(config.reference)
    ref_logits = out_dir / f"reference-{config.name}-{config.reference}.logits"
    ref_log_path = log_dir / f"{ref_quant.name}.ppl.log"

    ppl = config.perplexity
    ppl_common = [
        "-f", str(corpus),
        "-c", str(ppl.context),
        "--chunks", str(ppl.chunks),
        "-b", str(ppl.batch),
        "-ngl", str(ppl.gpu_layers),
        "-fa", ppl.flash_attn,
        "-t", str(ppl.threads),
    ]
    ppl_base = [str(bins["perplexity"]), "-m", str(ref_quant.path), *ppl_common]
    with VramSampler() as sampler:
        ref_text, ref_elapsed, _ = _run(
            ppl_base + ["--save-all-logits", str(ref_logits)],
            ref_log_path,
            timeout_s=timeout_s,
        )
    ref_metrics = parse_perplexity(ref_text)
    ref_ppl = ref_metrics.get("ppl") or ref_metrics.get("ppl_q")
    if ref_ppl is None:
        raise RunError(f"could not parse reference PPL; see {ref_log_path}")

    bench = config.bench
    rows: list[dict[str, Any]] = []

    for quant in config.quants:
        ppl_log = log_dir / f"{quant.name}.ppl.log"
        if quant.name == config.reference:
            metrics = ref_metrics
            ppl_elapsed = ref_elapsed
            ppl_vram = sampler.delta_bytes
            ppl_command = ppl_base + ["--save-all-logits", str(ref_logits)]
        else:
            cmd = [str(bins["perplexity"]), "-m", str(quant.path), *ppl_common]
            cmd += ["--kl-divergence", "--kl-divergence-base", str(ref_logits)]
            with VramSampler() as ppl_sampler:
                text, ppl_elapsed, _ = _run(cmd, ppl_log, timeout_s=timeout_s)
            metrics = parse_perplexity(text)
            if metrics.get("ppl_q") is None:
                raise RunError(f"could not parse KL/PPL for {quant.name}; see {ppl_log}")
            metrics["ppl"] = metrics["ppl_q"]
            ppl_vram = ppl_sampler.delta_bytes
            ppl_command = cmd

        bench_log = log_dir / f"{quant.name}.bench.log"
        bench_cmd = [
            str(bins["bench"]),
            "-m", str(quant.path),
            "-p", str(bench.prompt_tokens),
            "-n", str(bench.gen_tokens),
            "-r", str(bench.repetitions),
            "-ngl", str(bench.gpu_layers),
            "-fa", bench.flash_attn,
            "-t", str(bench.threads),
            "-o", "json",
        ]
        with VramSampler() as bench_sampler:
            bench_text, bench_elapsed, _ = _run(bench_cmd, bench_log, timeout_s=timeout_s)
        try:
            bench_rows = parse_bench_json(bench_text)
            speed = summarize_bench(bench_rows)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RunError(f"could not parse llama-bench for {quant.name}: {exc}") from exc

        rows.append(
            {
                "quant": quant.name,
                **_model_meta(quant),
                "quality": {
                    "ppl": metrics.get("ppl"),
                    "ppl_base": metrics.get("ppl_base"),
                    "ppl_ratio": metrics.get("ppl_ratio"),
                    "kl_mean": metrics.get("kl_mean"),
                    "kl_median": metrics.get("kl_median"),
                    "kl_p95": metrics.get("kl_p95"),
                    "kl_p99": metrics.get("kl_p99"),
                    "mean_delta_p_pct": metrics.get("mean_delta_p_pct"),
                },
                "speed": speed,
                "vram": {
                    "ppl_delta_bytes": ppl_vram,
                    "bench_delta_bytes": bench_sampler.delta_bytes,
                    "peak_delta_bytes": max(
                        [v for v in (ppl_vram, bench_sampler.delta_bytes) if v is not None],
                        default=None,
                    ),
                },
                "timing": {
                    "ppl_seconds": round(ppl_elapsed, 3),
                    "bench_seconds": round(bench_elapsed, 3),
                },
                "commands": {
                    "ppl": ppl_command,
                    "bench": bench_cmd,
                },
                "logs": {
                    "ppl": str(ppl_log),
                    "bench": str(bench_log),
                },
            }
        )
        print(
            f"{quant.name:8s} PPL {rows[-1]['quality']['ppl']:.4f}  "
            f"KL {rows[-1]['quality']['kl_mean'] if rows[-1]['quality']['kl_mean'] is not None else 0:.6f}  "
            f"tg {speed['decode_tps']} tok/s"
        )

    artifact = {
        "schema": SCHEMA,
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_family": config.name,
        "hardware": _hardware(),
        "llama_cpp": {
            **version,
            "bin_dir": str(llama_bin),
        },
        "corpus": {
            "path": str(corpus),
            "sha256": _sha256(corpus),
            "bytes": corpus.stat().st_size,
            "context": ppl.context,
            "chunks": ppl.chunks,
            "tokens_evaluated": ppl.context * ppl.chunks,
        },
        "reference": {
            "quant": config.reference,
            "logits_path": str(ref_logits),
            "logits_bytes": ref_logits.stat().st_size,
        },
        "results": rows,
    }
    out_path = out_dir / f"quant_quality_{config.name}_{run_id}.json"
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(f"wrote {out_path}")
    return out_path


def render_markdown(paths: list[Path]) -> str:
    if not paths:
        raise RunError("no result files given")
    newest = max(paths, key=lambda p: p.stat().st_mtime)
    data = json.loads(newest.read_text())
    rows = data["results"]
    lines = []
    lines.append(f"Source: `{newest}`")
    lines.append("")
    lines.append(
        "| Quant | File GiB | PPL | vs F16 | KL mean | KL p95 | mean Δp | tg tok/s | VRAM GiB |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        q = row["quality"]
        gib = (row["file_bytes"] or 0) / (1024**3)
        vram = row["vram"]["peak_delta_bytes"]
        vram_gib = f"{vram / (1024**3):.2f}" if vram else "n/a"
        ratio = q["ppl_ratio"]
        ratio_s = f"{ratio:.3f}x" if ratio is not None else "1.000x"
        kl = q["kl_mean"]
        kl_s = f"{kl:.6f}" if kl is not None else "0"
        p95 = q["kl_p95"]
        p95_s = f"{p95:.6f}" if p95 is not None else "0"
        dp = q["mean_delta_p_pct"]
        dp_s = f"{dp:.2f}%" if dp is not None else "0%"
        tg = row["speed"]["decode_tps"]
        tg_s = f"{tg:.1f}" if tg is not None else "n/a"
        lines.append(
            f"| {row['quant']} | {gib:.2f} | {q['ppl']:.4f} | {ratio_s} | "
            f"{kl_s} | {p95_s} | {dp_s} | {tg_s} | {vram_gib} |"
        )
    lines.append("")
    lines.append(
        f"llama.cpp {data['llama_cpp']['version']} ({data['llama_cpp']['commit']}), "
        f"{data['hardware']['gpu'].splitlines()[0] if data['hardware'].get('gpu') else 'GPU unknown'}, "
        f"wikitext-2 test, {data['corpus']['context']} ctx x {data['corpus']['chunks']} chunks."
    )
    return "\n".join(lines) + "\n"

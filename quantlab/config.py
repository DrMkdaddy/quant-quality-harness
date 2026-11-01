"""Configuration loading and path resolution.

The config is deliberately data-only. It names a model family, a reference
quant, and the quants to compare. Paths are resolved at run time from
``QUANTLAB_MODEL_DIR`` so the same manifest works on another machine.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_MODEL_DIR = Path.home() / "models" / "quant-lab"


class ConfigError(ValueError):
    """Raised when a manifest is missing required fields."""


@dataclass(frozen=True)
class QuantSpec:
    name: str
    filename: str
    url: str = ""
    sha256: str = ""

    @property
    def path(self) -> Path:
        return self._model_dir / self.filename

    def resolve(self, model_dir: Path) -> "QuantSpec":
        return QuantSpec(
            name=self.name,
            filename=self.filename,
            url=self.url,
            sha256=self.sha256,
            _model_dir=model_dir,
        )

    _model_dir: Path = field(default=DEFAULT_MODEL_DIR, repr=False, compare=False)


@dataclass(frozen=True)
class PerplexitySpec:
    context: int = 512
    chunks: int = 16
    batch: int = 512
    gpu_layers: int = 99
    flash_attn: str = "on"
    threads: int = 8


@dataclass(frozen=True)
class BenchSpec:
    prompt_tokens: int = 512
    gen_tokens: int = 128
    repetitions: int = 3
    gpu_layers: int = 99
    flash_attn: str = "on"
    threads: int = 8


@dataclass(frozen=True)
class Config:
    name: str
    corpus_url: str
    corpus_member: str
    reference: str
    quants: tuple[QuantSpec, ...]
    perplexity: PerplexitySpec
    bench: BenchSpec
    model_dir: Path
    corpus_dir: Path

    def quant(self, name: str) -> QuantSpec:
        for q in self.quants:
            if q.name == name:
                return q
        raise ConfigError(f"unknown quant {name!r}; choose from {[q.name for q in self.quants]}")

    @property
    def corpus_path(self) -> Path:
        return self.corpus_dir / Path(self.corpus_member).name


def _require(obj: dict[str, Any], key: str, kind: type | tuple[type, ...]) -> Any:
    if key not in obj:
        raise ConfigError(f"missing required key {key!r}")
    value = obj[key]
    if not isinstance(value, kind):
        raise ConfigError(f"{key!r} must be {kind}, got {type(value).__name__}")
    return value


def load_config(path: str | Path, model_dir: str | Path | None = None) -> Config:
    manifest = Path(path)
    data = json.loads(manifest.read_text())

    model_root = Path(
        model_dir or os.environ.get("QUANTLAB_MODEL_DIR") or DEFAULT_MODEL_DIR
    ).expanduser()

    qdata = _require(data, "quants", list)
    if not qdata:
        raise ConfigError("quants must not be empty")
    quants = tuple(
        QuantSpec(
            name=_require(q, "name", str),
            filename=_require(q, "filename", str),
            url=q.get("url", ""),
            sha256=q.get("sha256", ""),
        ).resolve(model_root)
        for q in qdata
    )
    names = [q.name for q in quants]
    if len(names) != len(set(names)):
        raise ConfigError(f"duplicate quant names: {names}")

    reference = _require(data, "reference", str)
    if reference not in names:
        raise ConfigError(f"reference {reference!r} is not one of {names}")

    ppl = data.get("perplexity", {})
    bench = data.get("bench", {})
    cfg = Config(
        name=_require(data, "name", str),
        corpus_url=_require(data, "corpus_url", str),
        corpus_member=_require(data, "corpus_member", str),
        reference=reference,
        quants=quants,
        perplexity=PerplexitySpec(
            context=int(ppl.get("context", 512)),
            chunks=int(ppl.get("chunks", 16)),
            batch=int(ppl.get("batch", 512)),
            gpu_layers=int(ppl.get("gpu_layers", 99)),
            flash_attn=str(ppl.get("flash_attn", "on")),
            threads=int(ppl.get("threads", 8)),
        ),
        bench=BenchSpec(
            prompt_tokens=int(bench.get("prompt_tokens", 512)),
            gen_tokens=int(bench.get("gen_tokens", 128)),
            repetitions=int(bench.get("repetitions", 3)),
            gpu_layers=int(bench.get("gpu_layers", 99)),
            flash_attn=str(bench.get("flash_attn", "on")),
            threads=int(bench.get("threads", 8)),
        ),
        model_dir=model_root,
        corpus_dir=Path(
            os.environ.get("QUANTLAB_CORPUS_DIR", model_root)
        ).expanduser(),
    )
    if cfg.perplexity.chunks * cfg.perplexity.context < 2 * cfg.perplexity.context:
        raise ConfigError("chunks * context must be at least 2 * context")
    return cfg

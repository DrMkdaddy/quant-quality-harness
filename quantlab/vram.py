"""Sample AMD VRAM usage while a benchmark runs."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time

from .parse import parse_vram_json


class VramSampler:
    """Poll ``rocm-smi`` in a background thread and keep the maximum.

    A 100 ms interval is frequent enough for a multi-second benchmark and
    cheap enough not to perturb it. On a machine without rocm-smi the sampler
    degrades to returning ``None`` rather than failing the run.
    """

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.peak_bytes: int | None = None
        self.baseline_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def available(self) -> bool:
        return shutil.which("rocm-smi") is not None

    def _read(self) -> int | None:
        proc = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return parse_vram_json(proc.stdout)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            value = self._read()
            if value is not None and (self.peak_bytes is None or value > self.peak_bytes):
                self.peak_bytes = value

    def __enter__(self) -> "VramSampler":
        if not self.available:
            return self
        self.baseline_bytes = self._read()
        self.peak_bytes = self.baseline_bytes
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    @property
    def delta_bytes(self) -> int | None:
        if self.peak_bytes is None or self.baseline_bytes is None:
            return None
        return max(0, self.peak_bytes - self.baseline_bytes)

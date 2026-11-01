# quant-quality-harness

Measure what a GGUF quantization costs: perplexity, KL divergence against a
full-precision reference, single-stream speed, and peak VRAM, on the same
corpus and with the same llama.cpp build.

The point is not to rank quants in the abstract. It is to produce one JSON
artifact per run that a resume bullet, a model card, or a reviewer can check,
and a single command that regenerates it.

## What it measures

For every quant in a model family:

| Metric | Source | Why it matters |
|---|---|---|
| Perplexity | `llama-perplexity`, fixed context and chunk count | The standard quality scalar |
| KL divergence vs F16 | `llama-perplexity --kl-divergence` | Detects distribution shift that PPL hides |
| Token agreement | `Mean Δp` from the same run | How often the quant changes the next token |
| Prompt / decode throughput | `llama-bench -p 512 -n 128` | What the quality costs in speed |
| Peak VRAM | `rocm-smi` sampled during the run | Whether the quant fits the card |

The harness records the exact command, model file hash, corpus hash, GPU,
llama.cpp build commit, and raw log path for every row.

## Quick start

```bash
# 1. Fetch the corpus and the five quants (about 18 GB, cached)
python3 -m quantlab fetch --config configs/qwen2.5-3b.json

# 2. Run the whole sweep (F16 reference + five quants)
python3 -m quantlab run --config configs/qwen2.5-3b.json

# 3. Render the Markdown table from results/*.json
python3 -m quantlab report results/quant_quality_*.json
```

Or the whole thing:

```bash
./scripts/reproduce.sh
```

Override paths without editing the config:

```bash
export QUANTLAB_MODEL_DIR=$HOME/models/quant-lab
export QUANTLAB_LLAMA_BIN=$HOME/llama.cpp/src-b11050/build/bin
python3 -m quantlab run --config configs/qwen2.5-3b.json
```

## Results

The committed run is in [`results/`](results/). The table below is generated
from that artifact, not typed by hand.

<!-- RESULTS:START -->
Run `python3 -m quantlab report results/quant_quality_*.json` after a sweep.
<!-- RESULTS:END -->

## Method

- Model family: `Qwen2.5-3B-Instruct`, because five quants of one model fit on
  a 16 GB card and the same tokenizer keeps KL comparable.
- Reference: the F16 GGUF. KL is computed against F16, not against Q8_0, so
  Q8_0 gets a real number instead of a zero.
- Corpus: `wikitext-2-raw-v1` test split, the same file llama.cpp's own
  `get-wikitext-2.sh` uses. Truncated deterministically to the first
  `context * chunks` tokens by llama.cpp.
- PPL and KL: `-c 512 --chunks 16 -ngl 99 -fa on`. The KL run also reports
  PPL(Q), so quality and divergence come from the same forward passes.
- Speed: `llama-bench -p 512 -n 128 -r 3 -ngl 99 -fa on`, median of three.
- VRAM: `rocm-smi --showmeminfo vram --json` sampled every 100 ms around the
  benchmark process; the reported value is the maximum used memory minus the
  pre-run baseline.
- Quant files come from `bartowski/Qwen2.5-3B-Instruct-GGUF`, one recipe for
  all five, so the comparison is not confounded by different imatrix data.

## Caveats

- One GPU, one model family. These numbers do not transfer to a different
  model without rerunning.
- PPL and KL are corpus-dependent. The corpus hash is in the artifact; a
  different corpus is a different experiment.
- The speed test is single-stream. Quantizations can behave differently at
  batch > 1; the serving scheduler project measures that separately.
- VRAM sampling is 100 ms, so a short spike between samples can be missed.
  The useful signal is the sustained footprint, not the absolute peak.

## Layout

```
quantlab/
  cli.py          fetch / run / report / check
  config.py       config schema and path resolution
  runner.py       subprocess orchestration and artifact writing
  parse.py        llama.cpp output parsers (unit-tested)
  vram.py         rocm-smi sampler
configs/          model-family manifests
scripts/          reproduce.sh
tests/            parser and config tests, no GPU required
results/          committed JSON artifact, logs, and rendered table
```

## License

MIT.

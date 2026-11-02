# quant-quality-harness

[![test](https://github.com/DrMkdaddy/quant-quality-harness/actions/workflows/test.yml/badge.svg)](https://github.com/DrMkdaddy/quant-quality-harness/actions/workflows/test.yml)

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
Source: `results/quant_quality_qwen2.5-3b-instruct_20261101T234442Z.json`

| Quant | File GiB | PPL | vs F16 | KL mean | KL p95 | mean Δp | tg tok/s | VRAM GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F16 | 5.75 | 9.1318 | 1.000x | 0 | 0 | 0% | 84.1 | 6.11 |
| Q8_0 | 3.06 | 9.1607 | 1.003x | 0.001973 | 0.005678 | -0.03% | 138.2 | 3.41 |
| Q6_K | 2.36 | 9.2858 | 1.017x | 0.006416 | 0.017362 | -0.15% | 173.9 | 2.73 |
| Q5_K_M | 2.07 | 9.3840 | 1.028x | 0.011180 | 0.031713 | -0.28% | 182.9 | 2.48 |
| Q4_K_M | 1.80 | 9.5187 | 1.042x | 0.028534 | 0.088485 | -0.48% | 200.6 | 2.17 |
| IQ4_XS | 1.62 | 9.5892 | 1.050x | 0.043635 | 0.133755 | -0.46% | 206.2 | 1.99 |

llama.cpp 0.4.1-dev (60b06ab9a), AMD Radeon RX 9070 XT, wikitext-2 test,
512 context x 8 chunks.

What the table says:

- Q4_K_M is the serving point: +4.2% PPL over F16, 2.4x the
  single-stream decode rate, and 2.17 GiB of VRAM instead of 6.11 GiB.
- Q8_0 is nearly free in quality (+0.3% PPL, KL 0.0020) but leaves
  half the speed on the table compared with Q4_K_M.
- IQ4_XS is the fastest file (206 tok/s) but its KL is 15x Q4_K_M's
  and its PPL is +5.0% over F16. The extra 5.6 tok/s over Q4_K_M is
  not worth doubling the distribution shift unless memory is the
  binding constraint.
- Q5_K_M is the conservative middle: +2.8% PPL, 183 tok/s, KL 0.011.

The reference row uses the KL path's `PPL(base)` (9.1318), not the
standalone F16 PPL (9.1813), because the two are computed over different
token subsets. The standalone number is kept in the JSON as
`ppl_standalone`.
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
- The sweep uses the local `llama-b11050` build. The newer local build's
  `llama-perplexity` segfaulted on the same model and corpus before producing
  a token, so the binary is pinned and its commit is recorded in the artifact.
  This is a harness provenance detail, not a claim about upstream.

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

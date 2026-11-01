import unittest

from quantlab.parse import (
    parse_bench_json,
    parse_llama_version,
    parse_perplexity,
    parse_vram_json,
    summarize_bench,
)


PERPLEXITY = """\
[1]12.3456,[2]11.1111,
Final estimate: PPL = 12.3456 +/- 0.12345
"""

KL = """\
====== Perplexity statistics ======
Mean PPL(Q)                   :  12.500000 ±  0.100000
Mean PPL(base)                :  12.000000 ±  0.090000
Mean ln(PPL(Q)/PPL(base))     :   0.040822 ±  0.001000
Mean PPL(Q)/PPL(base)         :   1.041667 ±  0.001000
Mean PPL(Q)-PPL(base)         :   0.500000 ±  0.001000

====== KL divergence statistics ======
Mean    KLD:   0.012345 ±  0.000100
Maximum KLD:   2.000000
99.9%   KLD:   1.000000
99.0%   KLD:   0.500000
95.0%   KLD:   0.100000
90.0%   KLD:   0.050000
Median  KLD:   0.010000
10.0%   KLD:   0.001000

====== Token probability statistics ======
Mean    Δp:  1.234 ± 0.100 %
"""

BENCH = """\
ggml_vulkan: warning
[
  {"test":"pp512","avg_ts":1234.5},
  {"test":"tg128","avg_ts":98.7}
]
"""

VRAM = '{"card0": {"VRAM Total Memory (B)": "17095983104", "VRAM Total Used Memory (B)": "1317588992"}}'


class ParsePerplexityTests(unittest.TestCase):
    def test_plain_ppl(self):
        parsed = parse_perplexity(PERPLEXITY)
        self.assertAlmostEqual(parsed["ppl"], 12.3456)
        self.assertIsNone(parsed["kl_mean"])

    def test_kl_run(self):
        parsed = parse_perplexity(KL)
        self.assertAlmostEqual(parsed["ppl_q"], 12.5)
        self.assertAlmostEqual(parsed["ppl_base"], 12.0)
        self.assertAlmostEqual(parsed["kl_mean"], 0.012345)
        self.assertAlmostEqual(parsed["kl_median"], 0.01)
        self.assertAlmostEqual(parsed["kl_p95"], 0.1)
        self.assertAlmostEqual(parsed["mean_delta_p_pct"], 1.234)
        self.assertIsNone(parsed["ppl"])

    def test_missing_values_are_none(self):
        parsed = parse_perplexity("nothing here")
        self.assertTrue(all(value is None for value in parsed.values()))


class ParseBenchTests(unittest.TestCase):
    def test_json_prefix_warning(self):
        rows = parse_bench_json(BENCH)
        self.assertEqual(len(rows), 2)
        speed = summarize_bench(rows)
        self.assertAlmostEqual(speed["prompt_tps"], 1234.5)
        self.assertAlmostEqual(speed["decode_tps"], 98.7)

    def test_no_json(self):
        with self.assertRaises(ValueError):
            parse_bench_json("no array")


class ParseVramTests(unittest.TestCase):
    def test_used_bytes(self):
        self.assertEqual(parse_vram_json(VRAM), 1317588992)

    def test_bad_json(self):
        self.assertIsNone(parse_vram_json("not json"))


class ParseVersionTests(unittest.TestCase):
    def test_version(self):
        parsed = parse_llama_version(
            "version: 0.7.0-dev (build 786, commit fcfb89b)\nbuilt with GNU"
        )
        self.assertEqual(parsed["version"], "0.7.0-dev")
        self.assertEqual(parsed["build"], "786")
        self.assertEqual(parsed["commit"], "fcfb89b")


if __name__ == "__main__":
    unittest.main()

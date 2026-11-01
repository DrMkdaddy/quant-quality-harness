import json
import tempfile
import unittest
from pathlib import Path

from quantlab.config import ConfigError, load_config


MANIFEST = {
    "name": "test-family",
    "corpus_url": "https://example.invalid/corpus.zip",
    "corpus_member": "wikitext-2-raw/wiki.test.raw",
    "reference": "F16",
    "perplexity": {"context": 512, "chunks": 8},
    "bench": {"prompt_tokens": 128, "gen_tokens": 32},
    "quants": [
        {"name": "F16", "filename": "f16.gguf"},
        {"name": "Q4_K_M", "filename": "q4.gguf"},
    ],
}


class ConfigTests(unittest.TestCase):
    def _write(self, data):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name) / "config.json"
        path.write_text(json.dumps(data))
        self.addCleanup(tmp.cleanup)
        return path

    def test_paths_resolve_under_model_dir(self):
        cfg = load_config(self._write(MANIFEST), model_dir="/models")
        self.assertEqual(cfg.quant("Q4_K_M").path, Path("/models/q4.gguf"))
        self.assertEqual(cfg.corpus_path, Path("/models/wiki.test.raw"))

    def test_reference_must_exist(self):
        data = dict(MANIFEST, reference="Q8_0")
        with self.assertRaises(ConfigError):
            load_config(self._write(data), model_dir="/models")

    def test_duplicate_names_rejected(self):
        data = dict(MANIFEST)
        data["quants"] = [
            {"name": "F16", "filename": "a.gguf"},
            {"name": "F16", "filename": "b.gguf"},
        ]
        with self.assertRaises(ConfigError):
            load_config(self._write(data), model_dir="/models")

    def test_short_corpus_rejected(self):
        data = dict(MANIFEST, perplexity={"context": 512, "chunks": 1})
        with self.assertRaises(ConfigError):
            load_config(self._write(data), model_dir="/models")


if __name__ == "__main__":
    unittest.main()

"""Unit tests for benchmark client dataset sampling."""

import json
import tempfile
import unittest

from llm_optimizer.bench_client import sample_sharegpt_requests


class _NoBosTokenizer:
    """Minimal tokenizer stub for models that define no BOS token (e.g. Qwen)."""

    bos_token = None

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        return "<|user|>" + messages[0]["content"] + "<|assistant|>"

    def encode(self, text):
        return list(range(len(text.split())))


class TestSampleSharegptRequests(unittest.TestCase):
    def test_chat_template_with_tokenizer_without_bos_token(self):
        rows = [
            {
                "conversations": [
                    {"from": "human", "value": f"write a function number {i}"},
                    {"from": "gpt", "value": "def f(): return 1 " * 4},
                ]
            }
            for i in range(3)
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(rows, f)
            f.flush()
            requests = sample_sharegpt_requests(
                dataset_path=f.name,
                num_requests=3,
                tokenizer=_NoBosTokenizer(),
                apply_chat_template=True,
            )

        self.assertEqual(len(requests), 3)
        for request in requests:
            self.assertTrue(request.prompt.startswith("<|user|>"))


if __name__ == "__main__":
    unittest.main()

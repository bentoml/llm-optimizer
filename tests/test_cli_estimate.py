"""Regression tests for estimate option precedence and interactive output."""

import unittest
from unittest.mock import patch

from click.testing import CliRunner

from llm_optimizer.cli import cli
from llm_optimizer.common import ModelConfig


class TestEstimateOptions(unittest.TestCase):
    def setUp(self):
        self.runner = CliRunner()
        self.interactive_params = {
            "model": "test/model",
            "input_len": 1024,
            "output_len": 512,
            "target": "latency",
            "constraints": None,
            "precision": "bf16",
            "framework": "sglang",
            "dataset": "sharegpt",
        }

    def invoke_estimate(self, args):
        with (
            patch(
                "llm_optimizer.cli.collect_interactive_parameters",
                return_value=self.interactive_params,
            ),
            patch(
                "llm_optimizer.cli.collect_gpu_configuration",
                return_value=("H100", 2),
            ),
            patch("llm_optimizer.cli.run_performance_estimation") as estimate,
            patch("llm_optimizer.cli.display_performance_estimation_results"),
        ):
            estimate.side_effect = lambda params: (params, None)
            result = self.runner.invoke(cli, ["estimate", *args])
        self.assertEqual(result.exit_code, 0, result.output)
        estimate.assert_called_once()
        return estimate.call_args.args[0]

    def test_interactive_choices_override_defaults(self):
        params = self.invoke_estimate(["--interactive"])
        self.assertEqual(params.framework, "sglang")
        self.assertEqual(params.target, "latency")
        self.assertEqual(params.dataset, "sharegpt")

    def test_explicit_default_values_override_interactive_choices(self):
        params = self.invoke_estimate(
            [
                "--interactive",
                "--framework",
                "both",
                "--target",
                "throughput",
                "--dataset",
                "random",
            ]
        )
        self.assertEqual(params.framework, "both")
        self.assertEqual(params.target, "throughput")
        self.assertEqual(params.dataset, "random")

    def test_explicit_nondefault_values_override_interactive_choices(self):
        params = self.invoke_estimate(["--interactive", "--framework", "vllm"])
        self.assertEqual(params.framework, "vllm")
        self.assertEqual(params.target, "latency")
        self.assertEqual(params.dataset, "sharegpt")

    def test_noninteractive_defaults(self):
        params = self.invoke_estimate(
            ["--model", "test/model", "--input-len", "1024", "--output-len", "512"]
        )
        self.assertEqual(params.framework, "both")
        self.assertEqual(params.target, "throughput")
        self.assertEqual(params.dataset, "random")

    def test_noninteractive_explicit_options(self):
        params = self.invoke_estimate(
            [
                "--model",
                "test/model",
                "--input-len",
                "1024",
                "--output-len",
                "512",
                "--framework",
                "vllm",
                "--target",
                "latency",
                "--dataset",
                "sharegpt",
            ]
        )
        self.assertEqual(params.framework, "vllm")
        self.assertEqual(params.target, "latency")
        self.assertEqual(params.dataset, "sharegpt")

    def test_interactive_choices_reach_generated_commands(self):
        model_config = ModelConfig(
            num_params=600_000_000,
            num_layers=28,
            hidden_dim=1024,
            num_heads=16,
            num_kv_heads=8,
        )
        for framework in ["sglang", "vllm", "both"]:
            with (
                self.subTest(framework=framework),
                patch("llm_optimizer.cli_utils.HAS_READLINE", False),
                patch(
                    "llm_optimizer.common.get_model_config_and_precision_from_hf",
                    return_value=model_config,
                ),
                patch(
                    "llm_optimizer.performance.get_model_config_and_precision_from_hf",
                    return_value=model_config,
                ),
            ):
                result = self.runner.invoke(
                    cli,
                    ["estimate", "--interactive", "--gpu", "H100", "--num-gpus", "2"],
                    input=f"test/model\n1024\n512\nlatency\n\nfp16\n{framework}\nsharegpt\n\n\n",
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertNotIn("Error:", result.output)
                self.assertIn("Target: latency", result.output)
                self.assertIn("dataset_name=sharegpt", result.output)
                for candidate in ["sglang", "vllm"]:
                    header = f"--- {candidate.upper()} ---"
                    if framework in [candidate, "both"]:
                        self.assertIn(header, result.output)
                        self.assertIn(f"--framework {candidate}", result.output)
                    else:
                        self.assertNotIn(header, result.output)
                        self.assertNotIn(f"--framework {candidate}", result.output)

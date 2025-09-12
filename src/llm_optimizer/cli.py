import json
import pathlib
import time
import typing as t

import click

import llm_optimizer.args as lo_args
import llm_optimizer.predefined as predefined
from llm_optimizer.cli_utils import (
    collect_gpu_configuration,
    collect_interactive_parameters,
    detect_gpu_type,
    get_gpu_count,
    normalize_gpu_choice,
)
from llm_optimizer.logging import get_logger, setup_logging
from llm_optimizer.performance import (
    PerformanceEstimationParams,
    display_performance_estimation_results,
    run_performance_estimation,
)
from llm_optimizer.predefined.gpus import list_available_gpus_with_lowercase
from llm_optimizer.server_utils import (
    start_server,
    terminate_process_top_down,
)
from llm_optimizer.utils import InfinityToNullEncoder

setup_logging()
logger = get_logger("main")

PREDEFINED_FRAMEWORKS = list(predefined.SERVER_CONFIGS.keys())


def construct_benchmark_settings(combo: list[lo_args.BaseArg]) -> dict[str, t.Any]:
    client_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.CLIENT]
    server_args = [arg for arg in combo if arg.scope == lo_args.ArgScope.SERVER]
    client_kv_pairs = lo_args.get_all_kv_pairs(client_args)
    server_cmd_args = lo_args.get_all_cmd_args(server_args)
    server_kv_pairs = lo_args.get_all_kv_pairs(server_args)
    return {
        "client_args": dict(client_kv_pairs),
        "server_args": dict(server_kv_pairs),
        "server_cmd_args": server_cmd_args,
    }


def extract_token_lengths(client_params: dict) -> tuple[int, int]:
    """Extract input and output token lengths from client parameters."""
    input_len = client_params.get("random_input_len", 0)
    output_len = client_params.get("random_output_len", 0)

    # If using sharegpt dataset, these might be None
    if input_len == 0 and output_len == 0:
        # Check if using sharegpt
        if client_params.get("dataset_name") == "sharegpt":
            # sharegpt_output_len might be specified
            output_len = client_params.get("sharegpt_output_len", -1)
            input_len = -1  # Variable for sharegpt

    return input_len, output_len


def find_best_throughput_configs(all_results: list, constraints: list = None) -> dict:
    """Find configurations with best input/output throughput."""
    best_configs = {
        "best_input_throughput": None,
        "best_output_throughput": None,
        "best_input_throughput_constrained": None,
        "best_output_throughput_constrained": None,
    }

    # Helper to check if result satisfies constraints
    def satisfies_constraints(result, constraints):
        if not constraints:
            return True

        metrics = result.get("results", {})
        for constraint in constraints:
            # Handle SLOConstraint objects only (no legacy dict support)
            metric_name = constraint.metric
            stat_type = constraint.stat_type
            operator = constraint.operator
            threshold = constraint.value

            # Map constraint metric names to result metric names
            metric_mapping = {
                "ttft": f"{stat_type}_ttft_ms",
                "itl": f"{stat_type}_itl_ms",
                "e2e_latency": f"{stat_type}_e2e_latency_ms",
            }

            result_metric = metric_mapping.get(
                metric_name, f"{stat_type}_{metric_name}"
            )
            value = metrics.get(result_metric)

            if value is None:
                return False

            if operator == "<" and value >= threshold:
                return False
            elif operator == ">" and value <= threshold:
                return False

        return True

    # Find best configurations
    for result in all_results:
        if "results" not in result:
            continue

        metrics = result["results"]
        input_tp = metrics.get("input_throughput", 0)
        output_tp = metrics.get("output_throughput", 0)

        # Update unconstrained bests
        if (
            best_configs["best_input_throughput"] is None
            or input_tp > best_configs["best_input_throughput"]["throughput"]
        ):
            best_configs["best_input_throughput"] = {
                "throughput": input_tp,
                "config": result["config"],
                "cmd": result.get("cmd", ""),
            }

        if (
            best_configs["best_output_throughput"] is None
            or output_tp > best_configs["best_output_throughput"]["throughput"]
        ):
            best_configs["best_output_throughput"] = {
                "throughput": output_tp,
                "config": result["config"],
                "cmd": result.get("cmd", ""),
            }

        # Update constrained bests if constraints are satisfied
        if constraints and satisfies_constraints(result, constraints):
            if (
                best_configs["best_input_throughput_constrained"] is None
                or input_tp
                > best_configs["best_input_throughput_constrained"]["throughput"]
            ):
                best_configs["best_input_throughput_constrained"] = {
                    "throughput": input_tp,
                    "config": result["config"],
                    "cmd": result.get("cmd", ""),
                }

            if (
                best_configs["best_output_throughput_constrained"] is None
                or output_tp
                > best_configs["best_output_throughput_constrained"]["throughput"]
            ):
                best_configs["best_output_throughput_constrained"] = {
                    "throughput": output_tp,
                    "config": result["config"],
                    "cmd": result.get("cmd", ""),
                }

    return best_configs


def get_config_id(client_params: dict, server_params: dict) -> str:
    """Create a descriptive ID for a configuration."""
    client_param_strs = [f"{k}-{v}" for k, v in sorted(client_params.items())]
    server_param_strs = [f"{k}-{v}" for k, v in sorted(server_params.items())]

    config_id_parts = []
    if client_param_strs:
        config_id_parts.append("client_" + "-".join(client_param_strs))
    if server_param_strs:
        config_id_parts.append("server_" + "-".join(server_param_strs))

    return "_".join(config_id_parts) or "default"


@click.group(invoke_without_command=True)
@click.option("--server-cmd", type=str, help="The command to start the server.")
@click.option("--model", type=str, help="The model to use.")
@click.option(
    "--framework",
    type=click.Choice(PREDEFINED_FRAMEWORKS),
    help="The framework to use.",
)
@click.option(
    "--server-args", type=str, help="Arguments for the server.", multiple=True
)
@click.option(
    "--client-args", type=str, help="Arguments for the client.", multiple=True
)
@click.option("--gpus", type=int, help="The number of GPUs to use.")
@click.option("--dry-run", is_flag=True, help="A dry run will not run the command.")
@click.option(
    "--output-dir", default="results", help="Directory to store output files."
)
@click.option(
    "--output-json",
    type=str,
    default=None,
    help="Path to output a single JSON file with all results.",
)
@click.option(
    "--continue",
    "-c",
    "continue_flag",
    is_flag=True,
    help="Skip configs that already have output files.",
)
@click.option(
    "--rest", type=int, default=10, help="Rest time in seconds between benchmark runs."
)
@click.option("--mute-server", is_flag=True, help="Suppress server process stdout.")
@click.option(
    "--ready-endpoint",
    default="/health",
    help="Endpoint to check if server is ready (e.g., /health, /readyz).",
)
@click.option(
    "--host", type=str, default="127.0.0.1", help="Server host to connect to."
)
@click.option("--port", type=int, default=None, help="Server port to connect to.")
@click.option(
    "--constraints", type=str, help="SLO constraints (e.g., 'ttft<300ms;itl<8.5ms')"
)
@click.pass_context
def cli(
    ctx,
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    constraints,
):
    """A CLI tool to optimize LLM performance."""
    if ctx.invoked_subcommand is None:
        # If no subcommand is provided, run the main benchmark command
        benchmark(
            server_cmd,
            model,
            framework,
            server_args,
            client_args,
            gpus,
            dry_run,
            output_dir,
            output_json,
            continue_flag,
            rest,
            mute_server,
            ready_endpoint,
            host,
            port,
            constraints,
        )


def benchmark(
    server_cmd,
    model,
    framework,
    server_args,
    client_args,
    gpus,
    dry_run,
    output_dir,
    output_json,
    continue_flag,
    rest,
    mute_server,
    ready_endpoint,
    host,
    port,
    constraints,
):
    """A CLI tool to optimize LLM performance."""
    import llm_optimizer.bench_client as bench_client
    from llm_optimizer.performance import parse_slo_constraints
    from llm_optimizer.visualization.visualize import (
        convert_constraints_for_visualization,
    )

    # Parse constraints if provided
    parsed_constraints = []
    constraints_for_viz = []
    if constraints:
        try:
            parsed_constraints = parse_slo_constraints(constraints)
            constraints_for_viz = convert_constraints_for_visualization(
                parsed_constraints
            )
            logger.info(
                f"Parsed {len(parsed_constraints)} constraint(s): {constraints}"
            )
        except ValueError as e:
            logger.error(f"Error parsing constraints: {e}")
            # Continue without constraints rather than failing

    if not server_cmd:
        if not model or not framework:
            raise click.UsageError(
                "If --server-cmd is not provided, both --model and "
                "--framework are required."
            )

        if port is None:
            port = {
                "sglang": 30000,
                "vllm": 8000,
                "max": 8000,
            }.get(framework, 30000)

        tmpl = predefined.SEVER_CMD_TMPL[framework]
        server_cmd = tmpl.format(
            model=model,
            host=host,
            port=port,
        )

    if gpus is None:
        gpus = get_gpu_count()

    # Detect GPU type
    gpu_type = detect_gpu_type() or "unknown"
    logger.info(f"Detected GPU type: {gpu_type}, Count: {gpus}")

    if dry_run:
        click.echo("Dry run mode enabled.")

    server_args = ";".join(server_args)
    client_args = ";".join(client_args)

    server_configs = None
    if framework:
        server_configs = predefined.SERVER_CONFIGS[framework]

    server_args_sets = lo_args.parse_args_str(
        server_args,
        scope=lo_args.ArgScope.SERVER,
        configs=server_configs,
        strict=False,
    )

    client_args_sets = lo_args.parse_args_str(
        client_args,
        scope=lo_args.ArgScope.CLIENT,
        configs=predefined.CLIENT_CONFIGS,
        strict=True,
    )

    all_combinations = lo_args.get_all_arg_combinations(
        client_args_sets=client_args_sets,
        server_args_sets=server_args_sets,
    )

    total_configs = len(all_combinations)
    logger.info(f"Generated {total_configs} configuration(s) to run.")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ready_url = f"http://{host}:{port}{ready_endpoint}"

    output_jsonl_path = None
    if output_json:
        output_jsonl_path = pathlib.Path(output_json).with_suffix(".jsonl")

        # Check if output files already exist (unless continue flag is set)
        if not continue_flag:
            if pathlib.Path(output_json).exists():
                logger.error(
                    f"Output file {output_json} already exists! "
                    "Use --continue to resume or choose a different output file."
                )
                return
            if output_jsonl_path.exists():
                logger.error(
                    f"JSONL file {output_jsonl_path} already exists! "
                    "Use --continue to resume or choose a different output file."
                )
                return

    completed_config_ids = set()
    if continue_flag and output_jsonl_path and output_jsonl_path.exists():
        logger.info(
            f"Found existing JSONL file, loading completed runs: {output_jsonl_path}"
        )

        # Validate that existing results are for the same model
        existing_model = None
        with open(output_jsonl_path) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    result = json.loads(line)
                    result_model = result.get("metadata", {}).get("model_tag")
                    current_model = model

                    if existing_model is None:
                        existing_model = result_model
                    elif result_model != existing_model:
                        logger.warning(
                            f"Inconsistent models in existing results: "
                            f"{existing_model} vs {result_model}"
                        )

                    if result_model != current_model:
                        logger.error(
                            f"Model mismatch! Existing results for '{existing_model}' "
                            f"but current model is '{current_model}'. "
                            "Cannot append to existing results."
                        )
                        return

                    client_params = result.get("config", {}).get("client_args", {})
                    server_params = result.get("config", {}).get("server_args", {})
                    config_id = get_config_id(client_params, server_params)
                    completed_config_ids.add(config_id)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not parse line {line_num} in {output_jsonl_path}: "
                        f"{line.strip()}"
                    )
        logger.info(
            f"Loaded {len(completed_config_ids)} completed runs for model "
            f"{existing_model}"
        )

    for idx, combo in enumerate(all_combinations):
        benchmark_settings = construct_benchmark_settings(combo)

        client_params = benchmark_settings["client_args"]
        server_params = benchmark_settings["server_args"]
        server_cmd_args = benchmark_settings["server_cmd_args"]

        config_id = get_config_id(client_params, server_params)
        output_file_path = output_dir / f"{config_id}.json"

        logger.info("-" * 80)
        logger.info(f"Starting run {idx + 1}/{total_configs}: {config_id}")

        if continue_flag:
            if output_jsonl_path:
                if config_id in completed_config_ids:
                    logger.info(
                        f"Skipping as config_id '{config_id}' found in "
                        f"{output_jsonl_path}"
                    )
                    continue
            elif output_file_path.exists():
                logger.info(
                    f"Skipping as output file already exists: {output_file_path}"
                )
                continue

        if dry_run:
            print(benchmark_settings)
            continue

        # Build Server Command & Start Server
        server_process = None
        full_server_cmd = f"{server_cmd} {' '.join(server_cmd_args)}"

        try:
            server_process = start_server(full_server_cmd, {}, ready_url, mute_server)

            # Run Benchmark
            # Currently are testing OpenAI-compatible API, so pass "vllm" as backend to bench_client
            # We keep the possibility of use different backend for different framework here
            backend_for_bench = "vllm"
            benchmark_args = {
                "backend": backend_for_bench,
                "model": model,
                "host": host,
                "port": port,
                "dataset_name": "sharegpt",  # default value
                "num_prompts": 1000,  # default value
                "request_rate": float("inf"),  # default value
                "seed": 1,  # default value
            }
            benchmark_args.update(client_params)
            benchmark_result = bench_client.run_benchmark(benchmark_args)

            # Extract additional metadata
            model_tag = model
            input_len, output_len = extract_token_lengths(client_params)

            result_with_config = {
                "config": benchmark_settings,
                "results": benchmark_result,
                "cmd": full_server_cmd,
                "constraints": constraints_for_viz,
                "metadata": {
                    "gpu_type": gpu_type,
                    "gpu_count": gpus,
                    "model_tag": model_tag,
                    "input_tokens": input_len,
                    "output_tokens": output_len,
                },
            }

            if output_jsonl_path:
                with open(output_jsonl_path, "a") as f:
                    f.write(
                        json.dumps(result_with_config, cls=InfinityToNullEncoder) + "\n"
                    )
                logger.info(f"Appended result to {output_jsonl_path}")
            else:
                with open(output_file_path, "w") as f:
                    json.dump(
                        result_with_config, f, indent=2, cls=InfinityToNullEncoder
                    )
                logger.info(f"Benchmark results saved to {output_file_path}")

        except Exception as e:
            logger.error(f"Error during run for config {config_id}: {e}")

        finally:
            # Clean up
            if server_process:
                terminate_process_top_down(server_process)

            if idx < total_configs - 1:
                logger.info(f"Resting for {rest} seconds before the next run.")
                time.sleep(rest)

    if output_jsonl_path and output_jsonl_path.exists():
        all_results = []
        with open(output_jsonl_path) as f:
            for line in f:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Already warned about this

        # Find best throughput configurations
        best_configs = find_best_throughput_configs(all_results, parsed_constraints)

        # Extract input/output lengths from first result (they should be consistent)
        input_len = output_len = None
        if all_results:
            first_result = all_results[0]
            first_client_params = first_result.get("config", {}).get("client_args", {})
            input_len, output_len = extract_token_lengths(first_client_params)

        # Create enhanced result structure
        enhanced_results = {
            "metadata": {
                "gpu_type": gpu_type,
                "gpu_count": gpus,
                "model_tag": model,
                "total_tests": len(all_results),
                "constraints": constraints_for_viz,
                "input_len": input_len,
                "output_len": output_len,
            },
            "best_configurations": best_configs,
            "test_results": all_results,
        }

        with open(output_json, "w") as f:
            json.dump(enhanced_results, f, indent=2, cls=InfinityToNullEncoder)
        logger.info(
            f"All benchmark results saved to {output_json} with enhanced metadata"
        )

    logger.info("-" * 80)
    logger.info("All benchmark runs completed.")

    # Auto-generate HTML visualization if output JSON exists
    if output_json and pathlib.Path(output_json).exists():
        try:
            from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

            # Create optimizer instance with default config
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
            optimizer = ParetoLLMOptimizer(str(config_path))

            # Generate HTML with same base name as JSON
            json_path = pathlib.Path(output_json)
            html_file = json_path.with_suffix(".html")

            logger.info("Generating visualization dashboard...")
            optimizer.generate_dashboard(output_json, output_file=str(html_file))
            logger.info(f"Visualization dashboard saved to {html_file}")

        except Exception as e:
            logger.warning(f"Could not generate visualization dashboard: {e}")
            # Don't fail the benchmark if visualization fails


@cli.command()
@click.option(
    "--data-file",
    type=str,
    required=True,
    help="Path to the JSON data file to visualize",
)
@click.option(
    "--config", type=str, default=None, help="Path to visualization config file"
)
@click.option(
    "-o",
    "--output",
    type=str,
    default=None,
    help="Output HTML file path (default: pareto_llm_dashboard.html)",
)
@click.option("--serve", is_flag=True, help="Start HTTP server after generating HTML")
@click.option(
    "--port",
    type=int,
    default=8080,
    help="Port to run the dashboard server (used with --serve)",
)
def visualize(data_file, config, output, serve, port):
    """Generate and open visualization dashboard from benchmark results."""
    try:
        from llm_optimizer.visualization.visualize import ParetoLLMOptimizer

        # Determine config file path
        if config is None:
            config_path = (
                pathlib.Path(__file__).parent
                / "visualization"
                / "visualization_config.json"
            )
        else:
            config_path = pathlib.Path(config)

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            return

        # Create optimizer instance
        optimizer = ParetoLLMOptimizer(str(config_path))

        # Check if data file exists
        if not pathlib.Path(data_file).exists():
            logger.error(f"Data file not found: {data_file}")
            return

        # Generate dashboard
        logger.info(f"Generating dashboard from {data_file}...")
        html_file = optimizer.generate_dashboard(data_file, output_file=output)
        logger.info(f"Dashboard generated: {html_file}")

        # Start server only if requested
        if serve:
            logger.info(f"Starting server on port {port}...")
            optimizer.start_server(port=port)
        else:
            logger.info(f"HTML dashboard ready at: {html_file}")
            logger.info("Use --serve to start a local server and open in browser")

    except Exception as e:
        logger.error(f"Failed to generate visualization: {e}")
        import traceback

        traceback.print_exc()


@cli.command("estimate")
@click.option(
    "--model",
    type=str,
    required=False,  # Made optional for interactive mode
    help="HuggingFace model ID (e.g., 'meta-llama/Meta-Llama-3-8B')",
)
@click.option(
    "--input-len", type=int, required=False, help="Input sequence length in tokens"
)
@click.option(
    "--output-len", type=int, required=False, help="Output sequence length to generate"
)
@click.option(
    "--gpu",
    type=click.Choice(list_available_gpus_with_lowercase(), case_sensitive=False),
    help="GPU model (auto-detected if not specified, case-insensitive)",
)
@click.option(
    "--num-gpus", type=int, help="Number of GPUs (auto-detected if not specified)"
)
@click.option(
    "--precision",
    type=click.Choice(["fp16", "bf16", "fp8"]),
    default=None,
    help="Model precision (auto-inferred from model config if not specified)",
)
@click.option(
    "--framework",
    type=click.Choice(["sglang", "vllm", "both"]),
    default="both",
    help="Framework to optimize for",
)
@click.option(
    "--constraints", type=str, help="SLO constraints (e.g., 'ttft<300ms;itl<8.5ms')"
)
@click.option(
    "--target",
    type=click.Choice(["throughput", "latency"]),
    default="throughput",
    help="Optimization target",
)
@click.option("--interactive", is_flag=True, help="Run in interactive mode")
@click.option(
    "--dataset",
    type=click.Choice(["random", "sharegpt"]),
    default="random",
    help="Dataset to use for benchmarking (default: random)",
)
def estimate_performance(
    model,
    input_len,
    output_len,
    gpu,
    num_gpus,
    precision,
    framework,
    constraints,
    target,
    interactive,
    dataset,
):
    """Estimate LLM performance and suggest optimal configurations."""

    # Normalize GPU input if provided via CLI
    if gpu:
        gpu = normalize_gpu_choice(gpu)

    # Validate that required parameters are provided either via CLI or interactive mode
    if not interactive and (not model or input_len is None or output_len is None):
        click.echo(
            "Error: --model, --input-len, and --output-len are required "
            "when not using --interactive mode"
        )
        click.echo(
            "Use --interactive for guided input or provide all required parameters"
        )
        return

    try:
        # Parameter collection phase
        if interactive:
            # Collect interactive parameters
            interactive_params = collect_interactive_parameters()

            # Update parameters with interactive input
            model = model or interactive_params["model"]
            input_len = (
                input_len if input_len is not None else interactive_params["input_len"]
            )
            output_len = (
                output_len
                if output_len is not None
                else interactive_params["output_len"]
            )
            target = target or interactive_params["target"]
            constraints = constraints or interactive_params["constraints"]
            precision = precision or interactive_params["precision"]
            framework = framework or interactive_params["framework"]
            dataset = dataset or interactive_params["dataset"]

        # GPU configuration (both modes may need this)
        if interactive or not gpu or not num_gpus:
            gpu, num_gpus = collect_gpu_configuration(
                interactive=interactive, gpu=gpu, num_gpus=num_gpus
            )

        # Build parameters for common estimation function
        params = PerformanceEstimationParams(
            model=model,
            input_len=input_len,
            output_len=output_len,
            gpu=gpu,
            num_gpus=num_gpus,
            precision=precision,
            framework=framework,
            constraints=constraints,
            target=target,
            dataset=dataset,
        )

        # Run common estimation function
        updated_params, result = run_performance_estimation(params)

        # Display results consistently
        display_performance_estimation_results(updated_params, result)

    except Exception as e:
        click.echo(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    cli()

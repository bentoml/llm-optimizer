"""
Command generation functions for LLM optimizer tuning.

This module contains functions that generate CLI commands for running
tuning configurations with llm-optimizer.
"""

from llm_optimizer.tuning.core import TuningConfig


def generate_llm_optimizer_commands(
    configs: list[TuningConfig],
    model_id: str,
    input_length: int,
    output_length: int,
    num_gpus: int = 1,
    host: str = "127.0.0.1",
    output_dir: str = "tuning_results",
    constraints: str = None,
    dataset: str = "random",
) -> list[str]:
    """
    Generate llm-optimizer CLI commands using args.py format.

    Args:
        configs: List of tuning configurations with args.py format strings
        model_id: HuggingFace model identifier
        input_length: Input sequence length
        output_length: Output sequence length
        num_gpus: Number of GPUs
        host: Server host
        output_dir: Output directory for results
        constraints: SLO constraints string to include in commands
        dataset: Dataset type ('random' or 'sharegpt')

    Returns:
        List of CLI commands to run
    """
    commands = []

    for i, config in enumerate(configs):
        # Build basic command structure
        cmd_parts = [
            "llm-optimizer",
            f"--framework {config.framework}",
            f"--model {model_id}",
            f"--gpus {num_gpus}",
            f"--host {host}",
        ]

        # Add server args if present
        if config.server_args_str.strip():
            cmd_parts.append(f'--server-args "{config.server_args_str}"')

        # Build client args with dataset-specific parameters
        fixed_client_args = [
            f"dataset_name={dataset}",
        ]

        # Add dataset-specific length arguments
        if dataset == "random":
            fixed_client_args.extend([
                f"random_input_len={input_length}",
                f"random_output_len={output_length}",
                "random_range_ratio=0.95",
            ])
        elif dataset == "sharegpt":
            fixed_client_args.extend([
                f"sharegpt_output_len={output_length}",
            ])

        # Combine fixed and tunable client args
        if config.client_args_str.strip():
            client_args_combined = ";".join(fixed_client_args + [config.client_args_str])
        else:
            client_args_combined = ";".join(fixed_client_args)

        cmd_parts.append(f'--client-args "{client_args_combined}"')

        # Add output options
        cmd_parts.extend([
            f"--output-dir {output_dir}",
            f"--output-json {output_dir}/config_{i + 1}_{config.framework}.json"
        ])

        # Add constraints if provided
        if constraints:
            cmd_parts.append(f'--constraints "{constraints}"')

        cmd = " ".join(cmd_parts)
        commands.append(cmd)

    return commands

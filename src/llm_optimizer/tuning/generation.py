"""
Configuration generation functions for LLM tuning.

This module contains functions that generate various types of tuning configurations
for different frameworks and use cases.
"""

from llm_optimizer.args import ArgScope, ArgSet
from llm_optimizer.common import (
    ModelConfig,
    calculate_min_tensor_parallel_size,
    generate_parameter_range,
    generate_tp_dp_combinations,
)
from llm_optimizer.performance import get_parameter_conservativeness_for_stat_type
from llm_optimizer.predefined import PARAMETER_MAPPINGS
from llm_optimizer.predefined.gpus import get_gpu_specs
from llm_optimizer.resources import GPUResourceManager
from llm_optimizer.tuning.core import (
    TuningConfig,
    calculate_chunked_prefill_size,
    calculate_memory_fraction,
    calculate_optimal_batch_tokens,
    calculate_optimal_max_seqs,
)


def generate_common_base_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    target_throughput: bool = True,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[dict]:
    """
    Generate common base configurations that work for both SGLang and vLLM.

    Returns dictionaries with server_arg_sets and client_arg_sets directly.

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        target_throughput: Whether to optimize for throughput (True) or latency (False)
        precision: Model precision
        sequence_length: Typical sequence length for calculations

    Returns:
        List of dictionaries with server_arg_sets, client_arg_sets, and description
    """
    # Initialize resource manager and get GPU specifications
    gpu_manager = GPUResourceManager()
    gpu_manager.get_total_resources(num_gpus, gpu_name, precision)

    # For backward compatibility, extract gpu_specs dict
    gpu_specs = get_gpu_specs(gpu_name)

    mapping = PARAMETER_MAPPINGS[framework.lower()]

    # Calculate optimal parameters
    max_seqs_configs = calculate_optimal_max_seqs(
        gpu_specs, model_config, precision, sequence_length, optimal_concurrency, target_throughput
    )
    calculate_memory_fraction(gpu_specs, model_config, precision, conservative=True)

    configs = []

    # Configuration 1: Conservative baseline
    server_arg_sets = []
    client_arg_sets = []

    # Add base client args
    max_concurrency = optimal_concurrency // 2
    num_prompts = max(1000, max_concurrency * 2)
    client_arg_sets.extend([
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[num_prompts]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[max_concurrency])
    ])

    # Add server args using framework mapping
    max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")

    if max_seqs_param:
        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=max_seqs_param,
            arg_type=int,
            values=[max_seqs_configs['conservative']]
        ))

    # Add multi-GPU parallelization for baseline
    if num_gpus > 1:
        min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)

        # Skip multi-GPU config if we don't have enough GPUs for minimum TP size
        if num_gpus < min_tp_size:
            # Can't fit this model with available GPUs
            pass  # Skip adding parallelization args
        else:
            tp_param = mapping.get("tensor_parallel", "tensor_parallel")
            dp_param = mapping.get("data_parallel", "data_parallel")

            if target_throughput:
                # Prefer data parallelism for throughput, but ensure TP meets minimum requirement
                tp_value = min_tp_size
                dp_value = num_gpus // tp_value
            else:
                # Use tensor parallelism for latency
                tp_value = min(min_tp_size, num_gpus, 8)  # Cap TP size
                dp_value = num_gpus // tp_value if tp_value < num_gpus else 1

            # Create composite ArgSet for TP/DP combination
            if tp_param and dp_param:
                server_arg_sets.append(ArgSet(
                    scope=ArgScope.SERVER,
                    name=(tp_param, dp_param),
                    arg_type=(int, int),
                    values=[(tp_value, dp_value)]
                ))

    configs.append({
        "server_arg_sets": server_arg_sets,
        "client_arg_sets": client_arg_sets,
        "description": "Conservative baseline - stable performance",
    })

    # Configuration 2: Aggressive throughput (if targeting throughput)
    if target_throughput:
        server_arg_sets = []
        client_arg_sets = []

        # Add base client args with higher concurrency
        max_concurrency = optimal_concurrency
        num_prompts = max(1000, max_concurrency * 2)
        client_arg_sets.extend([
            ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[num_prompts]),
            ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[max_concurrency])
        ])

        # Add server args
        if max_seqs_param:
            server_arg_sets.append(ArgSet(
                scope=ArgScope.SERVER,
                name=max_seqs_param,
                arg_type=int,
                values=[max_seqs_configs['aggressive']]
            ))


        # Add same parallelization strategy as config1
        if num_gpus > 1 and num_gpus >= min_tp_size and tp_param and dp_param:
            if target_throughput:
                # Use same strategy as conservative config - prefer DP but respect min_tp_size
                tp_value = min_tp_size
                dp_value = num_gpus // tp_value
            else:
                tp_value = min(min_tp_size, num_gpus, 8)
                dp_value = num_gpus // tp_value if tp_value < num_gpus else 1

            server_arg_sets.append(ArgSet(
                scope=ArgScope.SERVER,
                name=(tp_param, dp_param),
                arg_type=(int, int),
                values=[(tp_value, dp_value)]
            ))

        configs.append({
            "server_arg_sets": server_arg_sets,
            "client_arg_sets": client_arg_sets,
            "description": "Aggressive throughput - maximum request intake",
        })

    # Configuration 3: Memory optimized
    server_arg_sets = []
    client_arg_sets = []

    # Add base client args with lower concurrency
    max_concurrency = optimal_concurrency // 4
    num_prompts = max(1000, max_concurrency * 2)
    client_arg_sets.extend([
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[num_prompts]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=[max_concurrency])
    ])

    # Add server args
    if max_seqs_param:
        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=max_seqs_param,
            arg_type=int,
            values=[max_seqs_configs['memory_efficient']]
        ))


    # Use conservative parallelization for memory config
    if num_gpus > 1 and num_gpus >= min_tp_size and tp_param and dp_param:
        # Prefer smaller parallelization for memory efficiency, but respect min_tp_size
        max_dp = num_gpus // min_tp_size  # Maximum DP without violating min TP
        dp_value = min(2, max_dp) if max_dp > 0 else 1
        tp_value = num_gpus // dp_value

        # Ensure TP meets minimum requirement
        if tp_value < min_tp_size:
            tp_value = min_tp_size
            dp_value = num_gpus // tp_value

        server_arg_sets.append(ArgSet(
            scope=ArgScope.SERVER,
            name=(tp_param, dp_param),
            arg_type=(int, int),
            values=[(tp_value, dp_value)]
        ))

    configs.append({
        "server_arg_sets": server_arg_sets,
        "client_arg_sets": client_arg_sets,
        "description": "Memory efficient - conservative memory usage",
    })

    return configs


def generate_simple_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate simplified tuning configurations using args.py framework:
    - max_concurrency for client_args: 3 values [n/2, n, n+n/2]
    - tp*dp for server_args: Composite argument for multi-GPU
    - Let frameworks use their default memory utilization

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length

    Returns:
        List of simplified tuning configurations using ArgSet format
    """
    configs = []

    # Generate 3 concurrency values using standard parameter range
    # This produces [n/2, n, 3n/2] with 50% variation
    concurrency_values = generate_parameter_range(optimal_concurrency, num_values=3, variation_factor=0.5)

    # Client ArgSets - max_concurrency is universal across frameworks
    num_prompts = max(1000, max(*concurrency_values) * 2)
    client_arg_sets = [
        ArgSet(scope=ArgScope.CLIENT, name="num_prompts", arg_type=int, values=[num_prompts]),
        ArgSet(scope=ArgScope.CLIENT, name="max_concurrency", arg_type=int, values=concurrency_values)
    ]

    if num_gpus > 1:
        # Generate TP/DP combinations for multi-GPU
        gpu_specs = get_gpu_specs(gpu_name)
        min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
        tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size)

        # Create composite ArgSet using framework-specific parameter names
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        tp_param = mapping.get("tensor_parallel", "tensor_parallel")
        dp_param = mapping.get("data_parallel", "data_parallel")

        server_arg_sets = [
            ArgSet(
                scope=ArgScope.SERVER,
                name=(tp_param, dp_param),
                arg_type=(int, int),
                values=tp_dp_combinations
            )
        ]

        config_desc = f"Simple - {framework.upper()} TP/DP: {tp_dp_combinations}"

        configs.append(TuningConfig(
            framework=framework,
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=config_desc
        ))
    else:
        # Single GPU - no server args needed
        configs.append(TuningConfig(
            framework=framework,
            server_arg_sets=[],  # No server args for single GPU
            client_arg_sets=client_arg_sets,
            description="Simple - Single GPU"
        ))

    return configs


def generate_advanced_tuning_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
) -> list[TuningConfig]:
    """
    Generate advanced tuning configurations that inherit from simple configs and add more parameters.

    Stage 2: Advanced tuning inherits all simple tuning base settings and adds more server parameters:
    - Starts with simple tuning configurations as base
    - Adds key server parameters with 3-value ranges
    - SGLang: chunked_prefill_size, schedule_conservativeness, schedule_policy
    - vLLM: max_num_batched_tokens

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs available
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length

    Returns:
        List of advanced tuning configurations that inherit from simple configs
    """
    # Start by getting the simple configs as the base
    simple_configs = generate_simple_tuning_configs(
        framework=framework,
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        optimal_concurrency=optimal_concurrency,
        precision=precision,
        sequence_length=sequence_length,
    )

    if not simple_configs:
        return []

    # Use the first simple config as base (they all have the same TP/DP and concurrency structure)
    base_config = simple_configs[0]
    gpu_specs = get_gpu_specs(gpu_name)

    advanced_configs = []

    if framework.lower() == "sglang":
        # Calculate optimal values and generate 3-value ranges for advanced parameters
        optimal_chunked_prefill = calculate_chunked_prefill_size(gpu_specs, model_config, precision, target_throughput=True)
        prefill_values = generate_parameter_range(optimal_chunked_prefill, min_val=1024, max_val=16384)
        conservativeness_values = [0.3, 0.6, 1.0]  # Aggressive to conservative

        # Build advanced server args by extending the base config
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        prefill_param = mapping.get("prefill_chunk_size", "prefill_chunk_size")

        # Copy base ArgSets and add advanced parameters
        server_arg_sets = base_config.server_arg_sets.copy()
        client_arg_sets = base_config.client_arg_sets.copy()

        # Add advanced server parameters
        server_arg_sets.extend([
            ArgSet(scope=ArgScope.SERVER, name=prefill_param, arg_type=int, values=prefill_values),
            ArgSet(scope=ArgScope.SERVER, name="schedule_conservativeness", arg_type=float, values=conservativeness_values),
            ArgSet(scope=ArgScope.SERVER, name="schedule_policy", arg_type=str, values=["fcfs"])
        ])

        advanced_configs.append(TuningConfig(
            framework="sglang",
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=f"Advanced - SGLang with prefill tuning: {prefill_values}"
        ))

    elif framework.lower() == "vllm":
        # Calculate optimal values and generate 3-value ranges for advanced parameters
        optimal_batch_tokens = calculate_optimal_batch_tokens(gpu_specs, model_config, precision, sequence_length)
        batch_values = generate_parameter_range(optimal_batch_tokens, min_val=1024, max_val=32768)

        # Build advanced server args by extending the base config
        mapping = PARAMETER_MAPPINGS[framework.lower()]
        batch_param = mapping.get("batch_size", "batch_size")

        # Copy base ArgSets and add advanced parameters
        server_arg_sets = base_config.server_arg_sets.copy()
        client_arg_sets = base_config.client_arg_sets.copy()

        # Add advanced server parameters
        server_arg_sets.append(
            ArgSet(scope=ArgScope.SERVER, name=batch_param, arg_type=int, values=batch_values)
        )

        advanced_configs.append(TuningConfig(
            framework="vllm",
            server_arg_sets=server_arg_sets,
            client_arg_sets=client_arg_sets,
            description=f"Advanced - vLLM with batch tuning: {batch_values}"
        ))

    return advanced_configs


def generate_simplified_throughput_configs(
    framework: str,
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    optimal_concurrency: int,
    precision: str = "fp16",
    sequence_length: int = 2048,
    constraints: list = None,
) -> list[TuningConfig]:
    """
    Generate simplified high-throughput configurations with parameter ranges.

    This generates only one optimized configuration per framework with parameter ranges
    for crucial tuning parameters, as specified in the improvement plan.

    Args:
        framework: Framework name ("sglang" or "vllm")
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        optimal_concurrency: Optimal concurrency level
        precision: Model precision
        sequence_length: Typical sequence length
        constraints: SLO constraints (for parameter conservativeness)

    Returns:
        List with single optimized TuningConfig containing parameter ranges
    """
    # Get GPU specifications
    gpu_specs = get_gpu_specs(gpu_name)

    # Calculate optimal parameters
    optimal_batch_tokens = calculate_optimal_batch_tokens(
        gpu_specs, model_config, precision, sequence_length
    )

    # Determine conservativeness based on constraints
    if constraints:
        # Use most restrictive constraint's stat_type
        max(
            get_parameter_conservativeness_for_stat_type(c.stat_type)
            for c in constraints
        )

    # Generate parameter ranges
    concurrency_range = generate_parameter_range(optimal_concurrency)
    batch_token_range = generate_parameter_range(optimal_batch_tokens, min_val=1024, max_val=32768)

    # Multi-GPU TP/DP combinations
    min_tp_size = calculate_min_tensor_parallel_size(model_config, gpu_specs, precision)
    tp_dp_combinations = generate_tp_dp_combinations(num_gpus, min_tp_size) if num_gpus > 1 else [(1, 1)]

    configs = []

    if framework.lower() == "vllm":
        # Base server args
        server_args = []

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[framework.lower()]

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            tp_param = mapping.get("tensor_parallel", "tensor_parallel")
            server_args.append(f"{tp_param}=[{','.join(tp_options)}]")

        # Add parameter ranges
        batch_param = mapping.get("batch_size", "batch_size")
        max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")
        server_args.extend([
            f"{batch_param}=[{','.join(map(str, batch_token_range))}]",
            f"{max_seqs_param}=[{','.join(map(str, concurrency_range))}]",
        ])

        # Client args with concurrency range
        client_args = [f"max_concurrency=[{','.join(map(str, concurrency_range))}]"]

        configs.append(TuningConfig(
            framework="vllm",
            server_args=server_args,
            client_args=client_args,
            description=f"High-throughput optimized - batch_tokens: {batch_token_range}, concurrency: {concurrency_range}"
        ))

    elif framework.lower() == "sglang":
        # Calculate chunked prefill size
        chunked_prefill_size = calculate_chunked_prefill_size(
            gpu_specs, model_config, precision, target_throughput=True
        )
        prefill_range = [chunked_prefill_size // 2, chunked_prefill_size, chunked_prefill_size * 2]
        prefill_range = [p for p in prefill_range if 1024 <= p <= 16384]  # Keep in reasonable range

        # Base server args
        server_args = []

        # Get parameter mapping
        mapping = PARAMETER_MAPPINGS[framework.lower()]

        # Add TP/DP combinations if multi-GPU
        if num_gpus > 1:
            tp_options = [str(tp) for tp, dp in tp_dp_combinations]
            dp_options = [str(dp) for tp, dp in tp_dp_combinations]
            tp_param = mapping.get("tensor_parallel", "tensor_parallel")
            dp_param = mapping.get("data_parallel", "data_parallel")
            server_args.extend([
                f"{tp_param}=[{','.join(tp_options)}]",
                f"{dp_param}=[{','.join(dp_options)}]",
            ])

        # Add parameter ranges using mapping
        prefill_param = mapping.get("prefill_chunk_size", "prefill_chunk_size")
        max_seqs_param = mapping.get("max_concurrent_requests", "max_concurrent_requests")

        server_args.extend([
            "schedule_conservativeness=0.3",  # Aggressive for throughput
            f"{prefill_param}=[{','.join(map(str, prefill_range))}]",
            f"{max_seqs_param}=[{','.join(map(str, concurrency_range))}]",
            "schedule_policy=fcfs",
        ])

        # Client args with concurrency range
        client_args = [f"max_concurrency=[{','.join(map(str, concurrency_range))}]"]

        configs.append(TuningConfig(
            framework="sglang",
            server_args=server_args,
            client_args=client_args,
            description=f"High-throughput optimized - prefill: {prefill_range}, concurrency: {concurrency_range}"
        ))

    return configs

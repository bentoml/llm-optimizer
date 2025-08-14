"""
Core tuning functionality including TuningConfig class and calculation functions.

This module contains the fundamental building blocks for LLM parameter tuning.
"""

from dataclasses import dataclass

from llm_optimizer.args import ArgSet, arg_sets_to_arg_str
from llm_optimizer.common import (
    ModelConfig,
    calculate_model_memory_bytes,
)
from llm_optimizer.predefined.gpus import get_precision_tflops
from llm_optimizer.resources import (
    ModelMemoryCalculator,
)


@dataclass
class TuningConfig:
    """Configuration for parameter tuning using args.py framework."""

    framework: str
    server_arg_sets: list[ArgSet]  # List of server argument sets
    client_arg_sets: list[ArgSet]  # List of client argument sets
    description: str

    @property
    def server_args_str(self) -> str:
        """Convert server ArgSets to argument string for backward compatibility."""
        return arg_sets_to_arg_str(self.server_arg_sets)

    @property
    def client_args_str(self) -> str:
        """Convert client ArgSets to argument string for backward compatibility."""
        return arg_sets_to_arg_str(self.client_arg_sets)


def calculate_optimal_batch_tokens(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
    memory_utilization: float = 0.8
) -> int:
    """Calculate optimal batch tokens based on GPU memory and bandwidth."""

    # Initialize memory calculator
    memory_calculator = ModelMemoryCalculator()

    # Calculate model memory
    model_memory_bytes = memory_calculator.calculate_model_memory(model_config, precision)
    model_memory_gb = model_memory_bytes / (1024**3)
    available_memory_gb = gpu_specs["VRAM_GB"] * memory_utilization - model_memory_gb

    if available_memory_gb <= 0:
        # Model doesn't fit, return minimum
        return 1024

    # Calculate memory breakdown for one token
    memory_breakdown = memory_calculator.calculate_total_memory_needed(
        model_config,
        batch_size=1,
        sequence_length=1,
        model_precision=precision
    )

    # Get per-token memory (KV cache + activations)
    total_memory_per_token = memory_breakdown.kv_cache_per_token_bytes + \
                           (memory_breakdown.activation_memory_bytes / sequence_length)

    # Calculate max batch tokens based on available memory
    available_memory_bytes = available_memory_gb * 1e9
    max_batch_tokens = int(available_memory_bytes / total_memory_per_token)

    # Apply bandwidth-based heuristics
    # Higher bandwidth GPUs can handle larger batches more efficiently
    bandwidth_factor = min(2.0, gpu_specs["Memory_Bandwidth_GBs"] / 2000)  # Normalize around 2TB/s

    # Suggest batch sizes that are powers of 2 or multiples of 1024 for efficiency
    optimal_batch = int(max_batch_tokens * bandwidth_factor)

    # Round to nearest efficient batch size
    efficient_sizes = [1024, 2048, 4096, 6144, 8192, 12288, 16384, 24576, 32768]
    optimal_batch = min(efficient_sizes, key=lambda x: abs(x - optimal_batch))

    return max(1024, optimal_batch)  # Minimum 1024 tokens


def calculate_optimal_max_seqs(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    sequence_length: int,
    optimal_concurrency: int,
    target_throughput: bool = True
) -> dict:
    """Calculate optimal max_seqs values for different performance targets."""

    # Base calculations using GPU specs
    tflops = get_precision_tflops(gpu_specs, precision)
    memory_bandwidth = gpu_specs["Memory_Bandwidth_GBs"]

    # Compute capacity limit (based on FLOPS)
    # Higher FLOPS GPUs can handle more concurrent sequences
    compute_scale = min(2.0, tflops / 300)  # Normalize around 300 TFLOPS

    # Memory bandwidth limit (higher bandwidth = more concurrent sequences)
    bandwidth_scale = min(2.0, memory_bandwidth / 2000)  # Normalize around 2TB/s

    # Combine factors
    capacity_factor = (compute_scale + bandwidth_scale) / 2

    # Calculate different configurations
    configs = {
        "conservative": max(16, int(optimal_concurrency * 0.25)),
        "balanced": max(32, int(optimal_concurrency * 0.5 * capacity_factor)),
        "aggressive": max(64, int(optimal_concurrency * capacity_factor)),
        "latency_optimized": min(32, int(optimal_concurrency * 0.3)),
        "memory_efficient": max(8, int(optimal_concurrency * 0.125))
    }

    return configs


def calculate_chunked_prefill_size(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    target_throughput: bool = True
) -> int:
    """Calculate optimal chunked prefill size based on GPU specs."""

    # Larger, more powerful GPUs can handle bigger prefill chunks
    tflops = get_precision_tflops(gpu_specs, precision)
    memory_bandwidth = gpu_specs["Memory_Bandwidth_GBs"]

    # Scale based on compute and memory capabilities
    compute_factor = min(2.0, tflops / 300)  # Normalize around 300 TFLOPS
    memory_factor = min(2.0, memory_bandwidth / 2000)  # Normalize around 2TB/s

    # Base chunk size
    base_chunk = 2048 if target_throughput else 1024

    # Apply scaling
    scaled_chunk = int(base_chunk * (compute_factor + memory_factor) / 2)

    # Round to powers of 2
    chunk_sizes = [1024, 2048, 4096, 8192, 16384]
    optimal_chunk = min(chunk_sizes, key=lambda x: abs(x - scaled_chunk))

    return optimal_chunk


def calculate_memory_fraction(
    gpu_specs: dict,
    model_config: ModelConfig,
    precision: str,
    conservative: bool = False
) -> float:
    """Calculate optimal GPU memory utilization fraction."""

    # Calculate model memory requirements
    model_memory_bytes = calculate_model_memory_bytes(model_config, precision)
    model_memory_gb = model_memory_bytes / (1024**3)
    total_vram_gb = gpu_specs["VRAM_GB"]

    # Model memory ratio
    model_ratio = model_memory_gb / total_vram_gb

    if conservative:
        # Conservative: leave more headroom
        if model_ratio > 0.6:  # Large model
            return 0.8
        elif model_ratio > 0.3:  # Medium model
            return 0.85
        else:  # Small model
            return 0.9
    else:
        # Aggressive: use more memory
        if model_ratio > 0.6:  # Large model
            return 0.9
        elif model_ratio > 0.3:  # Medium model
            return 0.95
        else:  # Small model
            return 0.95


def get_precision_tflops(gpu_specs: dict, precision: str) -> float:
    """Get TFLOPS for the specified precision from GPU specs."""
    if precision == "fp16" or precision == "bf16":
        return gpu_specs["FP16_TFLOPS"]
    elif precision == "fp8":
        if gpu_specs["FP8_TFLOPS"] is None:
            raise ValueError(f"FP8 not supported on this GPU architecture: {gpu_specs.get('Architecture', 'Unknown')}")
        return gpu_specs["FP8_TFLOPS"]
    else:
        raise ValueError(f"Unsupported precision: {precision}")

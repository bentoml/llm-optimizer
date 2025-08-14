"""Tuning management for LLM optimizer.

This package provides comprehensive tuning functionality including configuration generation,
parameter calculation, and command generation for different LLM frameworks.
"""

# Core functionality
# Command generation
from llm_optimizer.tuning.commands import generate_llm_optimizer_commands
from llm_optimizer.tuning.core import (
    TuningConfig,
    calculate_chunked_prefill_size,
    calculate_memory_fraction,
    calculate_optimal_batch_tokens,
    calculate_optimal_max_seqs,
    get_precision_tflops,
)

# Configuration generation functions
from llm_optimizer.tuning.generation import (
    generate_advanced_tuning_configs,
    generate_common_base_configs,
    generate_simple_tuning_configs,
    generate_simplified_throughput_configs,
)

# Strategy pattern for framework-specific generation
from llm_optimizer.tuning.strategy import (
    FrameworkStrategy,
    SGLangStrategy,
    VLLMStrategy,
    generate_tuning_configs,
    get_strategy_for_framework,
)

__all__ = [
    # Core classes and data structures
    "TuningConfig",

    # Calculation functions
    "calculate_optimal_batch_tokens",
    "calculate_optimal_max_seqs",
    "calculate_chunked_prefill_size",
    "calculate_memory_fraction",
    "get_precision_tflops",

    # Strategy pattern classes
    "FrameworkStrategy",
    "SGLangStrategy",
    "VLLMStrategy",
    "get_strategy_for_framework",

    # Main configuration generation functions
    "generate_tuning_configs",
    "generate_common_base_configs",
    "generate_simple_tuning_configs",
    "generate_advanced_tuning_configs",
    "generate_simplified_throughput_configs",

    # Command generation
    "generate_llm_optimizer_commands",
]

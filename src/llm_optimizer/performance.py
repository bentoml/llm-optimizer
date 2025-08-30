"""
LLM performance estimation module.

This module provides functions to estimate theoretical LLM performance
including latency, throughput, and optimal configurations.
"""

import typing as t
from dataclasses import dataclass
from typing import Optional

import click

from llm_optimizer.common import (
    ModelConfig,
    get_model_config_and_precision_from_hf,
    get_precision_bytes_per_param,
)
from llm_optimizer.resources import (
    GPUResourceManager,
    ModelMemoryCalculator,
)


def calculate_transformer_flops(
    model_config: "ModelConfig",
    sequence_length: int,
    include_attention: bool = True,
    include_mlp: bool = True,
) -> dict[str, float]:
    """
    Calculate FLOPS for transformer components based on roofline analysis methodology.

    This function implements the detailed FLOPS calculation from academic literature,
    accounting for both linear operations (MLP, projections) and quadratic attention terms.

    References:
    - "Attention Is All You Need" (Vaswani et al.)
    - JAX Scaling Book: https://jax-ml.github.io/scaling-book/transformers/
    - OpenAI Scaling Laws paper

    Args:
        model_config: Model configuration with architecture details
        sequence_length: Input sequence length (T)
        include_attention: Whether to include attention FLOPS (default True)
        include_mlp: Whether to include MLP/feedforward FLOPS (default True)

    Returns:
        Dictionary with FLOPS breakdown:
        - 'attention_qkv': Query/Key/Value projection FLOPS
        - 'attention_scores': QK^T matmul FLOPS
        - 'attention_softmax': Softmax operation FLOPS
        - 'attention_output': Attention-Value matmul FLOPS
        - 'attention_proj': Output projection FLOPS
        - 'mlp': Feed-forward network FLOPS
        - 'total': Total FLOPS per token
    """
    # Model parameters
    d_model = model_config.hidden_dim  # Hidden dimension
    n_layers = model_config.num_layers  # Number of transformer layers
    n_heads = model_config.num_heads    # Number of attention heads
    n_kv_heads = model_config.num_kv_heads  # Number of KV heads (for GQA/MQA)
    d_head = d_model // n_heads         # Dimension per head

    # Calculate intermediate dimension (typically 4x hidden dim in standard transformers)
    # We'll estimate it from total parameters if not directly available
    # Standard transformer: d_ff ≈ 4 * d_model
    d_ff = 4 * d_model  # Feed-forward intermediate dimension

    flops_breakdown = {}

    if include_attention:
        # 1. QKV Projections: Linear transformations to create queries, keys, values
        # Q: [B, T, d_model] @ [d_model, d_model] -> [B, T, d_model]
        # K,V: [B, T, d_model] @ [d_model, d_kv] -> [B, T, d_kv] where d_kv = n_kv_heads * d_head
        d_kv = n_kv_heads * d_head
        qkv_flops = 2 * sequence_length * (d_model * d_model + 2 * d_model * d_kv)
        flops_breakdown['attention_qkv'] = qkv_flops * n_layers

        # 2. Attention Scores: Q @ K^T
        # [B, n_heads, T, d_head] @ [B, n_heads, d_head, T] -> [B, n_heads, T, T]
        # Note: For GQA/MQA, keys/values are broadcast across query heads
        qk_flops = 2 * n_heads * sequence_length * sequence_length * d_head
        flops_breakdown['attention_scores'] = qk_flops * n_layers

        # 3. Softmax: Applied to attention scores
        # Includes exponential, sum reduction, and division operations
        # Approximation: ~3 ops per element for softmax
        softmax_flops = 3 * n_heads * sequence_length * sequence_length
        flops_breakdown['attention_softmax'] = softmax_flops * n_layers

        # 4. Attention Output: Softmax @ V
        # [B, n_heads, T, T] @ [B, n_heads, T, d_head] -> [B, n_heads, T, d_head]
        av_flops = 2 * n_heads * sequence_length * sequence_length * d_head
        flops_breakdown['attention_output'] = av_flops * n_layers

        # 5. Output Projection: Final linear layer
        # [B, T, d_model] @ [d_model, d_model] -> [B, T, d_model]
        proj_flops = 2 * sequence_length * d_model * d_model
        flops_breakdown['attention_proj'] = proj_flops * n_layers
    else:
        for key in ['attention_qkv', 'attention_scores', 'attention_softmax',
                   'attention_output', 'attention_proj']:
            flops_breakdown[key] = 0.0

    if include_mlp:
        # MLP/Feed-Forward Network: Two linear transformations with activation
        # Up projection: [B, T, d_model] @ [d_model, d_ff] -> [B, T, d_ff]
        # Down projection: [B, T, d_ff] @ [d_ff, d_model] -> [B, T, d_model]
        # Note: Modern transformers often use SwiGLU which has additional complexity
        mlp_flops = 2 * sequence_length * (d_model * d_ff + d_ff * d_model)
        flops_breakdown['mlp'] = mlp_flops * n_layers
    else:
        flops_breakdown['mlp'] = 0.0

    # Total FLOPS per token
    flops_breakdown['total'] = sum(flops_breakdown.values())

    return flops_breakdown


def calculate_memory_access_bytes(
    model_config: "ModelConfig",
    sequence_length: int,
    batch_size: int = 1,
    bytes_per_param: int = 2,  # FP16 = 2 bytes
    include_kv_cache: bool = True,
) -> dict[str, float]:
    """
    Calculate memory access patterns for transformer inference following roofline methodology.

    This function computes the actual bytes that need to be moved from memory during
    transformer inference, which is crucial for determining arithmetic intensity and
    whether operations are memory-bound or compute-bound.

    References:
    - Roofline analysis papers and methodology
    - "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363)

    Args:
        model_config: Model architecture configuration
        sequence_length: Current sequence length (including generated tokens)
        batch_size: Number of concurrent sequences
        bytes_per_param: Bytes per parameter (2 for FP16, 1 for FP8, etc.)
        include_kv_cache: Whether to include KV cache access in calculation

    Returns:
        Dictionary with memory access breakdown:
        - 'model_weights': Bytes for loading model parameters
        - 'kv_cache': Bytes for KV cache access
        - 'activations': Bytes for intermediate activations
        - 'total': Total bytes accessed per token
    """
    d_model = model_config.hidden_dim
    n_layers = model_config.num_layers
    n_kv_heads = model_config.num_kv_heads
    d_head = d_model // model_config.num_heads

    memory_breakdown = {}

    # 1. Model Weights Access
    # During decode, we need to load:
    # - QKV projection weights: d_model * (d_model + 2*n_kv_heads*d_head) per layer
    # - Output projection weights: d_model * d_model per layer
    # - MLP weights: d_model * d_ff * 2 per layer (up and down projections)
    d_kv = n_kv_heads * d_head
    d_ff = 4 * d_model  # Estimated feed-forward dimension

    weights_per_layer = (
        d_model * (d_model + 2 * d_kv) +  # QKV projections
        d_model * d_model +                # Output projection
        d_model * d_ff * 2                 # MLP up/down projections
    )

    total_model_weights = weights_per_layer * n_layers * bytes_per_param
    memory_breakdown['model_weights'] = total_model_weights

    # 2. KV Cache Access
    if include_kv_cache:
        # KV cache size per token: 2 (K+V) * n_layers * n_kv_heads * d_head * bytes_per_param
        kv_per_token = 2 * n_layers * n_kv_heads * d_head * bytes_per_param

        # During decode: we access all previous tokens' KV cache
        # During prefill: we write to KV cache (similar access pattern)
        total_kv_access = kv_per_token * sequence_length * batch_size
        memory_breakdown['kv_cache'] = total_kv_access
    else:
        memory_breakdown['kv_cache'] = 0.0

    # 3. Activations
    # Intermediate activations during forward pass
    # This includes attention scores, MLP activations, etc.
    # Approximation based on sequence length and model dimensions
    activations_size = (
        batch_size * sequence_length * d_model * bytes_per_param +  # Input/output activations
        batch_size * model_config.num_heads * sequence_length * sequence_length * bytes_per_param  # Attention matrices
    )
    memory_breakdown['activations'] = activations_size

    # Total memory access per token generation
    memory_breakdown['total'] = sum(memory_breakdown.values())

    return memory_breakdown


def calculate_arithmetic_intensity(
    flops: float,
    memory_bytes: float,
) -> float:
    """
    Calculate arithmetic intensity following roofline analysis methodology.

    Arithmetic Intensity (AI) = Total FLOPS / Total Bytes Moved

    This is the key metric in roofline analysis that determines whether
    an operation is compute-bound or memory-bound.

    Args:
        flops: Total floating-point operations
        memory_bytes: Total bytes moved from memory

    Returns:
        Arithmetic intensity in operations per byte
    """
    if memory_bytes <= 0:
        return float('inf')
    return flops / memory_bytes


def determine_performance_bound(
    arithmetic_intensity: float,
    hardware_ops_per_byte: float,
) -> bool:
    """
    Determine if operation is memory-bound based on roofline analysis.

    Args:
        arithmetic_intensity: Operations per byte for this workload
        hardware_ops_per_byte: Hardware's peak ops/byte ratio (TFLOPS / TB/s)

    Returns:
        True if memory-bound, False if compute-bound
    """
    return arithmetic_intensity < hardware_ops_per_byte



@dataclass
class PerformanceResult:
    """Performance estimation result with roofline analysis."""

    ttft_ms: float
    itl_ms: float
    e2e_latency_s: float
    output_throughput_tps: float
    input_throughput_tps: float
    requests_per_sec: float
    bottleneck_is_memory: bool
    concurrency: int
    memory_needed_gb: float = 0.0
    usable_vram_gb: float = 0.0

    # Roofline analysis metrics
    prefill_arithmetic_intensity: float = 0.0
    decode_arithmetic_intensity: float = 0.0
    hardware_ops_per_byte: float = 0.0
    prefill_is_memory_bound: bool = False
    decode_is_memory_bound: bool = False


@dataclass
class SLOConstraint:
    """SLO constraint specification with statistical metric support."""

    metric: str  # ttft, itl, e2e_latency
    stat_type: str  # mean, median, p95, p99
    operator: str  # <, >, <=, >=
    value: float
    unit: str  # ms, s




def estimate_llm_performance(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    precision: str,
    concurrency: int,
    input_length: int,
    output_length: int,
    mfu_prefill: float = 0.45,
    mfu_decode: float = 0.30,
    vram_util_factor: float = 0.90,
) -> PerformanceResult:
    """
    Estimate LLM performance using roofline analysis methodology.

    This function implements proper roofline analysis for transformer inference,
    distinguishing between prefill (compute-bound) and decode (memory-bound) phases,
    calculating arithmetic intensity, and providing accurate bottleneck analysis.

    Methodology based on:
    - JAX Scaling Book roofline analysis: https://jax-ml.github.io/scaling-book/roofline/
    - "LLM Inference Unveiled: Survey and Roofline Model Insights" (arXiv:2402.16363)
    - Transformer inference optimization literature

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name (e.g., "H100", "A100")
        model_config: Model configuration with architecture details
        precision: Precision ("fp16" or "fp8")
        concurrency: Number of concurrent requests (batch size)
        input_length: Input sequence length in tokens
        output_length: Output sequence length to generate
        mfu_prefill: Model FLOPs utilization for prefill phase (default 0.45)
        mfu_decode: Model FLOPs utilization for decode phase (default 0.30)
        vram_util_factor: VRAM utilization factor (default 0.90)

    Returns:
        PerformanceResult with roofline analysis metrics

    Raises:
        ValueError: If GPU or precision not supported
    """
    # Initialize resource managers
    gpu_manager = GPUResourceManager()
    memory_calculator = ModelMemoryCalculator()

    # Get aggregated GPU resources
    gpu_resources = gpu_manager.get_total_resources(num_gpus, gpu_name, precision)

    # Apply VRAM utilization factor
    total_usable_vram_bytes = gpu_resources.total_memory_bytes * vram_util_factor
    total_usable_vram = total_usable_vram_bytes / (1024**3)  # Convert to GB for calculations

    # Hardware roofline threshold: ops/byte ratio
    # If workload AI < this threshold → memory bound, else → compute bound
    hardware_ops_per_byte = gpu_manager.get_compute_memory_ratio(gpu_resources)

    # Calculate model memory
    model_size_bytes = memory_calculator.calculate_model_memory(model_config, precision)
    model_size_gb = model_size_bytes / (1024**3)

    # =========================================================================
    # VRAM CONSTRAINT CHECK
    # =========================================================================
    # Calculate KV cache memory for the workload
    total_seq_len_per_request = input_length + output_length
    kv_cache_memory_bytes = memory_calculator.calculate_kv_cache_memory(
        model_config,
        sequence_length=total_seq_len_per_request,
        batch_size=concurrency,
        precision=precision
    )
    kv_cache_memory_gb = kv_cache_memory_bytes / (1024**3)

    # Total memory needed
    total_memory_needed_gb = model_size_gb + kv_cache_memory_gb

    # Early return if insufficient VRAM
    if total_memory_needed_gb > total_usable_vram:
        return PerformanceResult(
            ttft_ms=float("inf"),
            itl_ms=float("inf"),
            e2e_latency_s=float("inf"),
            output_throughput_tps=0.0,
            input_throughput_tps=0.0,
            requests_per_sec=0.0,
            bottleneck_is_memory=True,
            concurrency=concurrency,
            memory_needed_gb=total_memory_needed_gb,
            usable_vram_gb=total_usable_vram,
            hardware_ops_per_byte=hardware_ops_per_byte,
        )

    # =========================================================================
    # PREFILL PHASE ANALYSIS (TTFT)
    # =========================================================================
    # Prefill processes all input tokens in parallel, similar to training
    # High arithmetic intensity due to large matrix multiplications

    # Get bytes per parameter for memory calculations
    bytes_per_param = get_precision_bytes_per_param(precision)

    # Calculate FLOPS for prefill using detailed transformer breakdown
    prefill_flops_breakdown = calculate_transformer_flops(
        model_config=model_config,
        sequence_length=input_length,
        include_attention=True,
        include_mlp=True,
    )
    prefill_flops_per_token = prefill_flops_breakdown['total']
    total_prefill_flops = prefill_flops_per_token * concurrency

    # Memory access for prefill: model weights + KV cache writes + activations
    prefill_memory_breakdown = calculate_memory_access_bytes(
        model_config=model_config,
        sequence_length=input_length,
        batch_size=concurrency,
        bytes_per_param=bytes_per_param,
        include_kv_cache=True,  # Writing to KV cache during prefill
    )
    prefill_memory_bytes = prefill_memory_breakdown['total']

    # Prefill arithmetic intensity and bound determination
    prefill_arithmetic_intensity = calculate_arithmetic_intensity(
        total_prefill_flops, prefill_memory_bytes
    )
    prefill_is_memory_bound = determine_performance_bound(
        prefill_arithmetic_intensity, hardware_ops_per_byte
    )

    # Prefill performance calculation
    if prefill_is_memory_bound:
        # Memory bandwidth limited
        ttft_s = prefill_memory_bytes / gpu_resources.total_bandwidth_bytes_per_sec
    else:
        # Compute limited
        effective_prefill_tflops = gpu_resources.total_tflops * mfu_prefill
        ttft_s = total_prefill_flops / (effective_prefill_tflops * 1e12)

    # =========================================================================
    # DECODE PHASE ANALYSIS (ITL)
    # =========================================================================
    # Decode generates one token at a time, typically memory-bound
    # Lower arithmetic intensity due to small matrix-vector multiplications

    # Calculate FLOPS for single token generation
    # During decode: sequence_length = 1 for new token, but we access full KV cache
    decode_flops_breakdown = calculate_transformer_flops(
        model_config=model_config,
        sequence_length=1,  # Generating one token
        include_attention=True,
        include_mlp=True,
    )
    decode_flops_per_token = decode_flops_breakdown['total']

    # However, attention still needs to attend to all previous tokens
    # Add the quadratic attention terms for full context
    current_seq_len = input_length + 1  # Assuming we're generating first output token
    attention_context_flops = (
        # QK^T with full context: [1, d_head] @ [d_head, current_seq_len]
        2 * model_config.num_heads * 1 * current_seq_len * (model_config.hidden_dim // model_config.num_heads) +
        # Softmax over context
        3 * model_config.num_heads * 1 * current_seq_len +
        # Attention @ V: [1, current_seq_len] @ [current_seq_len, d_head]
        2 * model_config.num_heads * 1 * current_seq_len * (model_config.hidden_dim // model_config.num_heads)
    ) * model_config.num_layers

    # Total decode FLOPS per token (amortized across concurrency)
    total_decode_flops = (decode_flops_per_token + attention_context_flops) * concurrency

    # Memory access for decode: model weights + full KV cache read + activations
    decode_memory_breakdown = calculate_memory_access_bytes(
        model_config=model_config,
        sequence_length=current_seq_len,  # Access all previous tokens in KV cache
        batch_size=concurrency,
        bytes_per_param=bytes_per_param,
        include_kv_cache=True,
    )
    decode_memory_bytes = decode_memory_breakdown['total']

    # Decode arithmetic intensity and bound determination
    decode_arithmetic_intensity = calculate_arithmetic_intensity(
        total_decode_flops, decode_memory_bytes
    )
    decode_is_memory_bound = determine_performance_bound(
        decode_arithmetic_intensity, hardware_ops_per_byte
    )

    # Decode performance calculation
    if decode_is_memory_bound:
        # Memory bandwidth limited (typical case)
        itl_s = decode_memory_bytes / gpu_resources.total_bandwidth_bytes_per_sec
    else:
        # Compute limited (rare for decode, but possible with very high batch sizes)
        effective_decode_tflops = gpu_resources.total_tflops * mfu_decode
        itl_s = total_decode_flops / (effective_decode_tflops * 1e12)

    # =========================================================================
    # AGGREGATE PERFORMANCE METRICS
    # =========================================================================
    # Overall bottleneck determination: decode is typically the limiting factor
    overall_bottleneck_is_memory = decode_is_memory_bound

    # Throughput calculations
    output_throughput_tps = concurrency / itl_s if itl_s > 0 else float("inf")
    input_throughput_tps = (
        (input_length * concurrency) / ttft_s if ttft_s > 0 else float("inf")
    )

    # End-to-end latency and request throughput
    e2e_latency_s = ttft_s + (output_length * itl_s)
    requests_per_sec = (
        concurrency / e2e_latency_s if e2e_latency_s > 0 else float("inf")
    )

    return PerformanceResult(
        ttft_ms=ttft_s * 1000,
        itl_ms=itl_s * 1000,
        e2e_latency_s=e2e_latency_s,
        output_throughput_tps=output_throughput_tps,
        input_throughput_tps=input_throughput_tps,
        requests_per_sec=requests_per_sec,
        bottleneck_is_memory=overall_bottleneck_is_memory,
        concurrency=concurrency,
        memory_needed_gb=total_memory_needed_gb,
        usable_vram_gb=total_usable_vram,
        # Roofline analysis metrics
        prefill_arithmetic_intensity=prefill_arithmetic_intensity,
        decode_arithmetic_intensity=decode_arithmetic_intensity,
        hardware_ops_per_byte=hardware_ops_per_byte,
        prefill_is_memory_bound=prefill_is_memory_bound,
        decode_is_memory_bound=decode_is_memory_bound,
    )


def find_best_performance(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    precision: str,
    input_length: int,
    output_length: int,
    max_concurrency: int = 1024,
    **kwargs,
) -> dict[str, PerformanceResult]:
    """
    Find best latency and throughput configurations by varying concurrency.

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        precision: Precision to use
        input_length: Input sequence length
        output_length: Output sequence length
        max_concurrency: Maximum concurrency to test
        **kwargs: Additional arguments for estimate_llm_performance

    Returns:
        Dictionary with "best_latency", "best_input_throughput", "best_output_throughput"
    """
    best_latency = None
    best_input_throughput = None
    best_output_throughput = None

    # Test different concurrency levels
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    concurrency_levels = [c for c in concurrency_levels if c <= max_concurrency]

    for concurrency in concurrency_levels:
        try:
            result = estimate_llm_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                concurrency=concurrency,
                input_length=input_length,
                output_length=output_length,
                **kwargs,
            )

            # Skip if VRAM insufficient
            if result.ttft_ms == float("inf"):
                break

            # Best latency (lowest TTFT + ITL at concurrency=1)
            if best_latency is None or (
                concurrency == 1
                and result.ttft_ms + result.itl_ms
                < best_latency.ttft_ms + best_latency.itl_ms
            ):
                best_latency = result

            # Best input throughput
            if (
                best_input_throughput is None
                or result.input_throughput_tps
                > best_input_throughput.input_throughput_tps
            ):
                best_input_throughput = result

            # Best output throughput
            if (
                best_output_throughput is None
                or result.output_throughput_tps
                > best_output_throughput.output_throughput_tps
            ):
                best_output_throughput = result

        except ValueError:
            # Skip invalid configurations
            continue

    return {
        "best_latency": best_latency,
        "best_input_throughput": best_input_throughput,
        "best_output_throughput": best_output_throughput,
    }


def calculate_concurrency_limits(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    precision: str,
    input_length: int,
    output_length: int,
    mfu_prefill: float = 0.45,
    mfu_decode: float = 0.30,
    vram_util_factor: float = 0.90,
    **kwargs,
) -> dict[str, int]:
    """
    Calculate theoretical concurrency limits based on three bottlenecks:
    1. Prefill computation capacity (affects input throughput)
    2. Decode computation/memory bandwidth capacity (affects output throughput)
    3. KV cache memory constraints (hard limit)

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        precision: Precision to use
        input_length: Input sequence length
        output_length: Output sequence length
        mfu_prefill: Model FLOPs utilization for prefill
        mfu_decode: Model FLOPs utilization for decode
        vram_util_factor: VRAM utilization factor
        **kwargs: Additional arguments

    Returns:
        Dictionary with concurrency limits for each bottleneck
    """
    # Initialize resource managers
    gpu_manager = GPUResourceManager()
    memory_calculator = ModelMemoryCalculator()

    # Get aggregated GPU resources
    gpu_resources = gpu_manager.get_total_resources(num_gpus, gpu_name, precision)
    total_usable_vram_bytes = gpu_resources.total_memory_bytes * vram_util_factor
    total_usable_vram = total_usable_vram_bytes / (1024**3)  # Convert to GB for calculations

    # Calculate model memory
    model_size_bytes = memory_calculator.calculate_model_memory(model_config, precision)
    model_size_gb = model_size_bytes / (1024**3)

    # Calculate KV cache per request
    total_seq_len = input_length + output_length
    kv_cache_per_request_bytes = memory_calculator.calculate_kv_cache_memory(
        model_config,
        sequence_length=total_seq_len,
        batch_size=1,  # Per request
        precision=precision
    )
    kv_cache_per_request_gb = kv_cache_per_request_bytes / (1024**3)

    # 1. KV Cache Memory Limit (Hard constraint)
    available_kv_memory = total_usable_vram - model_size_gb
    max_concurrency_kv = int(available_kv_memory / kv_cache_per_request_gb) if kv_cache_per_request_gb > 0 else 1024
    max_concurrency_kv = max(1, max_concurrency_kv)  # At least 1

    # 2. Prefill Computation Limit (for input throughput)
    # Calculate how many requests can be processed simultaneously based on compute capacity
    # Assume we can pipeline prefill operations - multiple requests can share GPU time
    prefill_flops_per_request = 2 * model_config.num_params * input_length
    effective_prefill_flops = gpu_resources.total_tflops * 1e12 * mfu_prefill

    # Assuming we want to maintain reasonable TTFT (<500ms) under load
    target_prefill_time_s = 0.5  # 500ms target per request
    # But we can handle multiple requests in parallel/pipeline
    max_concurrent_prefills = int(effective_prefill_flops * target_prefill_time_s / prefill_flops_per_request)
    max_concurrent_prefills = max(1, min(max_concurrent_prefills, 512))  # Cap at reasonable limit

    # 3. Decode Computation/Memory Bandwidth Limit (for output throughput)
    # This is about sustained token generation capacity across all concurrent requests
    decode_flops_per_token = 2 * model_config.num_params
    effective_decode_flops = gpu_resources.total_tflops * 1e12 * mfu_decode

    # Compute limit: How many tokens can we generate per second?
    decode_tokens_per_sec_compute = effective_decode_flops / decode_flops_per_token

    # Memory bandwidth limit: How many tokens can we generate per second?
    from llm_optimizer.common import get_precision_bytes_per_param
    bytes_per_param = get_precision_bytes_per_param(precision)
    decode_tokens_per_sec_memory = gpu_resources.total_bandwidth_bytes_per_sec / (bytes_per_param * model_config.num_params)

    # Take the bottleneck (minimum)
    decode_tokens_per_sec = min(decode_tokens_per_sec_compute, decode_tokens_per_sec_memory)

    # For reasonable ITL (<50ms per token), we can serve this many concurrent requests
    target_itl_s = 0.05  # 50ms target per token
    # Each request needs 1 token per ITL period, so max concurrent = total_tokens_per_sec * ITL_period
    max_concurrent_decodes = int(decode_tokens_per_sec * target_itl_s)
    max_concurrent_decodes = max(1, min(max_concurrent_decodes, 1024))  # Cap at reasonable limit

    return {
        "kv_cache_limit": max_concurrency_kv,
        "prefill_compute_limit": max_concurrent_prefills,
        "decode_capacity_limit": max_concurrent_decodes,
        "overall_limit": min(max_concurrency_kv, max_concurrent_prefills, max_concurrent_decodes),
    }


def find_optimal_concurrency_threshold(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    precision: str,
    input_length: int,
    output_length: int,
    throughput_improvement_threshold: float = 0.05,
    **kwargs,
) -> int:
    """
    Find the optimal concurrency level considering both theoretical limits and empirical performance.

    This function:
    1. Calculates theoretical concurrency limits based on three bottlenecks
    2. Tests empirical performance up to those limits
    3. Returns the concurrency that maximizes throughput while staying within constraints

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        precision: Precision to use
        input_length: Input sequence length
        output_length: Output sequence length
        throughput_improvement_threshold: Minimum improvement to continue (default 5%)
        **kwargs: Additional arguments for estimate_llm_performance

    Returns:
        Optimal concurrency threshold
    """
    # First, calculate theoretical limits
    limits = calculate_concurrency_limits(
        num_gpus=num_gpus,
        gpu_name=gpu_name,
        model_config=model_config,
        precision=precision,
        input_length=input_length,
        output_length=output_length,
        **kwargs,
    )

    # Don't exceed the overall theoretical limit
    max_practical_concurrency = min(limits["overall_limit"], 1024)

    # Test concurrency levels up to the theoretical limit
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    concurrency_levels = [c for c in concurrency_levels if c <= max_practical_concurrency]

    # Always test at least up to 16 even if theoretical limits suggest lower
    if max_practical_concurrency < 16:
        concurrency_levels.extend([c for c in [2, 4, 8, 16] if c not in concurrency_levels])
        concurrency_levels = sorted(set(concurrency_levels))

    prev_throughput = 0
    optimal_concurrency = 1
    best_throughput = 0

    for concurrency in concurrency_levels:
        try:
            result = estimate_llm_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                concurrency=concurrency,
                input_length=input_length,
                output_length=output_length,
                **kwargs,
            )

            # Skip if VRAM insufficient
            if result.ttft_ms == float("inf"):
                break

            # Use output throughput as the primary metric for optimization
            # Output throughput scales with concurrency and is the key metric for serving performance
            current_throughput = result.output_throughput_tps

            # Track the best overall throughput
            if current_throughput > best_throughput:
                best_throughput = current_throughput
                optimal_concurrency = concurrency

            # Check for diminishing returns only after we have some baseline
            if prev_throughput > 0:
                improvement = (current_throughput - prev_throughput) / prev_throughput
                # If improvement is minimal and we're past concurrency 8, consider stopping
                if improvement < throughput_improvement_threshold and concurrency >= 8:
                    break

            prev_throughput = current_throughput

        except ValueError:
            break

    return optimal_concurrency


def parse_slo_constraints(constraints_str: str) -> list[SLOConstraint]:
    """
    Parse SLO constraint string into list of constraint objects with statistical metric support.

    Examples:
        "ttft<300ms" -> [SLOConstraint(metric="ttft", stat_type="mean", operator="<", value=300, unit="ms")]
        "ttft:median<300ms" -> [SLOConstraint(metric="ttft", stat_type="median", operator="<", value=300, unit="ms")]
        "itl:p95<8.5ms;ttft:p99<0.5s" -> [SLOConstraint(...), SLOConstraint(...)]

    Supported statistical types: mean (default), median, p95, p99

    Args:
        constraints_str: Constraint string

    Returns:
        List of SLOConstraint objects

    Raises:
        ValueError: If constraint format is invalid
    """
    if not constraints_str.strip():
        return []

    constraints = []
    constraint_parts = constraints_str.split(";")

    # Valid statistical types
    valid_stat_types = ["mean", "median", "p95", "p99"]

    for part in constraint_parts:
        part = part.strip()
        if not part:
            continue

        # Parse operator
        operators = ["<=", ">=", "<", ">", "==", "!="]
        operator = None
        for op in operators:
            if op in part:
                operator = op
                break

        if not operator:
            raise ValueError(f"No valid operator found in constraint: {part}")

        # Split by operator
        metric_part, value_part = part.split(operator, 1)
        metric_part = metric_part.strip()

        # Parse metric and statistical type (e.g., "ttft:median" or just "ttft")
        if ":" in metric_part:
            metric, stat_type = metric_part.split(":", 1)
            metric = metric.strip()
            stat_type = stat_type.strip().lower()

            if stat_type not in valid_stat_types:
                raise ValueError(f"Invalid statistical type '{stat_type}'. Valid types: {valid_stat_types}")
        else:
            metric = metric_part
            stat_type = "mean"  # default

        # Parse value and unit
        value_part = value_part.strip()

        # Extract unit (ms, s)
        unit = "ms"  # default
        if value_part.endswith("ms"):
            unit = "ms"
            value_str = value_part[:-2]
        elif value_part.endswith("s"):
            unit = "s"
            value_str = value_part[:-1]
        else:
            value_str = value_part

        try:
            value = float(value_str)
        except ValueError:
            raise ValueError(f"Invalid value in constraint: {value_part}")

        constraints.append(
            SLOConstraint(metric=metric, stat_type=stat_type, operator=operator, value=value, unit=unit)
        )

    return constraints




def get_stat_type_adjustment_factor(stat_type: str) -> float:
    """
    Get adjustment factor for different statistical types in theoretical estimation.

    These adjustment factors are heuristic estimates based on typical patterns
    observed in distributed systems latency distributions:
    - mean: No adjustment (baseline)
    - median: Slightly more conservative than mean for skewed distributions
    - p95: Higher adjustment to account for tail latencies
    - p99: Highest adjustment for extreme tail latencies

    Note: These factors are empirical approximations. For production SLOs,
    actual benchmarking and measurement are recommended to validate constraints.

    Args:
        stat_type: Statistical metric type

    Returns:
        Adjustment factor to apply to theoretical metric values
    """
    adjustment_factors = {
        "mean": 1.0,     # Baseline
        "median": 1.1,   # Slightly more conservative than mean
        "p95": 1.3,      # 30% higher for 95th percentile
        "p99": 1.6,      # 60% higher for 99th percentile
    }

    return adjustment_factors.get(stat_type, 1.0)


def get_parameter_conservativeness_for_stat_type(stat_type: str) -> float:
    """
    Get parameter conservativeness factor for different statistical requirements.

    Higher percentiles require more conservative server parameters:
    - Reduce concurrency for better tail latencies
    - Use smaller batch sizes for consistency
    - Lower memory utilization for stability

    Args:
        stat_type: Statistical metric type

    Returns:
        Conservativeness factor (0.0 to 1.0, where 1.0 is most conservative)
    """
    conservativeness = {
        "mean": 0.3,     # Aggressive settings for throughput
        "median": 0.5,   # Balanced settings
        "p95": 0.7,      # Conservative for good tail latency
        "p99": 0.9,      # Very conservative for excellent tail latency
    }

    return conservativeness.get(stat_type, 0.5)


def estimate_performance_under_constraints(
    num_gpus: int,
    gpu_name: str,
    model_config: ModelConfig,
    precision: str,
    input_length: int,
    output_length: int,
    constraints: list[SLOConstraint],
    max_concurrency: int = 1024,
    **kwargs,
) -> t.Optional[PerformanceResult]:
    """
    Find the best performance configuration that satisfies SLO constraints.

    Args:
        num_gpus: Number of GPUs
        gpu_name: GPU model name
        model_config: Model configuration
        precision: Precision to use
        input_length: Input sequence length
        output_length: Output sequence length
        constraints: List of SLO constraints
        max_concurrency: Maximum concurrency to test
        **kwargs: Additional arguments for estimate_llm_performance

    Returns:
        Best performance result that satisfies constraints, or None if impossible
    """

    def _satisfies_constraints(
        result: PerformanceResult, constraints: list[SLOConstraint]
    ) -> bool:
        """Check if result satisfies all constraints."""
        for constraint in constraints:
            # Get metric value
            if constraint.metric == "ttft":
                metric_value = (
                    result.ttft_ms if constraint.unit == "ms" else result.ttft_ms / 1000
                )
            elif constraint.metric == "itl":
                metric_value = (
                    result.itl_ms if constraint.unit == "ms" else result.itl_ms / 1000
                )
            elif constraint.metric == "e2e_latency":
                metric_value = (
                    result.e2e_latency_s * 1000
                    if constraint.unit == "ms"
                    else result.e2e_latency_s
                )
            else:
                continue  # Skip unknown metrics

            # Apply statistical type adjustment factor for theoretical estimation
            # Note: For true percentile evaluation, multiple samples would be needed
            stat_adjustment_factor = get_stat_type_adjustment_factor(constraint.stat_type)
            metric_value *= stat_adjustment_factor

            # Check constraint
            if constraint.operator == "<" and not (metric_value < constraint.value):
                return False
            elif constraint.operator == "<=" and not (metric_value <= constraint.value):
                return False
            elif constraint.operator == ">" and not (metric_value > constraint.value):
                return False
            elif constraint.operator == ">=" and not (metric_value >= constraint.value):
                return False
            elif constraint.operator == "==" and not (
                abs(metric_value - constraint.value) < 1e-6
            ):
                return False
            elif constraint.operator == "!=" and not (
                abs(metric_value - constraint.value) >= 1e-6
            ):
                return False

        return True

    best_result = None
    concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    concurrency_levels = [c for c in concurrency_levels if c <= max_concurrency]

    for concurrency in concurrency_levels:
        try:
            result = estimate_llm_performance(
                num_gpus=num_gpus,
                gpu_name=gpu_name,
                model_config=model_config,
                precision=precision,
                concurrency=concurrency,
                input_length=input_length,
                output_length=output_length,
                **kwargs,
            )

            # Skip if VRAM insufficient
            if result.ttft_ms == float("inf"):
                break

            # Check if constraints are satisfied
            if _satisfies_constraints(result, constraints):
                # Choose best throughput among valid results
                if (
                    best_result is None
                    or result.output_throughput_tps > best_result.output_throughput_tps
                ):
                    best_result = result

        except ValueError:
            continue

    return best_result


@dataclass
class PerformanceEstimationParams:
    """Parameters for performance estimation."""
    model: str
    input_len: int
    output_len: int
    gpu: str
    num_gpus: int
    precision: Optional[str] = None
    framework: str = "both"
    constraints: Optional[str] = None
    target: str = "throughput"
    dataset: str = "random"


@dataclass
class PerformanceEstimationResult:
    """Results from performance estimation."""
    model_config: t.Any
    best_configs: dict
    concurrency_limits: dict
    optimal_concurrency: int
    constrained_result: Optional[t.Any] = None
    tuning_commands: Optional[dict] = None


def run_performance_estimation(params: PerformanceEstimationParams) -> tuple[PerformanceEstimationParams, PerformanceEstimationResult]:
    """
    Run performance estimation with given parameters.

    This function contains the core estimation logic used by both interactive
    and non-interactive modes to ensure consistent behavior.

    Args:
        params: Performance estimation parameters

    Returns:
        Tuple of (updated parameters with inferred precision, PerformanceEstimationResult containing all computed results)

    Raises:
        ValueError: If constraints cannot be parsed or satisfied
        Exception: If model config cannot be loaded
    """
    # Load model configuration and infer precision if not explicitly set
    model_config = get_model_config_and_precision_from_hf(params.model)

    # Use inferred precision if precision was not explicitly provided by user
    if params.precision is None:  # Not specified by user
        precision = model_config.inferred_precision
        click.echo(f"💡 Inferred precision from model config: {precision}")
    else:
        precision = params.precision
        click.echo(f"🔧 Using user-specified precision: {precision}")

    # Create updated params with correct precision
    updated_params = PerformanceEstimationParams(
        model=params.model,
        input_len=params.input_len,
        output_len=params.output_len,
        gpu=params.gpu,
        num_gpus=params.num_gpus,
        precision=precision,
        framework=params.framework,
        constraints=params.constraints,
        target=params.target,
        dataset=params.dataset,
    )

    # Parse constraints if provided
    parsed_constraints = []
    if updated_params.constraints:
        parsed_constraints = parse_slo_constraints(updated_params.constraints)

    # Find best performance configurations
    best_configs = find_best_performance(
        num_gpus=updated_params.num_gpus,
        gpu_name=updated_params.gpu,
        model_config=model_config,
        precision=updated_params.precision,
        input_length=updated_params.input_len,
        output_length=updated_params.output_len,
    )

    # Calculate theoretical concurrency limits
    concurrency_limits = calculate_concurrency_limits(
        num_gpus=updated_params.num_gpus,
        gpu_name=updated_params.gpu,
        model_config=model_config,
        precision=updated_params.precision,
        input_length=updated_params.input_len,
        output_length=updated_params.output_len,
    )

    # Find optimal concurrency
    optimal_concurrency = find_optimal_concurrency_threshold(
        num_gpus=updated_params.num_gpus,
        gpu_name=updated_params.gpu,
        model_config=model_config,
        precision=updated_params.precision,
        input_length=updated_params.input_len,
        output_length=updated_params.output_len,
    )

    # Check constraints if provided
    constrained_result = None
    if parsed_constraints:
        constrained_result = estimate_performance_under_constraints(
            num_gpus=updated_params.num_gpus,
            gpu_name=updated_params.gpu,
            model_config=model_config,
            precision=updated_params.precision,
            input_length=updated_params.input_len,
            output_length=updated_params.output_len,
            constraints=parsed_constraints,
        )

        if not constrained_result:
            raise ValueError("Cannot satisfy the given constraints with this configuration")

    # Generate tuning configurations
    from llm_optimizer.tuning import (
        generate_advanced_tuning_configs,
        generate_llm_optimizer_commands,
        generate_simple_tuning_configs,
        generate_tuning_configs,
    )

    # Use constrained result if available, otherwise best throughput
    reference_concurrency = optimal_concurrency
    if constrained_result:
        reference_concurrency = constrained_result.concurrency
    elif updated_params.target == "latency" and best_configs["best_latency"]:
        reference_concurrency = best_configs["best_latency"].concurrency
    elif best_configs["best_output_throughput"]:
        reference_concurrency = best_configs["best_output_throughput"].concurrency

    target_throughput = updated_params.target == "throughput"
    frameworks_to_test = (
        ["sglang", "vllm"] if updated_params.framework == "both" else [updated_params.framework]
    )

    tuning_commands = {
        "simple": {},
        "advanced": {}
    }

    for fw in frameworks_to_test:
        # Use two-stage tuning approach for throughput optimization
        if parsed_constraints or (target_throughput and updated_params.target == "throughput"):
            # Stage 1: Simple tuning (concurrency + TP/DP only)
            simple_configs = generate_simple_tuning_configs(
                framework=fw,
                num_gpus=updated_params.num_gpus,
                gpu_name=updated_params.gpu,
                model_config=model_config,
                optimal_concurrency=reference_concurrency,
                precision=updated_params.precision,
                sequence_length=updated_params.input_len,
            )

            simple_commands = generate_llm_optimizer_commands(
                configs=simple_configs,
                model_id=updated_params.model,
                input_length=updated_params.input_len,
                output_length=updated_params.output_len,
                num_gpus=updated_params.num_gpus,
                constraints=updated_params.constraints,
                dataset=updated_params.dataset,
            )

            tuning_commands["simple"][fw] = {
                "configs": simple_configs,
                "commands": simple_commands
            }

            # Stage 2: Advanced tuning (additional server parameters)
            advanced_configs = generate_advanced_tuning_configs(
                framework=fw,
                num_gpus=updated_params.num_gpus,
                gpu_name=updated_params.gpu,
                model_config=model_config,
                optimal_concurrency=reference_concurrency,
                precision=updated_params.precision,
                sequence_length=updated_params.input_len,
            )

            advanced_commands = generate_llm_optimizer_commands(
                configs=advanced_configs,
                model_id=updated_params.model,
                input_length=updated_params.input_len,
                output_length=updated_params.output_len,
                num_gpus=updated_params.num_gpus,
                constraints=updated_params.constraints,
                dataset=updated_params.dataset,
            )

            tuning_commands["advanced"][fw] = {
                "configs": advanced_configs,
                "commands": advanced_commands
            }
        else:
            # For latency optimization, use traditional approach with multiple configs
            tuning_configs = generate_tuning_configs(
                framework=fw,
                num_gpus=updated_params.num_gpus,
                gpu_name=updated_params.gpu,
                model_config=model_config,
                optimal_concurrency=reference_concurrency,
                target_throughput=target_throughput,
                precision=updated_params.precision,
                sequence_length=updated_params.input_len,
            )

            commands = generate_llm_optimizer_commands(
                configs=tuning_configs,
                model_id=updated_params.model,
                input_length=updated_params.input_len,
                output_length=updated_params.output_len,
                num_gpus=updated_params.num_gpus,
                constraints=updated_params.constraints,
                dataset=updated_params.dataset,
            )

            # For latency optimization, put everything in "simple" to maintain compatibility
            tuning_commands["simple"][fw] = {
                "configs": tuning_configs,
                "commands": commands
            }

    result = PerformanceEstimationResult(
        model_config=model_config,
        best_configs=best_configs,
        concurrency_limits=concurrency_limits,
        optimal_concurrency=optimal_concurrency,
        constrained_result=constrained_result,
        tuning_commands=tuning_commands
    )

    return updated_params, result


def display_performance_estimation_results(params: PerformanceEstimationParams, result: PerformanceEstimationResult):
    """Display performance estimation results in a consistent format."""

    click.echo("\n=== Configuration ===")
    click.echo(f"Model: {params.model}")
    click.echo(f"GPU: {params.num_gpus}x {params.gpu}")
    click.echo(f"Precision: {params.precision}")
    click.echo(f"Input/Output: {params.input_len}/{params.output_len} tokens")
    click.echo(f"Target: {params.target}")
    if params.constraints:
        click.echo(f"Constraints: {params.constraints}")

    # Model info
    click.echo("\nFetching model configuration...")
    click.echo(
        f"Model: {result.model_config.num_params:.1f}B parameters, {result.model_config.num_layers} layers"
    )

    # Parse constraints if provided
    parsed_constraints = []
    if params.constraints:
        try:
            parsed_constraints = parse_slo_constraints(params.constraints)
            click.echo(f"Parsed {len(parsed_constraints)} constraint(s)")
        except ValueError as e:
            click.echo(f"Error parsing constraints: {e}")
            return

    # Performance Analysis
    click.echo("\n=== Performance Analysis ===")
    if result.best_configs["best_latency"]:
        latency_config = result.best_configs["best_latency"]
        click.echo(f"Best Latency (concurrency={latency_config.concurrency}):")
        click.echo(f"  TTFT: {latency_config.ttft_ms:.1f} ms")
        click.echo(f"  ITL: {latency_config.itl_ms:.1f} ms")
        click.echo(f"  E2E: {latency_config.e2e_latency_s:.2f} s")

    if result.best_configs["best_output_throughput"]:
        throughput_config = result.best_configs["best_output_throughput"]
        click.echo(
            f"\nBest Throughput (concurrency={throughput_config.concurrency}):"
        )
        click.echo(
            f"  Output: {throughput_config.output_throughput_tps:.1f} tokens/s"
        )
        click.echo(
            f"  Input: {throughput_config.input_throughput_tps:.1f} tokens/s"
        )
        click.echo(f"  Requests: {throughput_config.requests_per_sec:.2f} req/s")
        click.echo(
            f"  Bottleneck: {'Memory' if throughput_config.bottleneck_is_memory else 'Compute'}"
        )

    # Roofline Analysis
    click.echo("\n=== Roofline Analysis ===")
    if result.best_configs["best_output_throughput"]:
        config = result.best_configs["best_output_throughput"]
        click.echo(f"Hardware Ops/Byte Ratio: {config.hardware_ops_per_byte:.1f} ops/byte")
        click.echo(f"Prefill Arithmetic Intensity: {config.prefill_arithmetic_intensity:.1f} ops/byte")
        click.echo(f"Decode Arithmetic Intensity: {config.decode_arithmetic_intensity:.1f} ops/byte")
        click.echo(f"Prefill Phase: {'Memory Bound' if config.prefill_is_memory_bound else 'Compute Bound'}")
        click.echo(f"Decode Phase: {'Memory Bound' if config.decode_is_memory_bound else 'Compute Bound'}")

    # Concurrency Analysis
    click.echo("\n=== Concurrency Analysis ===")
    click.echo(f"KV Cache Memory Limit: {result.concurrency_limits['kv_cache_limit']} concurrent requests")
    click.echo(f"Prefill Compute Limit: {result.concurrency_limits['prefill_compute_limit']} concurrent requests")
    click.echo(f"Decode Capacity Limit: {result.concurrency_limits['decode_capacity_limit']} concurrent requests")
    click.echo(f"Theoretical Overall Limit: {result.concurrency_limits['overall_limit']} concurrent requests")
    click.echo(f"Empirical Optimal Concurrency: {result.optimal_concurrency} concurrent requests")

    # Constrained Performance
    if result.constrained_result:
        click.echo("\n=== Performance under Constraints ===")
        click.echo(f"Concurrency: {result.constrained_result.concurrency}")
        click.echo(f"TTFT: {result.constrained_result.ttft_ms:.1f} ms")
        click.echo(f"ITL: {result.constrained_result.itl_ms:.1f} ms")
        click.echo(
            f"Output throughput: {result.constrained_result.output_throughput_tps:.1f} tokens/s"
        )
    elif params.constraints:
        click.echo(
            "\n❌ Cannot satisfy the given constraints with this configuration"
        )
        return

    # Tuning Commands (Two-Stage Structure)
    if result.tuning_commands:
        click.echo("\n=== Tuning Commands ===")

        # Get all frameworks that have either simple or advanced configs
        all_frameworks = set()
        if result.tuning_commands.get("simple"):
            all_frameworks.update(result.tuning_commands["simple"].keys())
        if result.tuning_commands.get("advanced"):
            all_frameworks.update(result.tuning_commands["advanced"].keys())

        # Output simple and advanced configs for each framework together
        for fw in sorted(all_frameworks):
            click.echo(f"\n--- {fw.upper()} ---")

            # Simple configs
            if result.tuning_commands.get("simple") and fw in result.tuning_commands["simple"]:
                simple_data = result.tuning_commands["simple"][fw]
                click.echo("Simple (concurrency + TP/DP):")
                for config, cmd in zip(simple_data["configs"], simple_data["commands"]):
                    click.echo(f"  {cmd}")

            # Advanced configs
            if result.tuning_commands.get("advanced") and fw in result.tuning_commands["advanced"]:
                advanced_data = result.tuning_commands["advanced"][fw]
                click.echo("Advanced (additional parameters):")
                for config, cmd in zip(advanced_data["configs"], advanced_data["commands"]):
                    click.echo(f"  {cmd}")

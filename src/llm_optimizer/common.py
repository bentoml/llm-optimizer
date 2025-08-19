"""
Common functionality shared between performance estimation and tuning configuration modules.

This module contains shared calculations, utilities, and data structures to avoid
code duplication and ensure consistency across the codebase.
"""

import json
from dataclasses import dataclass
from typing import Optional

from huggingface_hub import hf_hub_download


@dataclass
class ModelConfig:
    """Model configuration data structure."""
    num_params: int  # actual number of parameters
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: Optional[int] = None
    vocab_size: int = 32000
    inferred_precision: str = "fp16"  # Inferred model precision

    def __post_init__(self):
        # Default num_kv_heads to num_heads if not specified
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads


def get_precision_bytes_per_param(precision: str) -> int:
    """
    Get bytes per parameter for different precisions.

    Args:
        precision: Model precision ("fp16" or "fp8")

    Returns:
        Number of bytes per parameter

    Raises:
        ValueError: If precision is not supported
    """
    precision_map = {
        "fp16": 2,
        "bf16": 2,  # bf16 uses same memory as fp16
        "fp8": 1,
    }

    if precision not in precision_map:
        raise ValueError(f"Unsupported precision: {precision}. Use {list(precision_map.keys())}")

    return precision_map[precision]


def calculate_model_memory_bytes(
    model_config: ModelConfig,
    precision: str,
    safety_factor: float = 1.0
) -> int:
    """
    Calculate model memory usage in bytes.

    Args:
        model_config: Model configuration
        precision: Model precision
        safety_factor: Safety multiplier for overhead

    Returns:
        Model memory usage in bytes
    """

    from llm_optimizer.resources import ModelMemoryCalculator
    memory_calculator = ModelMemoryCalculator()
    model_memory_bytes = memory_calculator.calculate_model_memory(model_config, precision)
    return int(model_memory_bytes * safety_factor)


def calculate_min_tensor_parallel_size(
    model_config: ModelConfig,
    gpu_specs: dict,
    precision: str,
    safety_factor: float = 1.2
) -> int:
    """
    Calculate minimum tensor parallel size needed to fit model.

    Args:
        model_config: Model configuration
        gpu_specs: GPU specifications dictionary
        precision: Model precision
        safety_factor: Safety factor for memory calculations

    Returns:
        Minimum tensor parallel size
    """
    model_memory_bytes = calculate_model_memory_bytes(model_config, precision, safety_factor)
    single_gpu_vram_bytes = int(gpu_specs["VRAM_GB"] * 1024**3)

    min_tp_size = max(1, int(model_memory_bytes // single_gpu_vram_bytes) + 1)
    return min_tp_size


def generate_parameter_range(
    optimal_value: int,
    num_values: int = 3,
    variation_factor: float = 0.5,
    min_val: int = 1,
    max_val: Optional[int] = None
) -> list[int]:
    """
    Generate a range of parameter values around an optimal value.

    Args:
        optimal_value: Base value to generate range around
        num_values: Number of values to generate
        variation_factor: How much to vary (as fraction of optimal_value)
        min_val: Minimum allowed value
        max_val: Maximum allowed value (optional)

    Returns:
        List of parameter values
    """
    if num_values == 1:
        return [optimal_value]

    if optimal_value <= 4:
        # For small values, use simple ±1 variation
        variation = 1
    else:
        # For larger values, use percentage-based variation
        variation = max(1, int(optimal_value * variation_factor))

    # Generate symmetric range around optimal
    half_range = (num_values - 1) // 2
    values = []

    for i in range(-half_range, half_range + 1):
        value = optimal_value + i * variation
        value = max(min_val, value)
        if max_val:
            value = min(max_val, value)
        values.append(value)

    # Ensure we have the requested number of unique values
    values = sorted(set(values))

    # If we don't have enough values, add more around the optimal
    while len(values) < num_values and len(values) < 10:  # Prevent infinite loop
        # Add values between existing ones
        new_values = []
        for i in range(len(values) - 1):
            mid = (values[i] + values[i + 1]) // 2
            if mid not in values and mid != values[i] and mid != values[i + 1]:
                new_values.append(mid)

        if new_values:
            values.extend(new_values)
            values = sorted(set(values))
        else:
            break

    return values[:num_values]


def generate_tp_dp_combinations(num_gpus: int, min_tp_size: int = 1) -> list[tuple[int, int]]:
    """
    Generate tensor parallel (TP) and data parallel (DP) combinations.

    Args:
        num_gpus: Total number of GPUs
        min_tp_size: Minimum tensor parallel size needed

    Returns:
        List of (TP, DP) tuples where TP * DP = num_gpus
    """
    combinations = []

    # Generate all valid combinations
    for tp_size in range(min_tp_size, num_gpus + 1):
        if num_gpus % tp_size == 0:  # Ensure even division
            dp_size = num_gpus // tp_size
            combinations.append((tp_size, dp_size))

    # If no valid combinations, use all GPUs for TP
    if not combinations:
        combinations = [(num_gpus, 1)]

    return combinations


def calculate_model_parameters_from_config(config: dict) -> int:
    """
    Calculate the total number of parameters from model config.

    Args:
        config: HuggingFace model config dictionary

    Returns:
        Total number of parameters (int)
    """
    try:
        # Extract parameters needed for calculation
        h = config["hidden_size"]
        n_layers = config["num_hidden_layers"]
        i = config["intermediate_size"]
        v = config["vocab_size"]
        n_heads = config.get("num_attention_heads", 0)
        n_kv_heads = config.get("num_key_value_heads", n_heads)

        # Calculate params per layer
        head_dim = h // n_heads if n_heads > 0 else 0
        attention_params = n_layers * (
            h * (n_heads * head_dim) + h * (n_kv_heads * head_dim) * 2 + h * h
        )

        # FFN params (assuming SwiGLU)
        ffn_params = n_layers * (h * i * 2 + i * h)

        # Embedding and output params
        embedding_params = v * h
        output_params = v * h if not config.get("tie_word_embeddings", False) else 0

        total_params = attention_params + ffn_params + embedding_params + output_params
        return total_params

    except KeyError:
        # If we can't calculate from config, return 0
        return 0


def get_safetensor_total_size(model_id: str) -> Optional[int]:
    """
    Calculate total size of all safetensor files for a model.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        Total size in bytes, or None if cannot determine
    """
    try:
        from huggingface_hub import list_repo_files

        # Get all files in the repo
        files = list_repo_files(repo_id=model_id)

        # Find all safetensor files
        safetensor_files = [f for f in files if f.endswith('.safetensors')]

        if not safetensor_files:
            return None

        total_size = 0

        for file_path in safetensor_files:
            try:
                # Download just the file info to get size
                from huggingface_hub import get_hf_file_metadata, hf_hub_url
                url = hf_hub_url(repo_id=model_id, filename=file_path)
                metadata = get_hf_file_metadata(url)

                if hasattr(metadata, 'size') and metadata.size:
                    total_size += metadata.size
                else:
                    # Fallback: try to get size from repo info
                    from huggingface_hub import repo_info
                    info = repo_info(repo_id=model_id)
                    if info.siblings:
                        for sibling in info.siblings:
                            if sibling.rfilename == file_path and sibling.size:
                                total_size += sibling.size
                                break

            except Exception:
                # Skip files we can't get size for
                continue

        return total_size if total_size > 0 else None

    except Exception:
        # If any step fails, return None
        return None


def infer_precision_from_model_size(config: dict, model_id: str) -> Optional[str]:
    """
    Infer precision from parameter count and total model size.

    This calculates bytes per parameter and estimates precision:
    - ~1 bytes/param: fp8
    - ~2 bytes/param: fp16/bf16
    - ~4 bytes/param: fp32

    Args:
        config: HuggingFace model config dictionary
        model_id: Model identifier for downloading size info

    Returns:
        Inferred precision string or None if cannot determine
    """
    if not model_id:
        return None

    # Calculate parameter count from config
    total_params = calculate_model_parameters_from_config(config)
    if total_params == 0:
        return None

    # Get total safetensor size
    total_size_bytes = get_safetensor_total_size(model_id)
    if total_size_bytes is None:
        return None

    # Calculate bytes per parameter
    bytes_per_param = total_size_bytes / total_params

    # Infer precision based on bytes per parameter
    # Add some tolerance for overhead, compression, etc.
    if bytes_per_param <= 1.3:  # ~1 byte per param + tolerance
        return "fp8"
    elif bytes_per_param <= 2.5:  # ~2 bytes per param + tolerance
        # Default to bf16 for modern models, fp16 for older ones
        return "bf16"
    elif bytes_per_param <= 4.5:  # ~4 bytes per param
        return "fp32"  # Though we don't typically use this
    else:
        # Unexpectedly large, might be fp64 or have significant overhead
        return None


def infer_precision_from_config(config: dict, model_id: str = None) -> str:
    """
    Infer model precision from HuggingFace config and model ID.

    Args:
        config: HuggingFace model config dictionary
        model_id: Original model ID/path (used for name-based inference)

    Returns:
        str: Inferred precision ("fp16", "bf16", or "fp8")
    """
    # Check quantization_config field first (highest priority for quantized models)
    quantization_config = config.get("quantization_config")
    if quantization_config:
        # Check for FP8 quantization
        if isinstance(quantization_config, dict):
            # Look for compression method indicating FP8
            quant_method = quantization_config.get("quant_method", "").lower()
            format_name = quantization_config.get("format", "").lower()

            # Common FP8 quantization indicators
            fp8_indicators = [
                "compressed-tensors",
                "fp8",
                "float8",
                "e4m3", "e5m2",  # FP8 formats
                "fbgemm_fp8",
                "float-quantized"
            ]

            if any(indicator in quant_method or indicator in format_name for indicator in fp8_indicators):
                return "fp8"

            # Check for bit configuration indicating FP8
            bits = quantization_config.get("bits")
            weight_bits = quantization_config.get("weight_bits")
            activation_bits = quantization_config.get("activation_bits")

            # 8-bit weights + 8-bit activations often indicates FP8
            if bits == 8 or (weight_bits == 8 and activation_bits == 8):
                return "fp8"

            # Check config groups for bit specifications
            config_groups = quantization_config.get("config_groups", {})
            if isinstance(config_groups, dict):
                for group_config in config_groups.values():
                    if isinstance(group_config, dict):
                        input_acts = group_config.get("input_activations", {})
                        weights = group_config.get("weights", {})

                        # Check if both weights and activations use 8-bit
                        if (isinstance(input_acts, dict) and input_acts.get("num_bits") == 8 and
                            isinstance(weights, dict) and weights.get("num_bits") == 8):
                            return "fp8"

    # Check model ID/name for precision hints (high priority)
    # This checks the original model ID passed to the function
    if model_id:
        model_id_lower = model_id.lower()
        if "fp8" in model_id_lower:
            return "fp8"
        elif "bf16" in model_id_lower or "bfloat16" in model_id_lower:
            return "bf16"
        elif "fp16" in model_id_lower or "float16" in model_id_lower:
            return "fp16"

    # Check torch_dtype field
    torch_dtype = config.get("torch_dtype")
    if torch_dtype:
        # Map torch dtypes to our precision names
        dtype_mapping = {
            "float16": "fp16",
            "bfloat16": "bf16",
            "torch.float16": "fp16",
            "torch.bfloat16": "bf16",
            "fp8": "fp8"
        }
        if torch_dtype in dtype_mapping:
            return dtype_mapping[torch_dtype]

    # Check model name/repo for precision hints (from config)
    model_name = config.get("_name_or_path", "").lower()
    if "fp8" in model_name:
        return "fp8"
    elif "bf16" in model_name or "bfloat16" in model_name:
        return "bf16"
    elif "fp16" in model_name or "float16" in model_name:
        return "fp16"

    # Check model architecture for precision hints
    config.get("model_type", "").lower()
    config.get("architectures", [])

    # Some models specify precision in their config content
    config_str = str(config).lower()
    if "fp8" in config_str or "float8" in config_str:
        return "fp8"
    elif "bf16" in config_str or "bfloat16" in config_str:
        return "bf16"

    # Try to infer precision from parameter count and model size
    # This method calculates bytes per parameter to estimate precision
    try:
        precision_from_size = infer_precision_from_model_size(config, model_id)
        if precision_from_size:
            return precision_from_size
    except Exception:
        # If size-based inference fails, continue to fallback
        pass

    # Default to fp16 if we can't determine
    return "fp16"


def get_model_config_from_hf(model_id: str) -> ModelConfig:
    """
    Downloads a model's config.json from Hugging Face and extracts configuration.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        ModelConfig object with extracted parameters

    Raises:
        RuntimeError: If config cannot be downloaded or parsed
        KeyError: If required keys are missing from config
    """
    return get_model_config_and_precision_from_hf(model_id)


def get_model_config_and_precision_from_hf(model_id: str) -> ModelConfig:
    """
    Downloads a model's config.json from Hugging Face and extracts configuration with inferred precision.

    Args:
        model_id: HuggingFace model identifier

    Returns:
        ModelConfig object with extracted parameters and inferred precision

    Raises:
        RuntimeError: If config cannot be downloaded or parsed
        KeyError: If required keys are missing from config
    """
    try:
        config_path = hf_hub_download(repo_id=model_id, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Could not download or read config.json for {model_id}: {e}"
        )

    try:
        # Extract basic config parameters
        h = config["hidden_size"]
        n_layers = config["num_hidden_layers"]
        v = config["vocab_size"]
        n_heads = config.get("num_attention_heads", 0)
        n_kv_heads = config.get("num_key_value_heads", n_heads)

        # Calculate total parameters using the dedicated function
        total_params = calculate_model_parameters_from_config(config)

        # Infer precision from config and model ID
        precision = infer_precision_from_config(config, model_id)

        model_config = ModelConfig(
            num_params=total_params,
            num_layers=n_layers,
            hidden_dim=h,
            vocab_size=v,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            inferred_precision=precision,
        )

        return model_config

    except KeyError as e:
        raise KeyError(f"Could not find required key {e} in config.json for {model_id}")


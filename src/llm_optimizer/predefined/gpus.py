"""
GPU specifications for LLM performance estimation.
Data compiled from official NVIDIA specifications and technical documentation.
"""

# GPU specifications from official NVIDIA sources and datasheets
# Researched and verified from NVIDIA official documentation (December 2024)
GPU_SPECS = {
    # NVIDIA Hopper Architecture GPUs
    "H100": {
        "FP16_TFLOPS": 989.5,  # Official NVIDIA specification for SXM5 variant
        "FP8_TFLOPS": 1978.0,  # 4th gen Tensor Cores with FP8 support
        "Memory_Bandwidth_GBs": 3350,  # 3.35 TB/s HBM3 memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3",
    },
    "H200": {
        "FP16_TFLOPS": 989.0,  # Official NVIDIA specification
        "FP8_TFLOPS": 1978.0,  # Official NVIDIA specification
        "Memory_Bandwidth_GBs": 4800,  # 4.8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 141,
        "Architecture": "Hopper",
        "Memory_Type": "HBM3e",
    },
    # NVIDIA Ampere Architecture GPUs
    "A100": {
        "FP16_TFLOPS": 312.0,  # Official NVIDIA specification for 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 2039,  # 2.039 TB/s HBM2e memory bandwidth
        "VRAM_GB": 80,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2e",
    },
    "A100-40GB": {
        "FP16_TFLOPS": 312.0,  # Same compute as 80GB variant
        "FP8_TFLOPS": None,  # Not supported on Ampere architecture
        "Memory_Bandwidth_GBs": 1555,  # 1.555 TB/s HBM2 memory bandwidth
        "VRAM_GB": 40,
        "Architecture": "Ampere",
        "Memory_Type": "HBM2",
    },
    # NVIDIA Ada Lovelace Architecture GPUs
    "L20": {
        "FP16_TFLOPS": 119.5,  # Official NVIDIA specification
        "FP8_TFLOPS": 239.0,  # Ada Lovelace supports FP8
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    "L40": {
        "FP16_TFLOPS": 181.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 362.0,  # Ada Lovelace 4th gen Tensor Cores
        "Memory_Bandwidth_GBs": 864,  # GDDR6 memory bandwidth
        "VRAM_GB": 48,
        "Architecture": "Ada Lovelace",
        "Memory_Type": "GDDR6",
    },
    # NVIDIA Blackwell Architecture GPUs (Future)
    "B100": {
        "FP16_TFLOPS": 1800.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 3500.0,  # 5th gen Tensor Cores
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
    },
    "B200": {
        "FP16_TFLOPS": 2250.0,  # Official NVIDIA specification (dense)
        "FP8_TFLOPS": 4500.0,  # 5th gen Tensor Cores enhanced
        "Memory_Bandwidth_GBs": 8000,  # 8 TB/s HBM3e memory bandwidth
        "VRAM_GB": 192,  # 2x96GB HBM3e stacks
        "Architecture": "Blackwell",
        "Memory_Type": "HBM3e",
    },
}


def get_gpu_specs(gpu_name: str) -> dict:
    """
    Get GPU specifications by name.

    Args:
        gpu_name: Name of the GPU (e.g., "H100", "h100", "A100", "a100")

    Returns:
        Dictionary containing GPU specifications

    Raises:
        ValueError: If GPU name is not found
    """
    # Normalize to uppercase for lookup
    normalized_name = gpu_name.upper()

    if normalized_name not in GPU_SPECS:
        available = ", ".join(GPU_SPECS.keys())
        available_lower = ", ".join([name.lower() for name in GPU_SPECS.keys()])
        raise ValueError(f"GPU '{gpu_name}' not found. Available GPUs: {available} (case-insensitive: {available_lower})")

    return GPU_SPECS[normalized_name].copy()


def list_available_gpus() -> list[str]:
    """Return list of available GPU names."""
    return list(GPU_SPECS.keys())


def list_available_gpus_with_lowercase() -> list[str]:
    """Return list of available GPU names including lowercase variants."""
    gpus = list(GPU_SPECS.keys())
    # Add lowercase variants for better tab completion
    gpus.extend([name.lower() for name in GPU_SPECS.keys()])
    return sorted(gpus)


def get_precision_tflops(gpu_name: str, precision: str) -> float:
    """
    Get TFLOPS for a specific precision.

    Args:
        gpu_name: Name of the GPU (case-insensitive)
        precision: "bf16", "fp16" or "fp8"

    Returns:
        TFLOPS value for the precision

    Raises:
        ValueError: If GPU or precision not supported
    """
    specs = get_gpu_specs(gpu_name)

    if precision in ("fp16", "bf16"):
        return specs["FP16_TFLOPS"]
    elif precision == "fp8":
        if specs["FP8_TFLOPS"] is None:
            raise ValueError(f"FP8 precision not supported on {gpu_name}")
        return specs["FP8_TFLOPS"]
    else:
        raise ValueError(f"Unsupported precision: {precision}. Use 'fp16' or 'fp8'")

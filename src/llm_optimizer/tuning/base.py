"""Base classes and common parameters for framework configuration."""

from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class CommonTuningParams:
    """Common parameters across all frameworks."""
    model: str
    max_concurrent_requests: int
    tensor_parallel: int
    data_parallel: int = 1
    memory_utilization: float = 0.9
    precision: str = "fp16"
    sequence_length: int = 2048
    batch_size: Optional[int] = None
    prefill_chunk_size: Optional[int] = None

    def to_framework_params(self, framework: str) -> dict[str, Any]:
        """Convert to framework-specific parameter names."""
        from llm_optimizer.predefined import PARAMETER_MAPPINGS

        mapping = PARAMETER_MAPPINGS.get(framework.lower(), {})
        result = {}

        for key, value in asdict(self).items():
            if value is None:
                continue

            mapped_key = mapping.get(key, key)
            if mapped_key is not None:  # Only include if framework supports it
                result[mapped_key] = value

        return result

    def validate(self) -> list[str]:
        """Validate parameter consistency and return list of errors."""
        errors = []

        if self.tensor_parallel < 1:
            errors.append("tensor_parallel must be >= 1")

        if self.data_parallel < 1:
            errors.append("data_parallel must be >= 1")

        if not (0.0 < self.memory_utilization <= 1.0):
            errors.append("memory_utilization must be between 0 and 1")

        if self.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be >= 1")

        if self.precision not in ["fp16", "bf16", "fp8", "int8"]:
            errors.append(f"unsupported precision: {self.precision}")

        if self.sequence_length < 1:
            errors.append("sequence_length must be >= 1")

        if self.batch_size is not None and self.batch_size < 1:
            errors.append("batch_size must be >= 1 if specified")

        if self.prefill_chunk_size is not None and self.prefill_chunk_size < 1:
            errors.append("prefill_chunk_size must be >= 1 if specified")

        return errors

    def scale_for_gpus(self, num_gpus: int) -> 'CommonTuningParams':
        """Scale parameters appropriately for given number of GPUs."""
        if num_gpus < 1:
            raise ValueError("num_gpus must be >= 1")

        # Adjust TP/DP to match available GPUs
        total_parallel = self.tensor_parallel * self.data_parallel

        if total_parallel > num_gpus:
            # Scale down - prefer TP reduction first
            new_tp = min(self.tensor_parallel, num_gpus)
            new_dp = num_gpus // new_tp
        elif total_parallel < num_gpus:
            # Scale up - prefer DP increase
            new_dp = self.data_parallel * (num_gpus // total_parallel)
            new_tp = self.tensor_parallel

            # If we can't scale DP evenly, adjust TP
            if new_tp * new_dp != num_gpus:
                new_tp = num_gpus // new_dp
        else:
            new_tp = self.tensor_parallel
            new_dp = self.data_parallel

        # Create new instance with updated parallelism
        return CommonTuningParams(
            model=self.model,
            max_concurrent_requests=self.max_concurrent_requests,
            tensor_parallel=new_tp,
            data_parallel=new_dp,
            memory_utilization=self.memory_utilization,
            precision=self.precision,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            prefill_chunk_size=self.prefill_chunk_size
        )


@dataclass
class WorkloadSpec:
    """Specification for workload characteristics."""
    input_length: int
    output_length: int
    concurrency: int
    target_throughput: bool = True

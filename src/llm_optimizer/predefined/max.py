# =========================================================================
#  MAX Platform Server Argument Configurations
# =========================================================================
# This dictionary is generated based on MAX server documentation.
# The keys are the normalized argument names.

from llm_optimizer.args import ArgConfig, ConfigsDict

# Parameter mapping from common names to framework-specific parameter names
PARAMETER_MAPPING = {
    "tensor_parallel": "tensor_parallel_degree",
    "data_parallel": "data_parallel_degree",
    "max_concurrent_requests": "max_batch_size",
}


SERVER_CONFIGS: ConfigsDict = {
    # --- Model Configuration ---
    "model_path": ArgConfig[str](name="model_path"),
    "served_model_name": ArgConfig[str](name="served_model_name"),
    "weight_path": ArgConfig[str](name="weight_path"),
    "quantization_encoding": ArgConfig[str](name="quantization_encoding"),
    "allow_safetensors_weights_fp32_bf6_bidirectional_cast": ArgConfig[bool](
        name="allow_safetensors_weights_fp32_bf6_bidirectional_cast"
    ),
    "huggingface_model_revision": ArgConfig[str](name="huggingface_model_revision"),
    "huggingface_weight_revision": ArgConfig[str](name="huggingface_weight_revision"),
    "trust_remote_code": ArgConfig[bool](name="trust_remote_code"),
    "force_download": ArgConfig[bool](name="force_download"),
    "vision_config_overrides": ArgConfig[str](name="vision_config_overrides"),
    "rope_type": ArgConfig[str](name="rope_type"),
    "use_subgraphs": ArgConfig[bool](name="use_subgraphs"),
    
    # --- Parallelism Configuration ---
    "tensor_parallel_degree": ArgConfig[int](name="tensor_parallel_degree"),
    "pipeline_parallel_degree": ArgConfig[int](name="pipeline_parallel_degree"),
    "data_parallel_degree": ArgConfig[int](name="data_parallel_degree"),
    
    # --- Draft Model Configuration (Speculative Decoding) ---
    "draft_model_path": ArgConfig[str](name="draft_model_path"),
    "draft_weight_path": ArgConfig[str](name="draft_weight_path"),
    "draft_quantization_encoding": ArgConfig[str](name="draft_quantization_encoding"),
    "draft_huggingface_model_revision": ArgConfig[str](name="draft_huggingface_model_revision"),
    "draft_huggingface_weight_revision": ArgConfig[str](name="draft_huggingface_weight_revision"),
    "draft_trust_remote_code": ArgConfig[bool](name="draft_trust_remote_code"),
    "draft_force_download": ArgConfig[bool](name="draft_force_download"),
    "draft_vision_config_overrides": ArgConfig[str](name="draft_vision_config_overrides"),
    "draft_rope_type": ArgConfig[str](name="draft_rope_type"),
    "draft_use_subgraphs": ArgConfig[bool](name="draft_use_subgraphs"),
    "draft_tensor_parallel_degree": ArgConfig[int](name="draft_tensor_parallel_degree"),
    "draft_pipeline_parallel_degree": ArgConfig[int](name="draft_pipeline_parallel_degree"),
    "draft_data_parallel_degree": ArgConfig[int](name="draft_data_parallel_degree"),
    
    # --- Server Configuration ---
    "port": ArgConfig[int](name="port"),
    "headless": ArgConfig[bool](name="headless"),
    "profile_serve": ArgConfig[bool](name="profile_serve"),
    "sim_failure": ArgConfig[int](name="sim_failure"),
    "log_prefix": ArgConfig[str](name="log_prefix"),
    "task": ArgConfig[str](name="task"),
    "task_arg": ArgConfig[str](name="task_arg"),
    "pretty_print_config": ArgConfig[bool](name="pretty_print_config"),
    
    # --- Pipeline Configuration ---
    "max_length": ArgConfig[int](name="max_length"),
    "pipeline_role": ArgConfig[str](name="pipeline_role"),
    "max_batch_size": ArgConfig[int](name="max_batch_size"),
    "max_ce_batch_size": ArgConfig[int](name="max_ce_batch_size"),
    "max_queue_size_tg": ArgConfig[int](name="max_queue_size_tg"),
    "min_batch_size_tg": ArgConfig[int](name="min_batch_size_tg"),
    "ce_delay_ms": ArgConfig[float](name="ce_delay_ms"),
    "enable_prioritize_first_decode": ArgConfig[bool](name="enable_prioritize_first_decode"),
    "enable_chunked_prefill": ArgConfig[bool](name="enable_chunked_prefill"),
    "enable_in_flight_batching": ArgConfig[bool](name="enable_in_flight_batching"),
    "max_num_steps": ArgConfig[int](name="max_num_steps"),
    "target_num_new_tokens": ArgConfig[int](name="target_num_new_tokens"),
    "enable_echo": ArgConfig[bool](name="enable_echo"),
    "pool_embeddings": ArgConfig[bool](name="pool_embeddings"),
    "use_experimental_kernels": ArgConfig[str](name="use_experimental_kernels"),
    "custom_architectures": ArgConfig[str](name="custom_architectures"),
    
    # --- KV Cache Configuration ---
    "cache_strategy": ArgConfig[str](name="cache_strategy"),
    "kv_cache_page_size": ArgConfig[int](name="kv_cache_page_size"),
    "enable_prefix_caching": ArgConfig[bool](name="enable_prefix_caching"),
    "enable_kvcache_swapping_to_host": ArgConfig[bool](name="enable_kvcache_swapping_to_host"),
    "device_memory_utilization": ArgConfig[float](name="device_memory_utilization"),
    "host_kvcache_swap_space_gb": ArgConfig[float](name="host_kvcache_swap_space_gb"),
    
    # --- LoRA Configuration ---
    "enable_lora": ArgConfig[bool](name="enable_lora"),
    "lora_paths": ArgConfig[str](name="lora_paths"),
    "max_lora_rank": ArgConfig[int](name="max_lora_rank"),
    "max_num_loras": ArgConfig[int](name="max_num_loras"),
    "lora_request_endpoint": ArgConfig[str](name="lora_request_endpoint"),
    "lora_response_endpoint": ArgConfig[str](name="lora_response_endpoint"),
    
    # --- Profiling Configuration ---
    "gpu_profiling": ArgConfig[str](name="gpu_profiling"),
    
    # --- Sampling Configuration ---
    "enable_structured_output": ArgConfig[bool](name="enable_structured_output"),
    "enable_variable_logits": ArgConfig[bool](name="enable_variable_logits"),
    "enable_min_tokens": ArgConfig[bool](name="enable_min_tokens"),
    "do_penalties": ArgConfig[bool](name="do_penalties"),
    "in_dtype": ArgConfig[str](name="in_dtype"),
    "out_dtype": ArgConfig[str](name="out_dtype"),
    
    # --- Sampling Parameters ---
    "top_k": ArgConfig[int](name="top_k"),
    "top_p": ArgConfig[float](name="top_p"),
    "min_p": ArgConfig[float](name="min_p"),
    "temperature": ArgConfig[float](name="temperature"),
    "frequency_penalty": ArgConfig[float](name="frequency_penalty"),
    "presence_penalty": ArgConfig[float](name="presence_penalty"),
    "repetition_penalty": ArgConfig[float](name="repetition_penalty"),
    "max_new_tokens": ArgConfig[int](name="max_new_tokens"),
    "min_new_tokens": ArgConfig[int](name="min_new_tokens"),
    "ignore_eos": ArgConfig[bool](name="ignore_eos"),
    "stop": ArgConfig[str](name="stop"),
    "stop_token_ids": ArgConfig[str](name="stop_token_ids"),
    "detokenize": ArgConfig[bool](name="detokenize"),
    "no_detokenize": ArgConfig[bool](name="no_detokenize"),
    "seed": ArgConfig[int](name="seed"),
    
    # --- Device Configuration ---
    "devices": ArgConfig[str](name="devices"),
    "draft_devices": ArgConfig[str](name="draft_devices"),
    
    # --- Logging Configuration ---
    "verbose": ArgConfig[bool](name="verbose"),
    "quiet": ArgConfig[bool](name="quiet"),
    "log_level": ArgConfig[str](name="log_level"),
}

# LLM Optimizer

llm-optimizer is a Python tool for benchmarking and optimizing inference performance of any open-source LLMs.

- 🧩 Benchmark across inference frameworks like SGLang and vLLM using their native arguments
- ⚡️ Find the optimal setup automatically for your use case without endless trial and error
- 🎯 Apply SLO constraints to focus only on configurations that meet your performance goals
- 🧮 Estimate performance theoretically without running full benchmarks
- 📊 Visualize results interactively with dashboards for clear analysis

Interested in optimizing disaggregated LLM inference? [👉 Contact us](https://www.bentoml.com/contact)

## Installation

Install llm-optimizer with `pip`.

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Get started

The quickest way to try llm-optimizer is with performance estimation. This feature predicts latency, throughput, and concurrency limits, helping you identify optimal configurations without running full benchmarks:

```bash
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512
```

More examples:

```bash
# With GPU specification
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 2048 \
  --output-len 1024 \
  --gpu H100 \
  --num-gpus 8

# With performance constraints and command generation
llm-optimizer estimate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 1024 \
  --output-len 512 \
  --gpu H100 \
  --num-gpus 4 \
  --constraints "ttft:mean<300ms;itl:p95<50ms"
```

For guided setup, you can also use interactive mode:

```bash
llm-optimizer estimate --interactive
```

## Run your first benchmark

Currently, llm-optimizer supports benchmarking with SGLang and vLLM to test different configurations of an LLM.

Here is an example using SGLang:

```bash
# SGLang with multiple TP/DP combinations
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,4),(2,2),(4,1)];chunked_prefill_size=[2048,4096,8192]" \
  --client-args "max_concurrency=[50,100,200];num_prompts=1000" \
  --output-json sglang_results.json
```

This command will:

- Test 3 TP/DP combinations × 3 prefill sizes = 9 server configurations
- Test each against 3 concurrency values = 27 client configurations
- In total, 27 unique benchmarks will be run, with results saved in `sglang_results.json`

More examples:

```bash
# vLLM with batch size tuning
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=[1,2,4];max_num_batched_tokens=[4096,8192,16384]" \
  --client-args "max_concurrency=[32,64,128];num_prompts=1000;dataset_name=sharegpt" \
  --output-json vllm_results.json

# Complex parameter grid for throughput optimization
llm-optimizer \
  --framework sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tp_size*dp_size=[(1,8),(2,4),(4,2)];schedule_conservativeness=[0.3,0.6,1.0];chunked_prefill_size=range(2048,8193,2048)" \
  --client-args "max_concurrency=range(50,201,50);request_rate=[10,20,50]" \
  --gpus 8 \
  --output-json complex_benchmark.json
```

## Apply constraints

Not every benchmark result is useful. You can apply constraints directly to your benchmarks so only configurations that meet your Service Level Objectives (SLOs) are returned.

```bash
# Latency-optimized configuration
llm-optimizer \
  --framework vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --server-args "tensor_parallel_size=[2,4];max_num_seqs=[16,32,64]" \
  --client-args "max_concurrency=[8,16,32];num_prompts=500" \
  --constraints "ttft<200ms;itl:p99<10ms" \
  --output-json latency_optimized.json
```

Currently llm-optimizer supports constraints on key performance metrics using mean, median, p95, or p99 values. Constraint syntax:

```bash
# Time to first token constraints
--constraints "ttft<300ms"                    # Mean TTFT under 300ms
--constraints "ttft:median<200ms"             # Median TTFT under 200ms
--constraints "ttft:p95<500ms"                # 95th percentile under 500ms

# Inter-token latency constraints
--constraints "itl:mean<20ms"                 # Mean ITL under 20ms
--constraints "itl:p99<50ms"                  # 99th percentile under 50ms

# End-to-end latency constraints
--constraints "e2e_latency:p95<2s"           # 95th percentile under 2s

# Combined constraints
--constraints "ttft:median<300ms;itl:p95<10ms;e2e_latency:p95<2s"
```

## Visualize benchmark results

llm-optimizer saves benchmark results in JSON format, including key inference metrics like TTFT, ITL, and concurrency. Since raw numbers can be difficult to interpret, it provides an interactive visualization tool to help you explore results more easily.

```bash
# Visualize results with Pareto frontier analysis
llm-optimizer visualize --data-file results.json --port 8080

# Combine multiple result files
llm-optimizer visualize --data-file "sglang_results.json,vllm_results.json" --port 8080
```

Open your browser at `http://localhost:8080/pareto_llm_dashboard.html`. The dashboard allows you to:

- Compare results from multiple runs side by side
- Explore trade-offs between different setups (e.g., latency vs. throughput)
- Identify the best-performing configurations for your workload

## Use custom server commands

By default, llm-optimizer manages server startup for supported frameworks. If you want more control, you can provide your own server command.

```bash
# Custom SGLang server
llm-optimizer \
  --server-cmd "python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000" \
  --client-args "max_concurrency=[25,50,100];num_prompts=1000" \
  --host 0.0.0.0 \
  --port 30000

# Custom vLLM server with specific GPU allocation
llm-optimizer \
  --server-cmd "vllm serve meta-llama/Llama-3.1-8B-Instruct --tensor-parallel-size 4" \
  --client-args "max_concurrency=[64,128];num_prompts=2000" \
  --port 8000
```

## Tune inference parameters

llm-optimizer exposes both server- and client-side parameters so you can experiment with different setups and measure their impact on performance. It uses native parameters from the respective frameworks (currently support vLLM and SGLang). Here are some common parameters and you can add others as needed.

### SGLang
- `tp_size*dp_size`: Tensor/Data parallelism combinations
- `chunked_prefill_size`: Prefill chunk size for throughput
- `schedule_conservativeness`: Request scheduling aggressiveness
- `schedule_policy`: Scheduling policy (fcfs, priority)

### vLLM
- `tensor_parallel_size`: Tensor parallelism degree
- `max_num_batched_tokens`: Maximum batch size in tokens
- `max_num_seqs`: Maximum concurrent sequences

### Client parameters
- `max_concurrency`: Maximum concurrent requests
- `num_prompts`: Total number of requests to send
- `dataset_name`: Dataset for request generation (`sharegpt`, `random`)
- `random_input/random_output`: Random sequence lengths

### Supported GPUs

H100, H200, A100, L20, L40, B100, B200 with accurate TFLOPS specifications.

## Development

```bash
# Code formatting and linting
ruff format
ruff check

# Type checking
mypy src/
```

## Community

llm-optimizer is actively maintained by the BentoML team. Feel free to reach out and [join our Slack community!](https://l.bentoml.com/join-slack)

## Contributing

As an open-source project, we welcome contributions of all kinds, such as new features, bug fixes, and documentation. Here are some of the ways to contribute:

- Repost a bug by [creating a GitHub issue](https://github.com/bentoml/llm-optimizer/issues/new/choose).
- [Submit a pull request](https://github.com/bentoml/llm-optimizer/compare) or help review other developers’ [pull requests](https://github.com/bentoml/llm-optimizer/pulls).

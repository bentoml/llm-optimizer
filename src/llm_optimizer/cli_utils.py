"""
CLI utility functions for interactive prompts, GPU detection, and user interface helpers.
"""

import os

import click
import pynvml

from llm_optimizer.predefined.gpus import (
    list_available_gpus,
    list_available_gpus_with_lowercase,
)

# Readline support for better interactive experience
try:
    import readline

    HAS_READLINE = True

    # Configure readline for better UX
    readline.set_startup_hook(None)
    readline.parse_and_bind("tab: complete")
    readline.parse_and_bind("set editing-mode emacs")  # Enable emacs-style editing
    readline.parse_and_bind("set completion-ignore-case on")

    # History file for model completions (optional)
    history_file = os.path.expanduser("~/.llm_optimizer_history")
    try:
        readline.read_history_file(history_file)
    except (FileNotFoundError, PermissionError):
        pass

    def save_history():
        try:
            readline.set_history_length(1000)
            readline.write_history_file(history_file)
        except:
            pass

    import atexit

    atexit.register(save_history)

except ImportError:
    HAS_READLINE = False


def friendly_prompt(
    message: str,
    default=None,
    choices=None,
    type_converter=None,
    completions=None,
    gpu_type_field=False,
):
    """
    User-friendly prompt with readline support for better interactive experience.

    Args:
        message: Prompt message to display
        default: Default value (shown in brackets)
        choices: List of valid choices (for validation)
        type_converter: Function to convert input (e.g., int)
        completions: List of completion options
        gpu_type_field: Special handling for GPU type validation

    Returns:
        User input with appropriate type conversion
    """
    if not HAS_READLINE:
        # Fallback to click.prompt if readline not available
        if choices:
            return click.prompt(message, type=click.Choice(choices), default=default)
        elif type_converter == int:
            return click.prompt(message, type=int, default=default)
        else:
            return click.prompt(message, default=default if default else "")

    # Setup completions if provided
    if completions:

        def completer(text, state):
            matches = [
                item for item in completions if item.lower().startswith(text.lower())
            ]
            try:
                return matches[state]
            except IndexError:
                return None

        readline.set_completer(completer)
        readline.set_completer_delims(" \t\n")
    else:
        readline.set_completer(None)

    # Format prompt with default
    if default is not None:
        prompt_text = f"{message} [{default}]: "
    else:
        prompt_text = f"{message}: "

    while True:
        try:
            user_input = input(prompt_text).strip()

            # Use default if empty input
            if not user_input and default is not None:
                user_input = str(default)

            # GPU type special handling (case-insensitive)
            if gpu_type_field:
                normalized_input = normalize_gpu_choice(user_input)
                if (
                    normalized_input != user_input.upper()
                    and user_input.upper() not in list_available_gpus()
                ):
                    available_gpus = ", ".join(list_available_gpus())
                    available_lower = ", ".join(
                        [name.lower() for name in list_available_gpus()]
                    )
                    print(
                        f"❌ Invalid GPU. Available: {available_gpus} (case-insensitive: {available_lower})"
                    )
                    continue
                user_input = normalized_input

            # Validate choices (skip for GPU types as they're handled above)
            elif choices and user_input not in choices:
                print(f"❌ Invalid choice. Please select from: {', '.join(choices)}")
                continue

            # Type conversion
            if type_converter:
                try:
                    return type_converter(user_input)
                except ValueError:
                    print(
                        f"❌ Invalid format. Please enter a valid {type_converter.__name__}."
                    )
                    continue

            return user_input

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            raise click.Abort()


def normalize_gpu_choice(user_input: str) -> str:
    """Normalize GPU input to uppercase for internal use."""
    if user_input.lower() in [name.lower() for name in list_available_gpus()]:
        return user_input.upper()
    return user_input  # Return as-is if not found (for error handling)


def friendly_confirm(message: str, default=True):
    """User-friendly yes/no confirmation with readline support."""
    if not HAS_READLINE:
        return click.confirm(message, default=default)

    default_text = "Y/n" if default else "y/N"
    prompt_text = f"{message} [{default_text}]: "

    while True:
        try:
            user_input = input(prompt_text).strip().lower()

            if not user_input:
                return default
            elif user_input in ["y", "yes", "true", "1"]:
                return True
            elif user_input in ["n", "no", "false", "0"]:
                return False
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")

        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            raise click.Abort()


def get_gpu_count():
    """Returns the number of available GPUs."""
    try:
        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetCount()
    except pynvml.NVMLError:
        return 0
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def detect_gpu_type():
    """Detect the GPU type from the system."""
    try:
        pynvml.nvmlInit()
        if pynvml.nvmlDeviceGetCount() == 0:
            return None

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)

        # different version of nvmlDeviceGetName may return bytes or str
        try:
            gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
        except AttributeError:
            gpu_name = pynvml.nvmlDeviceGetName(handle)

        # Map GPU names to our standardized names
        gpu_mapping = {
            "NVIDIA B200": "B200",
            "NVIDIA H100": "H100",
            "NVIDIA H200": "H200",
            "NVIDIA A100": "A100",
            "NVIDIA L20": "L20",
            "NVIDIA L40": "L40",
        }

        for full_name, short_name in gpu_mapping.items():
            if full_name in gpu_name:
                return short_name

        return gpu_name

    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass


def collect_interactive_parameters():
    """
    Collect parameters interactively from user input.

    Returns:
        dict: Dictionary with all collected parameters
    """
    click.echo("=== LLM Performance Estimation (Interactive Mode) ===")
    click.echo()

    # Get model
    click.echo("🤖 Model Selection")
    click.echo(
        "Popular options: meta-llama/Llama-3.2-1B, meta-llama/Meta-Llama-3-8B, meta-llama/Meta-Llama-3-70B"
    )
    model_completions = [
        "meta-llama/Llama-3.2-1B",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",
        "meta-llama/Llama-2-7b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf",
        "mistralai/Mistral-7B-v0.1",
        "microsoft/DialoGPT-medium",
    ]
    model = friendly_prompt("HuggingFace model ID", completions=model_completions)

    # Get input/output lengths
    click.echo("\n📏 Sequence Length Configuration")
    click.echo(
        "Typical values: 512 (short), 1024 (medium), 2048 (long), 4096 (very long)"
    )
    input_len = friendly_prompt(
        "Input sequence length (tokens)", default=1024, type_converter=int
    )
    output_len = friendly_prompt(
        "Output sequence length (tokens)", default=512, type_converter=int
    )

    # Get optimization target
    click.echo("\n🎯 Optimization Target")
    click.echo("• throughput: Maximize tokens/second (good for batch processing)")
    click.echo("• latency: Minimize response time (good for interactive use)")
    target = friendly_prompt(
        "Optimization target", default="throughput", choices=["throughput", "latency"]
    )

    # Get constraints
    click.echo("\n⚡ Performance Constraints (Optional)")
    click.echo("Examples:")
    click.echo("• 'ttft:median<300ms' - First token in under 300ms (median)")
    click.echo("• 'itl:p95<50ms' - Inter-token latency under 50ms (95th percentile)")
    click.echo("• 'ttft<200ms;itl:p99<10ms' - Multiple constraints")
    click.echo("Statistical types: mean, median, p95, p99")
    constraint_examples = [
        "ttft:median<300ms",
        "itl:p95<50ms",
        "ttft<200ms;itl:p99<10ms",
        "e2e_latency:p95<2s",
    ]
    constraints = friendly_prompt(
        "SLO constraints (press Enter to skip)",
        default="",
        completions=constraint_examples,
    )
    constraints = constraints if constraints.strip() else None

    # Get precision - infer from model config first
    click.echo("\n🔢 Model Precision")
    click.echo("• fp16: Standard precision (good balance)")
    click.echo("• bf16: Brain float 16, common in modern LLMs")
    click.echo("• fp8: Higher throughput but requires newer GPUs (H100+)")

    # Try to infer precision from the model config
    inferred_precision = "fp16"  # fallback default
    try:
        # Import here to avoid circular imports
        from llm_optimizer.common import get_model_config_and_precision_from_hf

        click.echo(f"🔍 Inferring precision from {model} config...")
        model_config = get_model_config_and_precision_from_hf(model)
        inferred_precision = model_config.inferred_precision
        click.echo(f"💡 Detected precision: {inferred_precision}")
    except ImportError:
        click.echo("⚠️  Could not import precision inference function")
        inferred_precision = "fp16"
    except Exception as e:
        click.echo(
            f"⚠️  Could not infer precision from model config ({type(e).__name__})"
        )
        click.echo("Using default: fp16")
        inferred_precision = "fp16"

    precision = friendly_prompt(
        "Model precision", default=inferred_precision, choices=["fp16", "bf16", "fp8"]
    )

    # Get framework
    click.echo("\n🚀 Framework Selection")
    click.echo("• sglang: Fast inference engine optimized for throughput")
    click.echo("• vllm: Popular serving framework with good compatibility")
    click.echo("• both: Generate configs for both frameworks")
    framework = friendly_prompt(
        "Framework", default="both", choices=["sglang", "vllm", "both"]
    )

    # Get dataset selection
    click.echo("\n📊 Dataset Selection")
    click.echo(
        "• random: Generate synthetic prompts with specified input/output lengths"
    )
    click.echo("• sharegpt: Use real conversational data from ShareGPT dataset")
    dataset = friendly_prompt(
        "Dataset type", default="random", choices=["random", "sharegpt"]
    )

    return {
        "model": model,
        "input_len": input_len,
        "output_len": output_len,
        "target": target,
        "constraints": constraints,
        "precision": precision,
        "framework": framework,
        "dataset": dataset,
    }


def collect_gpu_configuration(interactive=True, gpu=None, num_gpus=None):
    """
    Collect GPU configuration with auto-detection and user confirmation.

    Args:
        interactive: Whether to prompt user for confirmation/changes
        gpu: Pre-specified GPU type (if any)
        num_gpus: Pre-specified GPU count (if any)

    Returns:
        tuple: (gpu_type, num_gpus)
    """
    if interactive:
        click.echo("\n💻 GPU Configuration")

    # Handle GPU type
    if not gpu:
        detected_gpu = detect_gpu_type()
        if detected_gpu:
            if not interactive:
                click.echo(f"Auto-detected GPU: {detected_gpu}")
                # Check if the detected GPU is in the supported list
                supported_gpus = list_available_gpus()
                supported_gpus_lower = list_available_gpus_with_lowercase()
                if (
                    detected_gpu.upper() not in supported_gpus
                    and detected_gpu.lower() not in supported_gpus_lower
                ):
                    available_gpus = ", ".join(supported_gpus)
                    click.echo(f"Auto-detected GPU '{detected_gpu}' is not supported.")
                    click.echo(
                        f"Please specify a supported GPU with --gpu. "
                        f"Available GPUs: {available_gpus}"
                    )
                    raise click.ClickException(f"Unsupported GPU: {detected_gpu}")
                gpu = detected_gpu
            else:
                click.echo(f"Auto-detected GPU: {detected_gpu}")
                gpu = friendly_prompt(
                    "GPU model (press Enter to use auto-detected)",
                    default=detected_gpu,
                    completions=list_available_gpus_with_lowercase(),
                    gpu_type_field=True,
                )
        else:
            available_gpus = ", ".join(list_available_gpus())
            if not interactive:
                click.echo(
                    f"Could not auto-detect GPU. Available GPUs: {available_gpus}"
                )
                raise click.ClickException(
                    "GPU auto-detection failed and no GPU specified"
                )
            else:
                click.echo(
                    f"Could not auto-detect GPU. Available GPUs: {available_gpus}"
                )
                gpu = friendly_prompt(
                    "GPU model",
                    completions=list_available_gpus_with_lowercase(),
                    gpu_type_field=True,
                )
    elif interactive:
        # GPU was provided via command line but we're in interactive mode - ask for confirmation
        click.echo(f"Command-line GPU: {gpu}")
        gpu = friendly_prompt(
            "GPU model (press Enter to keep current)",
            default=gpu,
            completions=list_available_gpus_with_lowercase(),
            gpu_type_field=True,
        )

    # Handle GPU count
    if not num_gpus:
        detected_gpus = get_gpu_count()
        if detected_gpus > 0:
            if not interactive:
                click.echo(f"Auto-detected {detected_gpus} GPU(s)")
                num_gpus = detected_gpus
            else:
                click.echo(f"Auto-detected: {detected_gpus} GPU(s)")
                num_gpus = friendly_prompt(
                    "Number of GPUs (press Enter to use auto-detected)",
                    default=detected_gpus,
                    type_converter=int,
                )
        else:
            if not interactive:
                num_gpus = 1
                click.echo("No GPUs detected, using 1 GPU")
            else:
                num_gpus = friendly_prompt(
                    "Number of GPUs", default=1, type_converter=int
                )
    elif interactive:
        # GPU count was provided via command line but we're in interactive mode - ask for confirmation
        click.echo(f"Command-line GPU count: {num_gpus}")
        num_gpus = friendly_prompt(
            "Number of GPUs (press Enter to keep current)",
            default=num_gpus,
            type_converter=int,
        )

    return gpu, num_gpus

#!/usr/bin/env python3
"""
Pareto LLM Optimizer Dashboard

This script starts a local HTTP server to serve an interactive visualization
page for benchmark data using Plotly.js.
"""

import argparse
import json
import logging
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from llm_optimizer.utils import InfinityToNullEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def convert_constraints_for_visualization(parsed_constraints):
    """
    Convert parsed SLO constraints to the format expected by visualize.py.

    Args:
        parsed_constraints: List of SLOConstraint objects

    Returns:
        List of constraint objects with operator information preserved
        Example: [{"op": "<", "value": 1000.0, "name": "mean_ttft_ms"}]
    """
    if not parsed_constraints:
        return []

    constraints_list = []

    for constraint in parsed_constraints:
        # Convert metric name to match benchmark results format
        metric_mapping = {
            "ttft": "ttft_ms",
            "itl": "itl_ms",
            "tpot": "tpot_ms",
            "e2e_latency": "e2e_latency_ms",
        }

        base_metric = metric_mapping.get(constraint.metric, constraint.metric)

        # Convert value to milliseconds if needed
        value_ms = constraint.value
        if constraint.unit == "s":
            value_ms = constraint.value * 1000

        # Create field name with stat_type prefix
        if constraint.stat_type == "mean":
            field_name = f"mean_{base_metric}"
        else:
            field_name = f"{constraint.stat_type}_{base_metric}"

        constraints_list.append(
            {"op": constraint.operator, "value": value_ms, "name": field_name}
        )

    return constraints_list


class ParetoLLMOptimizer:
    def __init__(self, config_file: str = "visualization_config.json"):
        self.config_file = config_file
        self.config = self.load_config(config_file)

    def load_config(self, config_file: str) -> dict:
        """Load visualization configuration from JSON file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)

            # Framework will be detected later in load_benchmark_data

            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def load_benchmark_data(self, data_file: str) -> dict:
        """Load benchmark data from JSON files."""
        data = []
        constraints = {}
        metadata = {}
        best_configurations = {}
        json_files = [Path(data_file)]

        # Load data from the specified file
        try:
            with open(data_file, encoding="utf-8") as f:
                file_data = json.load(f)

                # Expect new enhanced structure only
                if not isinstance(file_data, dict) or "test_results" not in file_data:
                    raise ValueError(
                        "Invalid data format. Expected enhanced format with "
                        "'metadata', 'best_configurations', and 'test_results' fields."
                    )

                # Extract data from enhanced format
                data = file_data["test_results"]
                metadata = file_data.get("metadata", {})
                best_configurations = file_data.get("best_configurations", {})
                constraints = metadata.get("constraints", {})

                # If constraints not in metadata, try to extract from individual results
                if not constraints:
                    for item in data:
                        if (
                            isinstance(item, dict)
                            and "constraints" in item
                            and item["constraints"]
                        ):
                            constraints = item["constraints"]
                            break

            logger.info(f"Loaded data from {data_file}")
            logger.info(
                f"Metadata: GPU={metadata.get('gpu_type')} "
                f"x{metadata.get('gpu_count')}, Model={metadata.get('model_tag')}"
            )
            logger.info(f"Total tests: {metadata.get('total_tests', len(data))}")

        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}")
            raise

        logger.info(f"Loaded {len(data)} benchmark records from {data_file}")
        logger.info(f"Found constraints: {constraints}")

        return {
            "data": data,
            "constraints": constraints,
            "data_files": json_files,
            "metadata": metadata,
            "best_configurations": best_configurations,
        }

    def get_field_options(self, data: list = None) -> dict:
        """Get field options by category, with disabled fields for missing data."""
        if not self.config or "fields" not in self.config:
            return {}

        categories = {}
        for field_id, field_info in self.config["fields"].items():
            category = field_info.get("category", "other")
            if category not in categories:
                categories[category] = {
                    "label": self.config.get("categories", {})
                    .get(category, {})
                    .get("label", category.title()),
                    "description": self.config.get("categories", {})
                    .get(category, {})
                    .get("description", ""),
                    "fields": {},
                }

            categories[category]["fields"][field_id] = field_info

        return categories

    def get_field_categories(self) -> dict:
        """Get field category information for hover display."""
        if not self.config or "field_categories" not in self.config:
            return {}
        return self.config["field_categories"]

    def create_html_page(self, data_dict: dict) -> str:
        """Create the HTML page with embedded data and configuration."""
        # Load the HTML template
        # Try to find template relative to this file's location
        current_dir = Path(__file__).parent
        template_path = current_dir / "template.html"
        if not template_path.exists():
            logger.error(f"Template file not found at {template_path}")
            return ""

        with open(template_path, encoding="utf-8") as f:
            html_content = f.read()

        # Prepare data for embedding
        data_json = json.dumps(data_dict["data"], indent=2, cls=InfinityToNullEncoder)
        constraints_json = json.dumps(
            data_dict["constraints"], indent=2, cls=InfinityToNullEncoder
        )
        config_json = json.dumps(self.config, indent=2, cls=InfinityToNullEncoder)
        field_options_json = json.dumps(
            self.get_field_options(data_dict["data"]),
            indent=2,
            cls=InfinityToNullEncoder,
        )
        field_categories_json = json.dumps(
            self.get_field_categories(), indent=2, cls=InfinityToNullEncoder
        )
        defaults_json = json.dumps(
            self.config.get("defaults", {}), indent=2, cls=InfinityToNullEncoder
        )
        metadata_json = json.dumps(
            data_dict.get("metadata", {}), indent=2, cls=InfinityToNullEncoder
        )
        best_configs_json = json.dumps(
            data_dict.get("best_configurations", {}),
            indent=2,
            cls=InfinityToNullEncoder,
        )

        # Get UI configuration
        ui_config = self.config.get("ui", {})

        # Replace placeholders in the template
        html_content = html_content.replace(
            "{title}", ui_config.get("title", "Pareto LLM Optimizer Dashboard")
        )
        html_content = html_content.replace(
            "{subtitle}",
            ui_config.get(
                "subtitle", "Interactive Performance Analysis & Optimization Tool"
            ),
        )
        html_content = html_content.replace(
            "{description}",
            ui_config.get(
                "description",
                "Select different metrics for X and Y axes to analyze performance "
                "relationships and identify Pareto optimal configurations. "
                "Hover over data points to see detailed configuration information.",
            ),
        )
        html_content = html_content.replace("{data_json}", data_json)
        html_content = html_content.replace("{constraints_json}", constraints_json)
        html_content = html_content.replace("{config_json}", config_json)
        html_content = html_content.replace("{field_options_json}", field_options_json)
        html_content = html_content.replace(
            "{field_categories_json}", field_categories_json
        )
        html_content = html_content.replace("{defaults_json}", defaults_json)
        html_content = html_content.replace("{metadata_json}", metadata_json)
        html_content = html_content.replace("{best_configs_json}", best_configs_json)

        return html_content

    def generate_dashboard(self, data_file: str, output_file: str = None) -> str:
        """Generate the dashboard HTML file."""
        # Load benchmark data
        data_dict = self.load_benchmark_data(data_file)

        # Create HTML page
        html_content = self.create_html_page(data_dict)

        # Write to file
        html_file = output_file if output_file else "pareto_llm_dashboard.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Created Pareto LLM Dashboard: {html_file}")
        return html_file

    def start_server(self, port: int = 8080):
        """Start the HTTP server."""
        try:
            server = HTTPServer(("localhost", port), SimpleHTTPRequestHandler)
            logger.info(f"Starting server at http://localhost:{port}")
            logger.info(
                f"Dashboard available at: http://localhost:{port}/pareto_llm_dashboard.html"
            )
            logger.info("Press Ctrl+C to stop the server")

            # Open the dashboard in the default browser
            import webbrowser

            webbrowser.open(f"http://localhost:{port}/pareto_llm_dashboard.html")

            server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Error starting server: {e}")


def main():
    parser = argparse.ArgumentParser(description="Pareto LLM Optimizer Dashboard")
    parser.add_argument(
        "--data", required=True, help="Data file path (e.g., sample_data.json)"
    )
    parser.add_argument(
        "--config", default="visualization_config.json", help="Configuration file path"
    )
    parser.add_argument("--port", type=int, default=8080, help="Server port")

    args = parser.parse_args()

    # Create optimizer instance
    optimizer = ParetoLLMOptimizer(args.config)

    # Generate dashboard
    optimizer.generate_dashboard(args.data)

    # Start server
    optimizer.start_server(args.port)


if __name__ == "__main__":
    main()

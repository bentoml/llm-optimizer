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

# Add scipy for curve fitting
try:
    import numpy as np
    from scipy import stats
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
    print("scipy and numpy successfully imported")
except ImportError as e:
    SCIPY_AVAILABLE = False
    print(
        f"Warning: scipy not available. Error: {e}. Fit lines will not be calculated."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class ParetoLLMOptimizer:
    def __init__(self, config_file: str = "visualization_config.json"):
        self.config_file = config_file
        self.config = self.load_config(config_file)

    def load_config(self, config_file: str) -> dict:
        """Load visualization configuration from JSON file."""
        try:
            with open(config_file, encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
            return config
        except FileNotFoundError:
            logger.error(f"Configuration file {config_file} not found")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return {}

    def extract_value(self, data: dict, path: str):
        """Extract value from nested dictionary using dot notation path."""
        keys = path.split(".")
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def flatten_data(self, data: list) -> list:
        """Flatten the nested data structure according to the data mapping."""
        if not self.config or "data_mapping" not in self.config:
            return data

        flattened = []
        mapping = self.config["data_mapping"]

        for item in data:
            flattened_item = {}
            for field_name, data_path in mapping.items():
                value = self.extract_value(item, data_path)
                if value is not None:
                    flattened_item[field_name] = value
            flattened.append(flattened_item)

        return flattened

    def load_benchmark_data(self, data_file: str) -> dict:
        """Load benchmark data from JSON files."""
        data = []
        constraints = {}
        json_files = [Path(data_file)]

        # Load data from the specified file
        try:
            with open(data_file, encoding="utf-8") as f:
                file_data = json.load(f)
                if isinstance(file_data, list):
                    data.extend(file_data)
                    # Extract constraints from the first item that has them
                    for item in file_data:
                        if isinstance(item, dict) and "constraints" in item and item["constraints"]:
                            constraints = item["constraints"]
                            break
                else:
                    data.append(file_data)
                    # Extract constraints if present
                    if isinstance(file_data, dict) and "constraints" in file_data and file_data["constraints"]:
                        constraints = file_data["constraints"]
            logger.info(f"Loaded data from {data_file}")
        except Exception as e:
            logger.error(f"Error loading {data_file}: {e}")

        # Flatten the data according to the mapping
        flattened_data = self.flatten_data(data)
        logger.info(f"Loaded {len(flattened_data)} benchmark records from {data_file}")
        logger.info(f"Found constraints: {constraints}")
        return {
            "data": flattened_data,
            "constraints": constraints,
            "data_files": json_files,
        }

    def get_available_fields(self, data: list) -> set:
        """Get fields that have data in the dataset."""
        available_fields = set()

        for item in data:
            for field_name in item.keys():
                if item[field_name] is not None and item[field_name] != "":
                    available_fields.add(field_name)

        return available_fields

    def get_field_options(self, data: list = None) -> dict:
        """Get field options organized by category, with disabled fields for missing data."""
        if not self.config or "fields" not in self.config:
            return {}

        # Get available fields if data is provided
        available_fields = set()
        if data is not None:
            available_fields = self.get_available_fields(data)
            logger.info(f"Available fields with data: {available_fields}")

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

            # Add disabled flag for fields without data
            field_info_copy = field_info.copy()
            if data is not None and field_id not in available_fields:
                field_info_copy["disabled"] = True
                field_info_copy["disabled_reason"] = "No data available"
                logger.info(f"Field '{field_id}' disabled - no data available")
            else:
                field_info_copy["disabled"] = False

            categories[category]["fields"][field_id] = field_info_copy

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

        # Get available fields with data
        available_fields = self.get_available_fields(data_dict["data"])
        logger.info(f"Available fields: {available_fields}")

        # Prepare data for embedding
        data_json = json.dumps(data_dict["data"], indent=2)
        constraints_json = json.dumps(data_dict["constraints"], indent=2)
        config_json = json.dumps(self.config, indent=2)
        field_options_json = json.dumps(
            self.get_field_options(data_dict["data"]), indent=2
        )
        field_categories_json = json.dumps(self.get_field_categories(), indent=2)
        defaults_json = json.dumps(self.config.get("defaults", {}), indent=2)

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
                "Select different metrics for X and Y axes to analyze performance relationships and identify Pareto optimal configurations. Hover over data points to see detailed configuration information.",
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

        return html_content

    def generate_dashboard(self, data_file: str) -> str:
        """Generate the dashboard HTML file."""
        # Load benchmark data
        data_dict = self.load_benchmark_data(data_file)

        # Create HTML page
        html_content = self.create_html_page(data_dict)

        # Write to file
        html_file = "pareto_llm_dashboard.html"
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

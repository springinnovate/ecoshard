"""Code for ecoshard.geosplitter."""
import configparser
import os


class GeoSplitterConfig:
    REQUIRED_SECTIONS = {
        "input": [
            "aoi_path",
            "area_threshold_in_projected_units",
        ],
        "expected_output": [
        ],
        "function": [
            "module",
            "function_name"
            "input_to_arg_mapping"
        ],
    }

    def __init__(self, ini_file_path):
        self.ini_file_path = ini_file_path
        self.config = configparser.ConfigParser()

    def parse(self):
        # Ensure the INI file exists
        if not os.path.exists(self.ini_file_path):
            raise FileNotFoundError(
                f"INI file not found: {self.ini_file_path}")

        # Read and parse the INI file
        self.config.read(self.ini_file_path)

        # Validate the sections and fields
        self._validate_sections()

        # Validate the __file__ section
        self._validate_filename()

    def _validate_sections(self):
        missing_sections = []
        for section, keys in self.REQUIRED_SECTIONS.items():
            if section not in self.config:
                missing_sections.append(section)
                continue

            # Check for missing keys in each section
            missing_keys = [key for key in keys if key not in self.config[section]]
            if missing_keys:
                raise ValueError(f"Missing keys in section [{section}]: {', '.join(missing_keys)}")

        if missing_sections:
            raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

    def _validate_filename(self):
        if "__file__" not in self.config:
            raise ValueError("Missing section: [__file__]")

        expected_basename = os.path.basename(self.ini_file_path)
        actual_basename = self.config["__file__"].get("__file__", "").strip()

        if actual_basename != expected_basename:
            raise ValueError(
                f"INI file basename mismatch: expected '{expected_basename}', found '{actual_basename}'"
            )


def run_pipeline(config_path):
    """Execute the geosplitter pipeline with the config_path ini file."""

    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the GeoSplitter pipeline.")
    parser.add_argument("config_file", help="Path to the INI configuration file.")
    args = parser.parse_args()
    run_pipeline(args.config_file)

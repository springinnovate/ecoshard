"""Code for ecoshard.geosplitter."""
import configparser
import os
import logging
import sys


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


class GeoSplitterConfig:
    INI_FILE_BASE = 'ini_file_base'
    REQUIRED_SECTIONS = {
        INI_FILE_BASE: [
            "aoi_path",
            "aoi_subdivision_area_threshold",
        ],
        "expected_output": [
        ],
        "function": [
            "module",
            "function_name"
        ],
    }

    def __init__(self, ini_file_path):
        self.ini_file_path = ini_file_path
        self.ini_base = os.path.basename(os.path.splitext(ini_file_path)[0])
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.ini_file_path):
            raise FileNotFoundError(
                f"INI file not found: {self.ini_file_path}")

        self.config.read(self.ini_file_path)
        self._validate_sections()

    def _validate_sections(self):
        missing_sections = []
        for section, keys in self.REQUIRED_SECTIONS.items():
            if section == GeoSplitterConfig.INI_FILE_BASE:
                # remap it to the expected section name which is the basename of the ini file
                section = self.ini_base
            if section not in self.config:
                missing_sections.append(section)
                continue

            # Check for missing keys in each section
            missing_keys = [key for key in keys if key not in self.config[section]]
            if missing_keys:
                raise ValueError(f"Missing keys in section [{section}]: {', '.join(missing_keys)}")

        if missing_sections:
            raise ValueError(f"Missing required sections: {', '.join(missing_sections)}")

        LOGGER.info(f'{self.ini_file_path} is valid')


def run_pipeline(config_path):
    """Execute the geosplitter pipeline with the config_path ini file."""
    geosplitter = GeoSplitterConfig(config_path)
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the GeoSplitter pipeline.")
    parser.add_argument("config_file", help="Path to the INI configuration file.")
    args = parser.parse_args()
    run_pipeline(args.config_file)

"""Code for ecoshard.geosplitter."""
from datetime import datetime
import collections
import configparser
import os
import logging
import sys
import math

import ecoshard.taskgraph as taskgraph
import ecoshard.geoprocessing as geoprocessing
from osgeo import gdal
from osgeo import osr
from osgeo import ogr


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.WARNING)


class GeoSplitter:
    INI_FILE_BASE = 'ini_file_base'
    GLOBAL_WORKSPACE_DIR = 'global_workspace_dir'
    AOI_PATH = 'aoi_path'
    PROJECTION_SECTION = 'projection'
    PROJECTION_SOURCE = 'projection_source'
    SUBDIVISION_BLOCK_SIZE = 'SUBDIVISION_BLOCK_SIZE'
    REQUIRED_SECTIONS = {
        INI_FILE_BASE: [
            AOI_PATH,
            'aoi_subdivision_area_min_threshold',
            'GLOBAL_WORKSPACE_DIR',
        ],
        'expected_output': [
        ],
        'function': [
            'module',
            'function_name'
        ],
        PROJECTION_SECTION: [
            PROJECTION_SOURCE,
            SUBDIVISION_BLOCK_SIZE,
        ]
    }

    def __init__(self, ini_file_path):
        self.ini_file_path = ini_file_path
        self.ini_base = os.path.basename(os.path.splitext(ini_file_path)[0])
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.ini_file_path):
            raise FileNotFoundError(
                f"INI file not found: {self.ini_file_path}")

        self.config.read(self.ini_file_path)
        self.workspace_dir = self.config[self.ini_base][GeoSplitter.GLOBAL_WORKSPACE_DIR]
        self._validate_sections()
        self.task_graph = taskgraph.TaskGraph(
            self.workspace_dir, os.cpu_count(), 15.0,
            taskgraph_name='batch aois')
        self.aoi_path = self.config[self.ini_base][GeoSplitter.AOI_PATH]

        self.aoi_split_complete_token_path = os.path.join(
            self.workspace_dir, 'aoi_split_complete.token')
        self._batch_into_aoi_subsets(
            float(self.config[GeoSplitter.PROJECTION_SECTION][GeoSplitter.SUBDIVISION_BLOCK_SIZE]))

    def _validate_sections(self):
        missing_sections = []
        for section, keys in self.REQUIRED_SECTIONS.items():
            if section == GeoSplitter.INI_FILE_BASE:
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

    def _batch_into_aoi_subsets(
            self, subdivision_area_size):
        """Construct geospatially adjacent subsets.

        Breaks aois up smaller groups of proximally similar aois and limits
        the upper size to no more than `n_raster_block_sizes`.

        Args:
            subdivision_area_size (float): approximate size of the blocks to subdivide
                the AOIs in the same projected units of self.aoi_path.

        Returns:
            list of (job_id, aoi.gpkg) tuples where the job_id is a
            unique identifier for that subaoi set and aoi.gpkg is
            a subset of the original global aoi files.

        """
        # ensures we don't have more than 1000 aois per job
        aoi_path_area_list = []
        job_id_set = set()

        subbatch_job_index_map = collections.defaultdict(int)
        aoi_fid_index = collections.defaultdict(
            lambda: [list(), list(), 0])
        aoi_basename = os.path.splitext(os.path.basename(self.aoi_path))[0]
        aoi_ids = None
        aoi_vector = gdal.OpenEx(self.aoi_path, gdal.OF_VECTOR)
        aoi_layer = aoi_vector.GetLayer()

        subdivision_block_length = math.sqrt(subdivision_area_size)

        # aoi layer is either the layer or a list of features
        for aoi_feature in aoi_layer:
            fid = aoi_feature.GetFID()
            aoi_geom = aoi_feature.GetGeometryRef()
            aoi_centroid = aoi_geom.Centroid()
            # clamp into degree_separation squares
            x, y = [
                int(v//subdivision_block_length)*subdivision_block_length for v in (
                    aoi_centroid.GetX(), aoi_centroid.GetY())]
            job_id = f'{aoi_basename}_{x}_{y}'
            aoi_fid_index[job_id][0].append(fid)
            aoi_envelope = aoi_geom.GetEnvelope()
            aoi_bb = [aoi_envelope[i] for i in [0, 2, 1, 3]]
            aoi_fid_index[job_id][1].append(aoi_bb)
            aoi_fid_index[job_id][2] += aoi_geom.Area()

        aoi_geom = None
        aoi_feature = None

        aoi_subset_dir = os.path.join(self.workspace_dir, 'aoi_subsets')
        os.makedirs(aoi_subset_dir, exist_ok=True)

        for (job_id), (fid_list, aoi_envelope_list, area) in \
                sorted(
                    aoi_fid_index.items(), key=lambda x: x[1][-1],
                    reverse=True):
            if job_id in job_id_set:
                raise ValueError(f'{job_id} already processed')
            job_id_set.add(job_id)

            aoi_subset_path = os.path.join(
                aoi_subset_dir, f'{job_id}_a{area:.3f}.gpkg')
            if not os.path.exists(aoi_subset_path):
                self.task_graph.add_task(
                    func=GeoSplitter._create_fid_subset,
                    args=(
                        self.aoi_path, fid_list, aoi_subset_path),
                    target_path_list=[aoi_subset_path],
                    task_name=job_id)
            aoi_path_area_list.append(
                (area, aoi_subset_path))

        aoi_layer = None
        aoi_vector = None

        self.task_graph.join()

        # create a global sorted aoi path list so it's sorted by area overall
        # not just by region per area
        with open(self.aoi_split_complete_token_path, 'w') as token_file:
            token_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        sorted_aoi_path_list = [
            path for area, path in sorted(aoi_path_area_list, reverse=True)]
        return sorted_aoi_path_list

    @staticmethod
    def _create_fid_subset(
            base_vector_path, fid_list, target_vector_path):
        """Create subset of vector that matches fid list, projected into epsg."""
        vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        layer.SetAttributeFilter(
            f'"FID" in ('
            f'{", ".join([str(v) for v in fid_list])})')
        feature_count = layer.GetFeatureCount()
        gpkg_driver = ogr.GetDriverByName('gpkg')
        subset_vector = gpkg_driver.CreateDataSource(target_vector_path)
        subset_vector.CopyLayer(
            layer, os.path.basename(os.path.splitext(target_vector_path)[0]))
        subset_vector = None
        layer = None
        vector = None
        target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
        target_layer = target_vector.GetLayer()
        if feature_count != target_layer.GetFeatureCount():
            raise ValueError(
                f'expected {feature_count} in {target_vector_path} but got '
                f'{target_layer.GetFeatureCount()}')
        target_layer = None
        target_vector = None


def run_pipeline(config_path):
    """Execute the geosplitter pipeline with the config_path ini file."""
    geosplitter = GeoSplitter(config_path)
    pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the GeoSplitter pipeline.")
    parser.add_argument("config_file", help="Path to the INI configuration file.")
    args = parser.parse_args()
    run_pipeline(args.config_file)

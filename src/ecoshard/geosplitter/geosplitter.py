"""Code for ecoshard.geosplitter."""
import subprocess
import hashlib
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


MAX_DIRECTORIES_PER_LEVEL = 1000


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
        self.aoi_path_list = None  # will be set by _batch_into_aoi_subsets
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
        LOGGER.info(f'batching {self.aoi_path} into subsets at {self.workspace_dir}')
        self._batch_into_aoi_subsets(
            float(self.config[GeoSplitter.PROJECTION_SECTION][GeoSplitter.SUBDIVISION_BLOCK_SIZE]))
        LOGGER.info('done batching AOI subsets')
        LOGGER.debug(self.aoi_path_list[:10])

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

        aoi_fid_index = collections.defaultdict(lambda: [list(), 0])
        aoi_basename = os.path.splitext(os.path.basename(self.aoi_path))[0]
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
            aoi_fid_index[job_id][1] += aoi_geom.Area()

        aoi_geom = None
        aoi_feature = None

        n_subdirectory_levels = max(0, math.ceil(
            math.log(len(aoi_fid_index)) / math.log(MAX_DIRECTORIES_PER_LEVEL))-1)

        aoi_subset_dir = os.path.join(self.workspace_dir, 'aoi_subsets')

        for job_index, ((job_id), (fid_list, area)) in \
                enumerate(sorted(
                    aoi_fid_index.items(), key=lambda x: x[1][-1],
                    reverse=True)):
            if job_id in job_id_set:
                raise ValueError(f'{job_id} already processed')
            job_id_set.add(job_id)

            subdirectory_path = GeoSplitter._hash_to_subdirectory(
                job_id, n_subdirectory_levels, MAX_DIRECTORIES_PER_LEVEL)
            local_aoi_subset_dir = os.path.join(aoi_subset_dir, subdirectory_path, job_id)
            os.makedirs(local_aoi_subset_dir, exist_ok=True)

            aoi_subset_path = os.path.join(
                local_aoi_subset_dir, f'{job_id}.gpkg')
            if not os.path.exists(aoi_subset_path):
                job_id_prompt = f'{job_index} of {len(aoi_fid_index)}'
                self.task_graph.add_task(
                    func=GeoSplitter._create_fid_subset,
                    args=(
                        job_id_prompt, self.aoi_path, fid_list, aoi_subset_path),
                    target_path_list=[aoi_subset_path],
                    task_name=job_id)
            aoi_path_area_list.append((area, aoi_subset_path))

        aoi_layer = None
        aoi_vector = None

        self.task_graph.join()

        # create a global sorted aoi path list so it's sorted by area overall
        # not just by region per area
        with open(self.aoi_split_complete_token_path, 'w') as token_file:
            token_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.aoi_path_list = [
            path for area, path in sorted(aoi_path_area_list, reverse=True)]

    @staticmethod
    def _copy_to_new_layer(src_layer, target_vector_path):
        gpkg_driver = ogr.GetDriverByName('GPKG')
        target_vector = gpkg_driver.CreateDataSource(target_vector_path)

        # Create a brand-new layer with MULTIPOLYGON type
        layer_name = os.path.splitext(os.path.basename(target_vector_path))[0]
        spatial_ref = src_layer.GetSpatialRef()

        target_layer = target_vector.CreateLayer(
            layer_name,
            srs=spatial_ref,
            geom_type=ogr.wkbMultiPolygon
        )

        # Copy field definitions
        src_layer_defn = src_layer.GetLayerDefn()
        for i in range(src_layer_defn.GetFieldCount()):
            field_defn = src_layer_defn.GetFieldDefn(i)
            target_layer.CreateField(field_defn)

        # Now iterate and copy each feature
        for src_feat in src_layer:
            geom = src_feat.GetGeometryRef()

            # If we encounter POLYGONs, convert them to MULTIPOLYGON
            # to match the declared layer type
            if geom and geom.GetGeometryType() == ogr.wkbPolygon:
                geom = ogr.ForceToMultiPolygon(geom)

            dst_feat = ogr.Feature(target_layer.GetLayerDefn())
            dst_feat.SetFrom(src_feat)  # copy attributes
            dst_feat.SetGeometry(geom)  # set (possibly converted) geometry
            target_layer.CreateFeature(dst_feat)
        target_layer = None
        target_vector = None

    @staticmethod
    def _create_fid_subset(
            job_id_prompt, base_vector_path, fid_list, target_vector_path):
        """Create subset of vector that matches fid list, projected into epsg."""
        vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
        layer = vector.GetLayer()
        layer.SetAttributeFilter(
            f'"FID" in ('
            f'{", ".join([str(v) for v in fid_list])})')
        feature_count = layer.GetFeatureCount()
        layer_name = os.path.basename(os.path.splitext(target_vector_path)[0])
        cmd = [
            'ogr2ogr',
            '-f', 'GPKG',
            '-where', f'FID in ({", ".join(str(f) for f in fid_list)})',
            '-nln', layer_name,
            '-nlt', 'MULTIPOLYGON',
            target_vector_path,
            base_vector_path
        ]
        LOGGER.info(f'{job_id_prompt} subsetting {layer_name} ')
        subprocess.run(cmd, check=True)
        LOGGER.info(f'DONE with {job_id_prompt} subsetting {layer_name} ')

        target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
        target_layer = target_vector.GetLayer()
        if feature_count != target_layer.GetFeatureCount():
            raise ValueError(
                f'expected {feature_count} in {target_vector_path} but got '
                f'{target_layer.GetFeatureCount()}')
        target_layer = None
        target_vector = None

    @staticmethod
    def _hash_to_subdirectory(str_kernel, n_levels, max_directories_per_level):
        """Generate a multi-level subdirectory path based on a hash of the str_kernel.

        Returns:
            subdirectory path as a string

        """
        # get enough characters to make sure each level could get max_directories given
        # that the hash is 16 bit hex
        if n_levels == 0:
            return ''
        chars_per_level = math.ceil(math.log(max_directories_per_level) / math.log(16)) + 1
        job_hash = hashlib.md5(str(str_kernel).encode('utf-8')).hexdigest()
        subdirs = []
        for i in range(n_levels):
            start = i * chars_per_level
            end = start + chars_per_level
            subdirs.append(job_hash[start:end])

        return os.path.join(*subdirs)

    def split_raster_data(self, aoi_subset_path):
        """Split the config's spaital input based on the given aoi_subset_path."""

        pass


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

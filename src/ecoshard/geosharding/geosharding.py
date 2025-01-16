"""Code for ecoshard.geosplitter."""
import ast
import importlib
import re
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
import numpy

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


class GeoSharding:
    SHARD_ID = 'shard_id'
    MULTI_AOI_IN_BATCH = 'multi_aoi_in_batch'
    SHARD_AOI_PATH = 'shard_aoi_path'
    INI_FILE_BASE = 'ini_file_base'
    GLOBAL_WORKSPACE_DIR = 'global_workspace_dir'
    SHARD_WORKING_DIR = 'shard_working_dir'
    AOI_PATH = 'aoi_path'
    PROJECTION_SECTION = 'projection'
    EXPECTED_OUTPUT_SECTION = 'expected_output'
    PROJECTION_SOURCE = 'projection_source'
    SUBDIVISION_BLOCK_SIZE = 'subdivision_block_size'
    SPATIAL_INPUT_SECTION = 'spatial_input'
    NON_SPATIAL_INPUT_SECTION = 'non_spatial_input'
    TARGET_PIXEL_SIZE = 'target_pixel_size'
    TARGET_PROJECTION_AND_BB_SOURCE = 'target_projection_and_bb_source'
    FUNCTION_SECTION = 'function'
    MODULE = 'module'
    FUNCTION_NAME = 'function_name'
    REQUIRED_SECTIONS = {
        INI_FILE_BASE: [
            AOI_PATH,
            'aoi_subdivision_area_min_threshold',
            GLOBAL_WORKSPACE_DIR,
        ],
        FUNCTION_SECTION: [
            MODULE,
            FUNCTION_NAME,
        ],
        SPATIAL_INPUT_SECTION: [
        ],
        PROJECTION_SECTION: [
            PROJECTION_SOURCE,
            SUBDIVISION_BLOCK_SIZE,
            TARGET_PIXEL_SIZE,
        ],
        EXPECTED_OUTPUT_SECTION: [
            TARGET_PIXEL_SIZE,
            TARGET_PROJECTION_AND_BB_SOURCE,
        ],
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

        self._apply_dynamic_replacements_to_ini()
        self.workspace_dir = self.config[self.ini_base][GeoSharding.GLOBAL_WORKSPACE_DIR]
        self.task_graph = taskgraph.TaskGraph(
            self.workspace_dir, os.cpu_count(), 15.0,
            taskgraph_name='batch aois')
        self.aoi_path = self.config[self.ini_base][GeoSharding.AOI_PATH]

        self.aoi_split_complete_token_path = os.path.join(
            self.workspace_dir, 'aoi_split_complete.token')
        LOGGER.info(f'batching {self.aoi_path} into subsets at {self.workspace_dir}')
        self._batch_into_aoi_subsets(
            float(self.config[GeoSharding.PROJECTION_SECTION][GeoSharding.SUBDIVISION_BLOCK_SIZE]))
        LOGGER.info('done batching AOI subsets')
        # list of arguments to invoke the FUNCTION_NAME on, built in the create_geoshards step.
        self.shard_execution_args = dict()
        # used to record what the stitch targets are, source and dest
        self.stitch_targets = dict()

    def _apply_dynamic_replacements_to_ini(self):
        pattern = r'{{(.*?)}}'

        missing_sections = []
        for section, keys in self.REQUIRED_SECTIONS.items():
            if section == GeoSharding.INI_FILE_BASE:
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

        # Build a dict of all current config values for quick lookup
        config_dict = {
            (section, key): self.config[section][key]
            for section in self.config.sections()
            for key in self.config[section]
        }

        # we need to do multiple passes so we can allow chained references if desired
        something_changed = True
        while something_changed:
            something_changed = False
            for section in self.config.sections():
                for key in self.config[section]:
                    original_val = self.config[section][key]

                    def replacer(match):
                        var = match.group(1).strip()
                        for (conf_section, conf_key), conf_value in config_dict.items():
                            if conf_key.lower() == var.lower():
                                return conf_value
                        return match.group(0)

                    new_val = re.sub(pattern, replacer, original_val)
                    if new_val != original_val:
                        self.config[section][key] = new_val
                        config_dict[(section, key)] = new_val
                        something_changed = True

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

            subdirectory_path = GeoSharding._hash_to_subdirectory(
                job_id, n_subdirectory_levels, MAX_DIRECTORIES_PER_LEVEL)
            local_aoi_subset_dir = os.path.join(aoi_subset_dir, subdirectory_path, job_id)
            os.makedirs(local_aoi_subset_dir, exist_ok=True)

            aoi_subset_path = os.path.join(
                local_aoi_subset_dir, f'{job_id}.gpkg')
            if True or not os.path.exists(aoi_subset_path):
                job_id_prompt = f'{job_index} of {len(aoi_fid_index)}'
                self.task_graph.add_task(
                    func=GeoSharding._create_fid_subset,
                    args=(
                        job_id_prompt, self.aoi_path, fid_list, aoi_subset_path),
                    transient_run=True,
                    target_path_list=[aoi_subset_path],
                    task_name=job_id)
            aoi_path_area_list.append((area, aoi_subset_path))
            break

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
    def _create_fid_subset(
            job_id_prompt, base_vector_path, fid_list, target_vector_path,
            check_result=True):
        """Create subset of vector that matches fid list, projected into epsg."""
        intermediate_vector_path = '%s_tmp%s' % os.path.splitext(target_vector_path)
        for vector_path in [target_vector_path, intermediate_vector_path]:
            if os.path.exists(vector_path):
                os.remove(vector_path)
        layer_name = os.path.basename(os.path.splitext(target_vector_path)[0])
        vector = ogr.Open(base_vector_path)
        layer = vector.GetLayer()

        LOGGER.info(f'{job_id_prompt} subsetting {layer_name} ')
        # doing two step subsetting so we first extract then we try to fix the polygons
        extract_fids_cmd = [
            'ogr2ogr',
            '-f', 'GPKG',
            '-where', f'FID in ({", ".join(str(f) for f in fid_list)})',
            '-nln', layer_name,
            '-nlt', 'MULTIPOLYGON',
            intermediate_vector_path,
            base_vector_path
        ]
        subprocess.run(extract_fids_cmd, check=True)

        polygon_repair_cmd = [
            'ogr2ogr',
            '-f', 'GPKG',
            '-nln', layer_name,
            '-nlt', 'MULTIPOLYGON',
            '-makevalid',
            '-dialect', 'sqlite', '-sql', f'SELECT ST_Buffer(geom, 0) AS geometry FROM "{layer_name}"',
            target_vector_path,
            intermediate_vector_path
        ]
        subprocess.run(polygon_repair_cmd, check=True)

        os.remove(intermediate_vector_path)
        LOGGER.info(f'DONE with {job_id_prompt} subsetting {layer_name} ')

        if check_result:
            layer.SetAttributeFilter(
                f'"FID" in ('
                f'{", ".join([str(v) for v in fid_list])})')
            feature_count = layer.GetFeatureCount()
            target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
            target_layer = target_vector.GetLayer()
            if feature_count != target_layer.GetFeatureCount():
                raise ValueError(
                    f'expected {feature_count} in {target_vector_path} but got '
                    f'{target_layer.GetFeatureCount()}')
            target_layer = None
            target_vector = None
        layer = None
        vector = None

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

    @staticmethod
    def _replace_shard_refs(arg_dict, shard_dict):
        """
        Replace any remaining placeholders in config using shard_dict.
        """
        pattern = r'{{(.*?)}}'
        new_arg_dict = dict()
        for key, original_val in arg_dict.items():
            def replacer(match):
                var = match.group(1).strip()
                return str(shard_dict.get(var, match.group(0)))

            new_val = re.sub(pattern, replacer, original_val)
            new_arg_dict[key] = new_val
        return new_arg_dict

    def create_geoshards(self):
        """Split the config's spaital input based on the given aoi_subset_path."""

        for aoi_path in self.aoi_path_list:
            local_working_dir = os.path.dirname(aoi_path)
            shard_id = os.path.basename(os.path.splitext(aoi_path)[0])
            shard_replacement_dict = {
                GeoSharding.SHARD_ID: os.path.basename(os.path.splitext(aoi_path)[0]),
                GeoSharding.MULTI_AOI_IN_BATCH: None,
                GeoSharding.SHARD_AOI_PATH: aoi_path,
                GeoSharding.SHARD_WORKING_DIR: local_working_dir
            }
            base_raster_path_list = []
            warped_raster_path_list = []
            local_args = {}
            for key, base_raster_path in self.config[GeoSharding.SPATIAL_INPUT_SECTION].items():
                warped_raster_path = os.path.join(local_working_dir, os.path.basename(base_raster_path))
                base_raster_path_list.append(base_raster_path)
                warped_raster_path_list.append(warped_raster_path)
                local_args[key] = warped_raster_path
            warp_task = self.task_graph.add_task(
                func=GeoSharding._warp_raster_stack,
                args=(
                    base_raster_path_list,
                    warped_raster_path_list,
                    ['near'] * len(base_raster_path_list),
                    float(self.config[GeoSharding.PROJECTION_SECTION][GeoSharding.TARGET_PIXEL_SIZE]),
                    aoi_path),
                target_path_list=warped_raster_path_list,
                task_name=f'warp spatial inputs for {shard_id}')

            for key, base_argument in self.config[GeoSharding.NON_SPATIAL_INPUT_SECTION].items():
                local_args[key] = base_argument
            local_args = GeoSharding._replace_shard_refs(local_args, shard_replacement_dict)
            self.shard_execution_args[local_working_dir] = (warp_task, local_args, shard_replacement_dict)
            break

    def execute_on_shards(self):
        """Apply the FUNCTION section to each of the geoshards."""
        module = importlib.import_module(self.config[GeoSharding.FUNCTION_SECTION][GeoSharding.MODULE])
        func = getattr(module, self.config[GeoSharding.FUNCTION_SECTION][GeoSharding.FUNCTION_NAME])
        for shard_id, (task, args, _) in self.shard_execution_args.items():
            LOGGER.debug('executing shard_id')
            task.join()
            LOGGER.debug(args)
            func(args)

    @staticmethod
    def _warp_raster_stack(
            base_raster_path_list, warped_raster_path_list,
            resample_method_list, target_pixel_size, clip_vector_path):
        """Do an align of all the rasters but use a taskgraph to do it.

        Arguments are same as geoprocessing.align_and_resize_raster_stack.

        Allow for input rasters to be None.
        """
        clip_vector_info = geoprocessing.get_vector_info(clip_vector_path)
        # [minx, miny, maxx, maxy]
        buffered_clip_bounding_box = [
            clip_vector_info['bounding_box'][0] - 5*target_pixel_size,
            clip_vector_info['bounding_box'][1] - 5*target_pixel_size,
            clip_vector_info['bounding_box'][2] + 5*target_pixel_size,
            clip_vector_info['bounding_box'][3] + 5*target_pixel_size,
        ]
        for base_raster_path, warped_raster_path, resample_method in zip(
                base_raster_path_list, warped_raster_path_list,
                resample_method_list):
            if base_raster_path is None:
                continue
            base_raster_info = geoprocessing.get_raster_info(base_raster_path)
            working_dir = os.path.dirname(warped_raster_path)
            clipped_raster_path = '%s_clipped%s' % os.path.splitext(
                warped_raster_path)
            geoprocessing.warp_raster(
                base_raster_path, base_raster_info['pixel_size'],
                clipped_raster_path, resample_method,
                **{
                    'target_bb': buffered_clip_bounding_box,
                    'target_projection_wkt': base_raster_info['projection_wkt'],
                    'working_dir': working_dir,
                })

            vector_mask_options = {
                'mask_vector_path': clip_vector_path,
                'all_touched': True}
            geoprocessing.warp_raster(
                clipped_raster_path, (target_pixel_size, -target_pixel_size),
                warped_raster_path, resample_method,
                **{
                    'target_projection_wkt': clip_vector_info['projection_wkt'],
                    'vector_mask_options': vector_mask_options,
                    'working_dir': working_dir,
                })
            os.remove(clipped_raster_path)

    @staticmethod
    def get_bounding_box_and_projection(dataset_path):
        """
        Returns (minx, miny, maxx, maxy) for either a raster or vector dataset.
        """
        from ecoshard.geoprocessing import RASTER_TYPE, VECTOR_TYPE
        gis_type = geoprocessing.get_gis_type(dataset_path)
        if gis_type == RASTER_TYPE:
            get_fn = geoprocessing.get_raster_info
        elif gis_type == VECTOR_TYPE:
            get_fn = geoprocessing.get_vector_info
        else:
            raise ValueError(f'{dataset_path} does not appear to be a vector or raster')
        ds_info = get_fn(dataset_path)
        return ds_info['bounding_box'], ds_info['projection_wkt']

    def create_global_stitch_rasters(self):
        """Create the rasters that will be stitched into defined in EXPECTED_OUTPUT."""
        target_bb, target_projection_wkt = GeoSharding.get_bounding_box_and_projection(
            self.config[GeoSharding.EXPECTED_OUTPUT_SECTION][GeoSharding.TARGET_PROJECTION_AND_BB_SOURCE])
        target_pixel_size = float(
            self.config[GeoSharding.EXPECTED_OUTPUT_SECTION][GeoSharding.TARGET_PIXEL_SIZE])
        for raster_key, raster_target_local_str in self.config[GeoSharding.EXPECTED_OUTPUT_SECTION].items():
            if raster_key in [
                    GeoSharding.TARGET_PROJECTION_AND_BB_SOURCE,
                    GeoSharding.TARGET_PIXEL_SIZE]:
                continue
            LOGGER.debug(f'{raster_key}: {raster_target_local_str}')
            global_stitch_raster_path, local_source_path, target_nodata_str = raster_target_local_str.split(',')
            target_nodata = ast.literal_eval(target_nodata_str.strip())
            if not os.path.exists(global_stitch_raster_path):
                LOGGER.info(f'creating {global_stitch_raster_path}')
                driver = gdal.GetDriverByName('GTiff')
                n_cols = int((target_bb[2] - target_bb[0]) / target_pixel_size)
                n_rows = int((target_bb[3] - target_bb[1]) / target_pixel_size)
                LOGGER.info(f'**** creating raster of size {n_cols} by {n_rows}')
                target_raster = driver.Create(
                    global_stitch_raster_path,
                    n_cols, n_rows, 1,
                    gdal.GDT_Float32,
                    options=(
                        'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
                        'SPARSE_OK=TRUE', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
                target_raster.SetProjection(target_projection_wkt)
                target_raster.SetGeoTransform(
                    [target_bb[0], target_pixel_size, 0,
                     target_bb[3], 0, -target_pixel_size])
                target_band = target_raster.GetRasterBand(1)
                target_band.SetNoDataValue(target_nodata)
                target_raster = None
                target_band = None
            self.stitch_targets[raster_key] = (local_source_path, global_stitch_raster_path)

    def stitch_results(self):
        for raster_key, (template_local_raster_path, global_raster_path) in self.stitch_targets.items():
            for (_, local_args, shard_replacement_dict) in self.shard_execution_args.values():
                local_raster_path = GeoSharding._replace_shard_refs(
                    {'path': template_local_raster_path}, shard_replacement_dict)['path']
                local_args = GeoSharding._replace_shard_refs(local_args, shard_replacement_dict)
                global_raster = gdal.OpenEx(global_raster_path, gdal.GA_Update)
                global_band = global_raster.GetRasterBand(1)
                global_raster_info = geoprocessing.get_raster_info(global_raster_path)
                local_raster_info = geoprocessing.get_raster_info(local_raster_path)
                warp_options = gdal.WarpOptions(
                    dstSRS=global_raster_info['projection_wkt'],
                    xRes=abs(global_raster_info['geotransform'][1]),
                    yRes=abs(global_raster_info['geotransform'][5]),
                    srcNodata=local_raster_info['nodata'][0],
                    dstNodata=global_raster_info['nodata'][0],
                    format='MEM',
                    resampleAlg=gdal.GRA_NearestNeighbour
                )
                local_raster = gdal.OpenEx(local_raster_path)
                warped_local_raster = gdal.Warp('', local_raster, options=warp_options)
                warped_local_band = warped_local_raster.GetRasterBand(1)
                local_raster = None

                warped_gt = warped_local_raster.GetGeoTransform()
                warp_width = warped_local_raster.RasterXSize
                warp_height = warped_local_raster.RasterYSize

                global_inv_gt = gdal.InvGeoTransform(global_raster_info['geotransform'])
                px, py = [
                    int(numpy.round(v))
                    for v in gdal.ApplyGeoTransform(
                        global_inv_gt,
                        warped_gt[0],
                        warped_gt[3])]

                global_array = global_band.ReadAsArray(
                    px, py,
                    warp_width, warp_height)
                global_nodata = global_band.GetNoDataValue()
                if global_nodata is not None:
                    global_nodata_mask = global_array == global_nodata
                else:
                    global_nodata_mask = numpy.ones(global_array.shape, dtype=bool)
                local_array = warped_local_band.ReadAsArray()
                local_nodata = warped_local_band.GetNoDataValue()
                warped_local_band = None
                warped_local_raster = None
                if local_nodata is not None:
                    global_nodata_mask &= local_array != local_nodata
                global_array[global_nodata_mask] = local_array[global_nodata_mask]
                local_array = None
                global_band.WriteArray(global_array, px, py)
                global_band = None
                global_raster = None


def run_pipeline(config_path):
    """Execute the geosplitter pipeline with the config_path ini file."""
    geosharder = GeoSharding(config_path)
    geosharder.create_geoshards()
    #geosharder.execute_on_shards()
    geosharder.create_global_stitch_rasters()
    geosharder.stitch_results()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the GeoSharding pipeline.")
    parser.add_argument("config_file", help="Path to the INI configuration file.")
    args = parser.parse_args()
    run_pipeline(args.config_file)

"""Tracer for running convolution tests."""
import logging
import time

from ecoshard import geoprocessing
from osgeo import gdal
import numpy

gdal.SetCacheMax(2**27)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


def predict_bounds(signal_offset, kernel_offset, n_cols_signal, n_rows_signal, n_cols_kernel, n_rows_kernel):
    # Add result to current output to account for overlapping edges
    left_index_raster = (
        signal_offset['xoff'] - n_cols_kernel // 2 +
        kernel_offset['xoff'])
    right_index_raster = (
        signal_offset['xoff'] - n_cols_kernel // 2 +
        kernel_offset['xoff'] + signal_offset['win_xsize'] +
        kernel_offset['win_xsize'] - 1)
    top_index_raster = (
        signal_offset['yoff'] - n_rows_kernel // 2 +
        kernel_offset['yoff'])
    bottom_index_raster = (
        signal_offset['yoff'] - n_rows_kernel // 2 +
        kernel_offset['yoff'] + signal_offset['win_ysize'] +
        kernel_offset['win_ysize'] - 1)

    # we might abut the edge of the raster, clip if so
    if left_index_raster < 0:
        left_index_raster = 0
    if top_index_raster < 0:
        top_index_raster = 0
    if right_index_raster > n_cols_signal:
        right_index_raster = n_cols_signal
    if bottom_index_raster > n_rows_signal:
        bottom_index_raster = n_rows_signal

    index_dict = {
        'xoff': left_index_raster,
        'yoff': top_index_raster,
        'win_xsize': right_index_raster-left_index_raster,
        'win_ysize': bottom_index_raster-top_index_raster
    }


def main():
    """Entry point."""
    # raster_path = r"D:\ecoshard\fc_stack\fc_stack_hansen_forest_cover_2000-2020_md5_fbb58a.tif"
    # raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    # band = raster.GetRasterBand(1)

    LOGGER.debug('create rasters')
    signal_array = numpy.full((20000, 20000), 2)
    kernel_array = numpy.zeros((500, 500))
    offset = 50
    for array in [signal_array, kernel_array]:
        array[
            array.shape[0]//2-offset:array.shape[0]//2+offset,
            array.shape[1]//2-offset:array.shape[1]//2+offset] = 1.0
    signal_path = 'signal.tif'
    kernel_path = 'kernel.tif'

    geoprocessing.numpy_array_to_raster(
        signal_array, -1, (1, -1), (0, 0), None, signal_path)
    geoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1, -1), (0, 0), None, kernel_path)

    target_path = 'convolve2d.tif'
    LOGGER.debug('starting convolve')
    start_time = time.time()
    geoprocessing.convolve_2d(
        (signal_path, 1), (kernel_path, 1), target_path,
        working_dir='convolve_working_dir', largest_block=2**22)
    LOGGER.debug('all done')
    LOGGER.debug(f'took {time.time()-start_time}s')


if __name__ == '__main__':
    main()

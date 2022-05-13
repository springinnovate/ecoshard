"""Tracer for running convolution tests."""
import logging

from ecoshard import geoprocessing
from osgeo import gdal
import numpy


logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


def main():
    """Entry point."""
    raster_path = r"D:\ecoshard\fc_stack\fc_stack_hansen_forest_cover_2000-2020_md5_fbb58a.tif"
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(1)
    LOGGER.debug(band)

    signal_array = numpy.zeros((4000, 4000))
    kernel_array = numpy.zeros((100, 100))
    signal_path = 'signal.tif'
    kernel_path = 'kernel.tif'
    geoprocessing.numpy_array_to_raster(
        signal_array, -1, (1, -1), (0, 0), None, signal_path)
    geoprocessing.numpy_array_to_raster(
        kernel_array, -1, (1, -1), (0, 0), None, kernel_path)

    target_path = 'convolve2d.tif'
    LOGGER.debug('starting convolve')
    geoprocessing.convolve_2d(
        (signal_path, 1), (kernel_path, 1), target_path,
        working_dir='convolve_working_dir')
    LOGGER.debug('all done')


if __name__ == '__main__':
    main()

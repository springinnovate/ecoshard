from osgeo import osr
import os
from osgeo import gdal
import logging

import numpy
from ecoshard import geoprocessing
import ecoshard

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(__name__)


def main():
    # make raster
    raster_path = 'single_pixel.tif'
    geoprocessing.numpy_array_to_raster(
        numpy.array([[1]]), 0, (1, -1), (0, 0), None,
        'single_pixel.tif')

    # ecoshard.add_metadata(raster_path, {
    #     'RSTITLE': 'single pixel',
    #     'RSABSTR': 'test medatadata',
    #     }, 'GEO_METADATA')

    ecoshard.add_metadata(raster_path, {
        'TIFFTAG_DOCUMENTNAME': 'single pixel',
        })

    metadata = ecoshard.get_metadata(raster_path)
    LOGGER.debug(f'metadata: {metadata}')
    raster = gdal.OpenEx(raster_path)
    LOGGER.debug(raster.GetMetadataDomainList())

    return


if __name__ == '__main__':
    main()

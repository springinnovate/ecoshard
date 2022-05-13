"""Tracer for running convolution tests."""
import logging

from osgeo import gdal

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


if __name__ == '__main__':
    main()

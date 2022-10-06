from osgeo import osr
import os
from osgeo import gdal
import logging

import numpy
from ecoshard import geoprocessing
import time

gdal.SetCacheMax(120)

DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTiff', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS'))

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

# TODO: before doing any parallelization
def local_op(array):
    return (array ** 2 + array **0.52)**3.1*(array ** 2 + array **0.52)**3.1


def make_big_raster(target_path):
    factor = 10
    n_cols = 12800*factor
    n_rows = 6400*factor
    driver = gdal.GetDriverByName('GTiff')
    target_raster = driver.Create(
        target_path, n_cols, n_rows, 1, gdal.GDT_Float32,
        options=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS[1])
    target_raster.FlushCache()
    target_band = target_raster.GetRasterBand(1)

    # how the fill is progressing.
    for offsets in geoprocessing.iterblocks((target_path, 1), offset_only=True):
        fill_array = numpy.random.random(
            (offsets['win_ysize'], offsets['win_xsize']))
        target_band.WriteArray(
            fill_array, offsets['xoff'], offsets['yoff'])
    target_band = None
    target_raster = None


def main():
    # make raster
    start_time = time.time()
    input_path = 'big.tif'
    #make_big_raster(input_path)
    print(f'made it {time.time()-start_time}')

    test_raster_calc = True
    test_convolve = False

    if test_raster_calc:
        print('test raster calc')
        target_raster_path = 'big_out.tif'
        start_time = time.time()
        geoprocessing.raster_calculator(
            [(input_path, 1)], local_op, target_raster_path,
            gdal.GDT_Float32, None,
            calc_raster_stats=True,
            raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)
        print(f'{time.time()-start_time:.2f}')

    if test_convolve:
        kernel = numpy.random.random((2500, 2500))
        geoprocessing.numpy_array_to_raster(
            kernel, None, (1, -1), (0, 0), None,
            'kernel.tif')

        for power in range(22, 28):
            start_time = time.time()
            largest_block = 2**power
            geoprocessing.convolve_2d(
                (input_path, 1), ('kernel.tif', 1), 'convolve.tif',
                largest_block=largest_block,
                raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS)
            print(f'{time.time()-start_time:.2f},{power}')
    return

if __name__ == '__main__':
    main()

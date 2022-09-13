# coding=UTF-8
# distutils: language=c++
# cython: language_level=3
import logging
import multiprocessing
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
import zlib

cimport cython
cimport libcpp.algorithm
cimport numpy
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.stdio cimport fclose
from libc.stdio cimport FILE
from libc.stdio cimport fopen
from libc.stdio cimport fread
from libc.stdio cimport fwrite
from libcpp.vector cimport vector
from osgeo import gdal
from osgeo import osr
import numpy
import ecoshard.geoprocessing

cimport cython
cimport numpy
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from libc.time cimport time as ctime
from libc.time cimport time_t
from libcpp.deque cimport deque
from libcpp.list cimport list as clist
from libcpp.pair cimport pair
from libcpp.queue cimport queue
from libcpp.set cimport set as cset
from libcpp.stack cimport stack
from libcpp.vector cimport vector
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import numpy
import shapely.wkb
import shapely.ops
import scipy.stats

# Number of raster blocks to hold in memory at once per Managed Raster
cdef int MANAGED_RASTER_N_BLOCKS = 2**6

DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS = ('GTIFF', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256', 'NUM_THREADS=ALL_CPUS'))

# In GDAL 3.0 spatial references no longer ignore Geographic CRS Axis Order
# and conform to Lat first, Lon Second. Transforms expect (lat, lon) order
# as opposed to the GIS friendly (lon, lat). See
# https://trac.osgeo.org/gdal/wiki/rfc73_proj6_wkt2_srsbarn Axis order
# issues. SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER) swaps the
# axis order, which will use Lon,Lat order for Geographic CRS, but otherwise
# leaves Projected CRS alone
DEFAULT_OSR_AXIS_MAPPING_STRATEGY = osr.OAMS_TRADITIONAL_GIS_ORDER

LOGGER = logging.getLogger('ecoshard.geoprocessing.geoprocessing_core')

cdef float _NODATA = -1.0

# this is a least recently used cache written in C++ in an external file,
# exposing here so _ManagedRaster can use it
cdef extern from "./routing/LRUCache.h" nogil:
    cdef cppclass LRUCache[KEY_T, VAL_T]:
        LRUCache(int)
        void put(KEY_T&, VAL_T&, clist[pair[KEY_T,VAL_T]]&)
        clist[pair[KEY_T,VAL_T]].iterator begin()
        clist[pair[KEY_T,VAL_T]].iterator end()
        bint exist(KEY_T &)
        VAL_T get(KEY_T &)
        void clean(clist[pair[KEY_T,VAL_T]]&, int n_items)
        size_t size()

# this ctype is used to store the block ID and the block buffer as one object
# inside Managed Raster
ctypedef pair[int, double*] BlockBufferPair



cdef extern from "FastFileIterator.h" nogil:
    cdef cppclass FastFileIterator[DATA_T]:
        FastFileIterator(const char*, size_t)
        DATA_T next()
        size_t size()
    int FastFileIteratorCompare[DATA_T](FastFileIterator[DATA_T]*,
                                        FastFileIterator[DATA_T]*)


cdef extern from "CoordFastFileIterator.h" nogil:
    cdef cppclass CoordFastFileIterator[DATA_T]:
        CoordFastFileIterator(const char*, const char*, const char*, size_t)
        DATA_T next()
        long long coord()
        double area()
        size_t size()
    int CoordFastFileIteratorCompare[DATA_T](
            CoordFastFileIterator[DATA_T]*,
            CoordFastFileIterator[DATA_T]*)

# This resolves an issue on Mac OS X Catalina where cimporting ``push_heap``
# and ``pop_heap`` from the Standard Library would cause compilation to fail
# with an error message about the candidate function template not being
# viable.  The SO answer to a related question
# (https://stackoverflow.com/a/57586789/299084) suggests a workaround: don't
# tell Cython that we have a template function.  Using ``...`` here allows
# us to not have to specify all of the types for which we need a working
# ``push_heap`` and ``pop_heap``.
cdef extern from "<algorithm>" namespace "std":
    void push_heap(...)
    void pop_heap(...)

cdef class _ManagedRaster:
    cdef LRUCache[int, double*]* lru_cache
    cdef cset[int] dirty_blocks
    cdef int block_xsize
    cdef int block_ysize
    cdef int block_xmod
    cdef int block_ymod
    cdef int block_xbits
    cdef int block_ybits
    cdef int raster_x_size
    cdef int raster_y_size
    cdef int block_nx
    cdef int block_ny
    cdef int write_mode
    cdef bytes raster_path
    cdef int band_id
    cdef int closed

    def __cinit__(self, raster_path, band_id, write_mode):
        """Create new instance of Managed Raster.

        Parameters:
            raster_path (char*): path to raster that has block sizes that are
                powers of 2. If not, an exception is raised.
            band_id (int): which band in `raster_path` to index. Uses GDAL
                notation that starts at 1.
            write_mode (boolean): if true, this raster is writable and dirty
                memory blocks will be written back to the raster as blocks
                are swapped out of the cache or when the object deconstructs.

        Returns:
            None.
        """
        if not os.path.isfile(raster_path):
            LOGGER.error("%s is not a file.", raster_path)
            return
        raster_info = ecoshard.geoprocessing.get_raster_info(raster_path)
        self.raster_x_size, self.raster_y_size = raster_info['raster_size']
        self.block_xsize, self.block_ysize = raster_info['block_size']
        self.block_xmod = self.block_xsize-1
        self.block_ymod = self.block_ysize-1

        if not (1 <= band_id <= raster_info['n_bands']):
            err_msg = (
                "Error: band ID (%s) is not a valid band number. "
                "This exception is happening in Cython, so it will cause a "
                "hard seg-fault, but it's otherwise meant to be a "
                "ValueError." % (band_id))
            print(err_msg)
            raise ValueError(err_msg)
        self.band_id = band_id

        if (self.block_xsize & (self.block_xsize - 1) != 0) or (
                self.block_ysize & (self.block_ysize - 1) != 0):
            # If inputs are not a power of two, this will at least print
            # an error message. Unfortunately with Cython, the exception will
            # present itself as a hard seg-fault, but I'm leaving the
            # ValueError in here at least for readability.
            err_msg = (
                "Error: Block size is not a power of two: "
                "block_xsize: %d, %d, %s. This exception is happening"
                "in Cython, so it will cause a hard seg-fault, but it's"
                "otherwise meant to be a ValueError." % (
                    self.block_xsize, self.block_ysize, raster_path))
            print(err_msg)
            raise ValueError(err_msg)

        self.block_xbits = numpy.log2(self.block_xsize)
        self.block_ybits = numpy.log2(self.block_ysize)
        self.block_nx = (
            self.raster_x_size + (self.block_xsize) - 1) // self.block_xsize
        self.block_ny = (
            self.raster_y_size + (self.block_ysize) - 1) // self.block_ysize

        self.lru_cache = new LRUCache[int, double*](MANAGED_RASTER_N_BLOCKS)
        self.raster_path = <bytes> raster_path
        self.write_mode = write_mode
        self.closed = 0

    def __dealloc__(self):
        """Deallocate _ManagedRaster.

        This operation manually frees memory from the LRUCache and writes any
        dirty memory blocks back to the raster if `self.write_mode` is True.
        """
        self.close()

    def close(self):
        """Close the _ManagedRaster and free up resources.

            This call writes any dirty blocks to disk, frees up the memory
            allocated as part of the cache, and frees all GDAL references.

            Any subsequent calls to any other functions in _ManagedRaster will
            have undefined behavior.
        """
        if self.closed:
            return
        self.closed = 1
        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array = numpy.empty(
            (self.block_ysize, self.block_xsize))
        cdef double *double_buffer
        cdef int block_xi
        cdef int block_yi
        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize
        cdef int win_ysize

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff
        cdef int yoff

        cdef clist[BlockBufferPair].iterator it = self.lru_cache.begin()
        cdef clist[BlockBufferPair].iterator end = self.lru_cache.end()
        if not self.write_mode:
            while it != end:
                # write the changed value back if desired
                PyMem_Free(deref(it).second)
                inc(it)
            return

        raster = gdal.OpenEx(
            self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)

        # if we get here, we're in write_mode
        cdef cset[int].iterator dirty_itr
        while it != end:
            double_buffer = deref(it).second
            block_index = deref(it).first

            # write to disk if block is dirty
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr != self.dirty_blocks.end():
                self.dirty_blocks.erase(dirty_itr)
                block_xi = block_index % self.block_nx
                block_yi = block_index // self.block_nx

                # we need the offsets to subtract from global indexes for
                # cached array
                xoff = block_xi << self.block_xbits
                yoff = block_yi << self.block_ybits

                win_xsize = self.block_xsize
                win_ysize = self.block_ysize

                # clip window sizes if necessary
                if xoff+win_xsize > self.raster_x_size:
                    win_xsize = win_xsize - (
                        xoff+win_xsize - self.raster_x_size)
                if yoff+win_ysize > self.raster_y_size:
                    win_ysize = win_ysize - (
                        yoff+win_ysize - self.raster_y_size)

                for xi_copy in range(win_xsize):
                    for yi_copy in range(win_ysize):
                        block_array[yi_copy, xi_copy] = (
                            double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy])
                raster_band.WriteArray(
                    block_array[0:win_ysize, 0:win_xsize],
                    xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            inc(it)
        raster_band.FlushCache()
        raster_band = None
        raster = None

    cdef inline void set(self, int xi, int yi, double value):
        """Set the pixel at `xi,yi` to `value`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))] = value
        if self.write_mode:
            dirty_itr = self.dirty_blocks.find(block_index)
            if dirty_itr == self.dirty_blocks.end():
                self.dirty_blocks.insert(block_index)

    cdef inline double get(self, int xi, int yi):
        """Return the value of the pixel at `xi,yi`."""
        if xi < 0 or xi >= self.raster_x_size:
            LOGGER.error("x out of bounds %s" % xi)
        if yi < 0 or yi >= self.raster_y_size:
            LOGGER.error("y out of bounds %s" % yi)
        cdef int block_xi = xi >> self.block_xbits
        cdef int block_yi = yi >> self.block_ybits
        # this is the flat index for the block
        cdef int block_index = block_yi * self.block_nx + block_xi
        if not self.lru_cache.exist(block_index):
            self._load_block(block_index)
        return self.lru_cache.get(
            block_index)[
                ((yi & (self.block_ymod)) << self.block_xbits) +
                (xi & (self.block_xmod))]

    cdef void _load_block(self, int block_index) except *:
        cdef int block_xi = block_index % self.block_nx
        cdef int block_yi = block_index // self.block_nx

        # we need the offsets to subtract from global indexes for cached array
        cdef int xoff = block_xi << self.block_xbits
        cdef int yoff = block_yi << self.block_ybits

        cdef int xi_copy, yi_copy
        cdef numpy.ndarray[double, ndim=2] block_array
        cdef double *double_buffer
        cdef clist[BlockBufferPair] removed_value_list

        # determine the block aligned xoffset for read as array

        # initially the win size is the same as the block size unless
        # we're at the edge of a raster
        cdef int win_xsize = self.block_xsize
        cdef int win_ysize = self.block_ysize

        # load a new block
        if xoff+win_xsize > self.raster_x_size:
            win_xsize = win_xsize - (xoff+win_xsize - self.raster_x_size)
        if yoff+win_ysize > self.raster_y_size:
            win_ysize = win_ysize - (yoff+win_ysize - self.raster_y_size)

        raster = gdal.OpenEx(self.raster_path, gdal.OF_RASTER)
        raster_band = raster.GetRasterBand(self.band_id)
        block_array = raster_band.ReadAsArray(
            xoff=xoff, yoff=yoff, win_xsize=win_xsize,
            win_ysize=win_ysize).astype(numpy.float64)
        raster_band = None
        raster = None
        double_buffer = <double*>PyMem_Malloc(
            (sizeof(double) << self.block_xbits) * win_ysize)
        for xi_copy in range(win_xsize):
            for yi_copy in range(win_ysize):
                double_buffer[(yi_copy << self.block_xbits)+xi_copy] = (
                    block_array[yi_copy, xi_copy])
        self.lru_cache.put(
            <int>block_index, <double*>double_buffer, removed_value_list)

        if self.write_mode:
            n_attempts = 5
            while True:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    if n_attempts == 0:
                        raise RuntimeError(
                            f'could not open {self.raster_path} for writing')
                    LOGGER.warning(
                        f'opening {self.raster_path} resulted in null, '
                        f'trying {n_attempts} more times.')
                    n_attempts -= 1
                    time.sleep(0.5)
                raster_band = raster.GetRasterBand(self.band_id)
                break

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            raster_band = None
            raster = None

    cdef void flush(self) except *:
        cdef clist[BlockBufferPair] removed_value_list
        cdef double *double_buffer
        cdef cset[int].iterator dirty_itr
        cdef int block_index, block_xi, block_yi
        cdef int xoff, yoff, win_xsize, win_ysize

        self.lru_cache.clean(removed_value_list, 0)

        raster_band = None
        if self.write_mode:
            max_retries = 5
            while max_retries > 0:
                raster = gdal.OpenEx(
                    self.raster_path, gdal.GA_Update | gdal.OF_RASTER)
                if raster is None:
                    max_retries -= 1
                    LOGGER.error(
                        f'unable to open {self.raster_path}, retrying...')
                    time.sleep(0.2)
                    continue
                break
            if max_retries == 0:
                raise ValueError(
                    f'unable to open {self.raster_path} in '
                    'ManagedRaster.flush')
            raster_band = raster.GetRasterBand(self.band_id)

        block_array = numpy.empty(
            (self.block_ysize, self.block_xsize), dtype=numpy.double)
        while not removed_value_list.empty():
            # write the changed value back if desired
            double_buffer = removed_value_list.front().second

            if self.write_mode:
                block_index = removed_value_list.front().first

                # write back the block if it's dirty
                dirty_itr = self.dirty_blocks.find(block_index)
                if dirty_itr != self.dirty_blocks.end():
                    self.dirty_blocks.erase(dirty_itr)

                    block_xi = block_index % self.block_nx
                    block_yi = block_index // self.block_nx

                    xoff = block_xi << self.block_xbits
                    yoff = block_yi << self.block_ybits

                    win_xsize = self.block_xsize
                    win_ysize = self.block_ysize

                    if xoff+win_xsize > self.raster_x_size:
                        win_xsize = win_xsize - (
                            xoff+win_xsize - self.raster_x_size)
                    if yoff+win_ysize > self.raster_y_size:
                        win_ysize = win_ysize - (
                            yoff+win_ysize - self.raster_y_size)

                    for xi_copy in range(win_xsize):
                        for yi_copy in range(win_ysize):
                            block_array[yi_copy, xi_copy] = double_buffer[
                                (yi_copy << self.block_xbits) + xi_copy]
                    raster_band.WriteArray(
                        block_array[0:win_ysize, 0:win_xsize],
                        xoff=xoff, yoff=yoff)
            PyMem_Free(double_buffer)
            removed_value_list.pop_front()

        if self.write_mode:
            LOGGER.debug('flushing....')
            raster_band.FlushCache()
            raster_band = None
            raster = None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def _distance_transform_edt(
        region_raster_path, g_raster_path, float sample_d_x,
        float sample_d_y, target_distance_raster_path,
        raster_driver_creation_tuple):
    """Calculate the euclidean distance transform on base raster.

    Calculates the euclidean distance transform on the base raster in units of
    pixels multiplied by an optional scalar constant. The implementation is
    based off the algorithm described in:  Meijster, Arnold, Jos BTM Roerdink,
    and Wim H. Hesselink. "A general algorithm for computing distance
    transforms in linear time." Mathematical Morphology and its applications
    to image and signal processing. Springer, Boston, MA, 2002. 331-340.

    The base mask raster represents the area to distance transform from as
    any pixel that is not 0 or nodata. It is computationally convenient to
    calculate the distance transform on the entire raster irrespective of
    nodata placement and thus produces a raster that will have distance
    transform values even in pixels that are nodata in the base.

    Parameters:
        region_raster_path (string): path to a byte raster where region pixels
            are indicated by a 1 and 0 otherwise.
        g_raster_path (string): path to a raster created by this call that
            is used as the intermediate "g" variable described in Meijster
            et. al.
        sample_d_x (float):
        sample_d_y (float):
            These parameters scale the pixel distances when calculating the
            distance transform. ``d_x`` is the x direction when changing a
            column index, and ``d_y`` when changing a row index. Both values
            must be > 0.
        target_distance_raster_path (string): path to the target raster
            created by this call that is the exact euclidean distance
            transform from any pixel in the base raster that is not nodata and
            not 0. The units are in (pixel distance * sampling_distance).
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        None

    """
    cdef int yoff, row_index, block_ysize, win_ysize, n_rows
    cdef int xoff, block_xsize, win_xsize, n_cols
    cdef int q_index, local_x_index, local_y_index, u_index
    cdef int tq, sq
    cdef float gu, gsq, w
    cdef numpy.ndarray[numpy.float32_t, ndim=2] g_block
    cdef numpy.ndarray[numpy.int32_t, ndim=1] s_array
    cdef numpy.ndarray[numpy.int32_t, ndim=1] t_array
    cdef numpy.ndarray[numpy.float32_t, ndim=2] dt
    cdef numpy.ndarray[numpy.int8_t, ndim=2] mask_block

    mask_raster = gdal.OpenEx(region_raster_path, gdal.OF_RASTER)
    mask_band = mask_raster.GetRasterBand(1)

    n_cols = mask_raster.RasterXSize
    n_rows = mask_raster.RasterYSize

    raster_info = ecoshard.geoprocessing.get_raster_info(region_raster_path)
    ecoshard.geoprocessing.new_raster_from_base(
        region_raster_path, g_raster_path, gdal.GDT_Float32, [_NODATA],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    g_raster = gdal.OpenEx(g_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    g_band = g_raster.GetRasterBand(1)
    g_band_blocksize = g_band.GetBlockSize()

    # normalize the sample distances so we don't get a strange numerical
    # overflow
    max_sample = max(sample_d_x, sample_d_y)
    sample_d_x /= max_sample
    sample_d_y /= max_sample

    # distances can't be larger than half the perimeter of the raster.
    cdef float numerical_inf = max(sample_d_x, 1.0) * max(sample_d_y, 1.0) * (
        raster_info['raster_size'][0] + raster_info['raster_size'][1])
    # scan 1
    done = False
    block_xsize = raster_info['block_size'][0]
    mask_block = numpy.empty((n_rows, block_xsize), dtype=numpy.int8)
    g_block = numpy.empty((n_rows, block_xsize), dtype=numpy.float32)
    for xoff in numpy.arange(0, n_cols, block_xsize):
        win_xsize = block_xsize
        if xoff + win_xsize > n_cols:
            win_xsize = n_cols - xoff
            mask_block = numpy.empty((n_rows, win_xsize), dtype=numpy.int8)
            g_block = numpy.empty((n_rows, win_xsize), dtype=numpy.float32)
            done = True
        mask_band.ReadAsArray(
            xoff=xoff, yoff=0, win_xsize=win_xsize, win_ysize=n_rows,
            buf_obj=mask_block)
        # base case
        g_block[0, :] = (mask_block[0, :] == 0) * numerical_inf
        for row_index in range(1, n_rows):
            for local_x_index in range(win_xsize):
                if mask_block[row_index, local_x_index] == 1:
                    g_block[row_index, local_x_index] = 0
                else:
                    g_block[row_index, local_x_index] = (
                        g_block[row_index-1, local_x_index] + sample_d_y)
        for row_index in range(n_rows-2, -1, -1):
            for local_x_index in range(win_xsize):
                if (g_block[row_index+1, local_x_index] <
                        g_block[row_index, local_x_index]):
                    g_block[row_index, local_x_index] = (
                        sample_d_y + g_block[row_index+1, local_x_index])
        g_band.WriteArray(g_block, xoff=xoff, yoff=0)
        if done:
            break
    g_band.FlushCache()

    cdef float distance_nodata = -1.0

    ecoshard.geoprocessing.new_raster_from_base(
        region_raster_path, target_distance_raster_path.encode('utf-8'),
        gdal.GDT_Float32, [distance_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    target_distance_raster = gdal.OpenEx(
        target_distance_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_distance_band = target_distance_raster.GetRasterBand(1)

    LOGGER.info('Distance Transform Phase 2')
    s_array = numpy.empty(n_cols, dtype=numpy.int32)
    t_array = numpy.empty(n_cols, dtype=numpy.int32)

    done = False
    block_ysize = g_band_blocksize[1]
    g_block = numpy.empty((block_ysize, n_cols), dtype=numpy.float32)
    dt = numpy.empty((block_ysize, n_cols), dtype=numpy.float32)
    mask_block = numpy.empty((block_ysize, n_cols), dtype=numpy.int8)
    sq = 0  # initialize so compiler doesn't complain
    gsq = 0
    for yoff in numpy.arange(0, n_rows, block_ysize):
        win_ysize = block_ysize
        if yoff + win_ysize >= n_rows:
            win_ysize = n_rows - yoff
            g_block = numpy.empty((win_ysize, n_cols), dtype=numpy.float32)
            mask_block = numpy.empty((win_ysize, n_cols), dtype=numpy.int8)
            dt = numpy.empty((win_ysize, n_cols), dtype=numpy.float32)
            done = True
        g_band.ReadAsArray(
            xoff=0, yoff=yoff, win_xsize=n_cols, win_ysize=win_ysize,
            buf_obj=g_block)
        mask_band.ReadAsArray(
            xoff=0, yoff=yoff, win_xsize=n_cols, win_ysize=win_ysize,
            buf_obj=mask_block)
        for local_y_index in range(win_ysize):
            q_index = 0
            s_array[0] = 0
            t_array[0] = 0
            for u_index in range(1, n_cols):
                gu = g_block[local_y_index, u_index]**2
                while (q_index >= 0):
                    tq = t_array[q_index]
                    sq = s_array[q_index]
                    gsq = g_block[local_y_index, sq]**2
                    if ((sample_d_x*(tq-sq))**2 + gsq <= (
                            sample_d_x*(tq-u_index))**2 + gu):
                        break
                    q_index -= 1
                if q_index < 0:
                    q_index = 0
                    s_array[0] = u_index
                    sq = u_index
                    gsq = g_block[local_y_index, sq]**2
                else:
                    w = (float)(sample_d_x + ((
                        (sample_d_x*u_index)**2 - (sample_d_x*sq)**2 +
                        gu - gsq) / (2*sample_d_x*(u_index-sq))))
                    if w < n_cols*sample_d_x:
                        q_index += 1
                        s_array[q_index] = u_index
                        t_array[q_index] = <int>(w / sample_d_x)

            sq = s_array[q_index]
            gsq = g_block[local_y_index, sq]**2
            tq = t_array[q_index]
            for u_index in range(n_cols-1, -1, -1):
                if mask_block[local_y_index, u_index] != 1:
                    dt[local_y_index, u_index] = (
                        sample_d_x*(u_index-sq))**2+gsq
                else:
                    dt[local_y_index, u_index] = 0
                if u_index <= tq:
                    q_index -= 1
                    if q_index >= 0:
                        sq = s_array[q_index]
                        gsq = g_block[local_y_index, sq]**2
                        tq = t_array[q_index]

        valid_mask = g_block != _NODATA
        # "unnormalize" distances along with square root
        dt[valid_mask] = numpy.sqrt(dt[valid_mask]) * max_sample
        dt[~valid_mask] = _NODATA
        target_distance_band.WriteArray(dt, xoff=0, yoff=yoff)

        # we do this in the case where the blocksize is many times larger than
        # the raster size so we don't re-loop through the only block
        if done:
            break

    target_distance_band.ComputeStatistics(0)
    target_distance_band.FlushCache()
    target_distance_band = None
    mask_band = None
    g_band = None
    target_distance_raster = None
    mask_raster = None
    g_raster = None

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def calculate_slope(
        base_elevation_raster_path_band, target_slope_path,
        raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS):
    """Create a percent slope raster from DEM raster.

    Base algorithm is from Zevenbergen & Thorne "Quantitative Analysis of Land
    Surface Topography" 1987 although it has been modified to include the
    diagonal pixels by classic finite difference analysis.

    For the following notation, we define each pixel's DEM value by a letter
    with this spatial scheme::

        a b c
        d e f
        g h i

    Then the slope at ``e`` is defined at ``([dz/dx]^2 + [dz/dy]^2)^0.5``

    Where::

        [dz/dx] = ((c+2f+i)-(a+2d+g)/(8*x_cell_size)
        [dz/dy] = ((g+2h+i)-(a+2b+c))/(8*y_cell_size)

    In cases where a cell is nodata, we attempt to use the middle cell inline
    with the direction of differentiation (either in x or y direction).  If
    no inline pixel is defined, we use ``e`` and multiply the difference by
    ``2^0.5`` to account for the diagonal projection.

    Parameters:
        base_elevation_raster_path_band (string): a path/band tuple to a
            raster of height values. (path_to_raster, band_index)
        target_slope_path (string): path to target slope raster; will be a
            32 bit float GeoTIFF of same size/projection as calculate slope
            with units of percent slope.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Returns:
        ``None``
    """
    cdef numpy.npy_float64 a, b, c, d, e, f, g, h, i, dem_nodata
    cdef numpy.npy_float64 x_cell_size, y_cell_size,
    cdef numpy.npy_float64 dzdx_accumulator, dzdy_accumulator
    cdef int row_index, col_index, n_rows, n_cols,
    cdef int x_denom_factor, y_denom_factor, win_xsize, win_ysize
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dem_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] slope_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdx_array
    cdef numpy.ndarray[numpy.npy_float64, ndim=2] dzdy_array

    dem_raster = gdal.OpenEx(base_elevation_raster_path_band[0])
    dem_band = dem_raster.GetRasterBand(base_elevation_raster_path_band[1])
    dem_info = ecoshard.geoprocessing.get_raster_info(
        base_elevation_raster_path_band[0])
    raw_nodata = dem_info['nodata'][0]
    if raw_nodata is None:
        # if nodata is undefined, choose most negative 32 bit float
        raw_nodata = numpy.finfo(numpy.float32).min
    dem_nodata = raw_nodata
    x_cell_size, y_cell_size = dem_info['pixel_size']
    n_cols, n_rows = dem_info['raster_size']
    cdef numpy.npy_float64 slope_nodata = numpy.finfo(numpy.float32).min
    ecoshard.geoprocessing.new_raster_from_base(
        base_elevation_raster_path_band[0], target_slope_path,
        gdal.GDT_Float32, [slope_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple)
    target_slope_raster = gdal.OpenEx(target_slope_path, gdal.GA_Update)
    target_slope_band = target_slope_raster.GetRasterBand(1)

    for block_offset in ecoshard.geoprocessing.iterblocks(
            base_elevation_raster_path_band, offset_only=True):
        block_offset_copy = block_offset.copy()
        # try to expand the block around the edges if it fits
        x_start = 1
        win_xsize = block_offset['win_xsize']
        x_end = win_xsize+1
        y_start = 1
        win_ysize = block_offset['win_ysize']
        y_end = win_ysize+1

        if block_offset['xoff'] > 0:
            block_offset_copy['xoff'] -= 1
            block_offset_copy['win_xsize'] += 1
            x_start -= 1
        if block_offset['xoff']+win_xsize < n_cols:
            block_offset_copy['win_xsize'] += 1
            x_end += 1
        if block_offset['yoff'] > 0:
            block_offset_copy['yoff'] -= 1
            block_offset_copy['win_ysize'] += 1
            y_start -= 1
        if block_offset['yoff']+win_ysize < n_rows:
            block_offset_copy['win_ysize'] += 1
            y_end += 1

        dem_array = numpy.empty(
            (win_ysize+2, win_xsize+2),
            dtype=numpy.float64)
        dem_array[:] = dem_nodata
        slope_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)
        dzdx_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)
        dzdy_array = numpy.empty(
            (win_ysize, win_xsize),
            dtype=numpy.float64)

        dem_band.ReadAsArray(
            buf_obj=dem_array[y_start:y_end, x_start:x_end],
            **block_offset_copy)

        for row_index in range(1, win_ysize+1):
            for col_index in range(1, win_xsize+1):
                # Notation of the cell below comes from the algorithm
                # description, cells are arraged as follows:
                # abc
                # def
                # ghi
                e = dem_array[row_index, col_index]
                if e == dem_nodata:
                    # we use dzdx as a guard below, no need to set dzdy
                    dzdx_array[row_index-1, col_index-1] = slope_nodata
                    continue
                dzdx_accumulator = 0.0
                dzdy_accumulator = 0.0
                x_denom_factor = 0
                y_denom_factor = 0
                a = dem_array[row_index-1, col_index-1]
                b = dem_array[row_index-1, col_index]
                c = dem_array[row_index-1, col_index+1]
                d = dem_array[row_index, col_index-1]
                f = dem_array[row_index, col_index+1]
                g = dem_array[row_index+1, col_index-1]
                h = dem_array[row_index+1, col_index]
                i = dem_array[row_index+1, col_index+1]

                # a - c direction
                if a != dem_nodata and c != dem_nodata:
                    dzdx_accumulator += a - c
                    x_denom_factor += 2
                elif a != dem_nodata and b != dem_nodata:
                    dzdx_accumulator += a - b
                    x_denom_factor += 1
                elif b != dem_nodata and c != dem_nodata:
                    dzdx_accumulator += b - c
                    x_denom_factor += 1
                elif a != dem_nodata:
                    dzdx_accumulator += (a - e) * 2**0.5
                    x_denom_factor += 1
                elif c != dem_nodata:
                    dzdx_accumulator += (e - c) * 2**0.5
                    x_denom_factor += 1

                # d - f direction
                if d != dem_nodata and f != dem_nodata:
                    dzdx_accumulator += 2 * (d - f)
                    x_denom_factor += 4
                elif d != dem_nodata:
                    dzdx_accumulator += 2 * (d - e)
                    x_denom_factor += 2
                elif f != dem_nodata:
                    dzdx_accumulator += 2 * (e - f)
                    x_denom_factor += 2

                # g - i direction
                if g != dem_nodata and i != dem_nodata:
                    dzdx_accumulator += g - i
                    x_denom_factor += 2
                elif g != dem_nodata and h != dem_nodata:
                    dzdx_accumulator += g - h
                    x_denom_factor += 1
                elif h != dem_nodata and i != dem_nodata:
                    dzdx_accumulator += h - i
                    x_denom_factor += 1
                elif g != dem_nodata:
                    dzdx_accumulator += (g - e) * 2**0.5
                    x_denom_factor += 1
                elif i != dem_nodata:
                    dzdx_accumulator += (e - i) * 2**0.5
                    x_denom_factor += 1

                # a - g direction
                if a != dem_nodata and g != dem_nodata:
                    dzdy_accumulator += a - g
                    y_denom_factor += 2
                elif a != dem_nodata and d != dem_nodata:
                    dzdy_accumulator += a - d
                    y_denom_factor += 1
                elif d != dem_nodata and g != dem_nodata:
                    dzdy_accumulator += d - g
                    y_denom_factor += 1
                elif a != dem_nodata:
                    dzdy_accumulator += (a - e) * 2**0.5
                    y_denom_factor += 1
                elif g != dem_nodata:
                    dzdy_accumulator += (e - g) * 2**0.5
                    y_denom_factor += 1

                # b - h direction
                if b != dem_nodata and h != dem_nodata:
                    dzdy_accumulator += 2 * (b - h)
                    y_denom_factor += 4
                elif b != dem_nodata:
                    dzdy_accumulator += 2 * (b - e)
                    y_denom_factor += 2
                elif h != dem_nodata:
                    dzdy_accumulator += 2 * (e - h)
                    y_denom_factor += 2

                # c - i direction
                if c != dem_nodata and i != dem_nodata:
                    dzdy_accumulator += c - i
                    y_denom_factor += 2
                elif c != dem_nodata and f != dem_nodata:
                    dzdy_accumulator += c - f
                    y_denom_factor += 1
                elif f != dem_nodata and i != dem_nodata:
                    dzdy_accumulator += f - i
                    y_denom_factor += 1
                elif c != dem_nodata:
                    dzdy_accumulator += (c - e) * 2**0.5
                    y_denom_factor += 1
                elif i != dem_nodata:
                    dzdy_accumulator += (e - i) * 2**0.5
                    y_denom_factor += 1

                if x_denom_factor != 0:
                    dzdx_array[row_index-1, col_index-1] = (
                        dzdx_accumulator / (x_denom_factor * x_cell_size))
                else:
                    dzdx_array[row_index-1, col_index-1] = 0.0
                if y_denom_factor != 0:
                    dzdy_array[row_index-1, col_index-1] = (
                        dzdy_accumulator / (y_denom_factor * y_cell_size))
                else:
                    dzdy_array[row_index-1, col_index-1] = 0.0
        valid_mask = dzdx_array != slope_nodata
        slope_array[:] = slope_nodata
        # multiply by 100 for percent output
        slope_array[valid_mask] = 100.0 * numpy.sqrt(
            dzdx_array[valid_mask]**2 + dzdy_array[valid_mask]**2)
        target_slope_band.WriteArray(
            slope_array, xoff=block_offset['xoff'],
            yoff=block_offset['yoff'])

    dem_band = None
    target_slope_band = None
    dem_raster = None
    target_slope_raster = None


@cython.boundscheck(False)
@cython.cdivision(True)
def stats_worker(stats_work_queue):
    """Worker to calculate continuous min, max, mean and standard deviation.

    Parameters:
        stats_work_queue (Queue): a queue of 1D numpy arrays or None. If
            None, function puts a (min, max, mean, stddev) tuple to the
            queue and quits.
        expected_blocks (int): number of expected payloads through
            ``stats_work_queue``. Will terminate after this many.

    Returns:
        None

    """
    LOGGER.debug(f'stats worker PID: {os.getpid()}')
    cdef numpy.ndarray[numpy.float64_t, ndim=1] block
    cdef double M_local = 0.0
    cdef double S_local = 0.0
    cdef double min_value = 0.0
    cdef double max_value = 0.0
    cdef double x = 0.0
    cdef int i, n_elements
    cdef long long n = 0L
    payload = None

    while True:
        try:
            existing_shm = None
            payload = stats_work_queue.get()
            if payload is None:
                break
            if isinstance(payload, numpy.ndarray):
                # if the payload is a normal array take it as the array block
                block = payload
            else:
                # if not an ndarray, it is a shared memory pointer tuple
                shape, dtype, existing_shm = payload
                block = numpy.ndarray(
                    shape, dtype=dtype, buffer=existing_shm.buf)
            if block.size == 0:
                continue
            n_elements = block.size
            with nogil:
                for i in range(n_elements):
                    n = n + 1
                    x = block[i]
                    if n <= 0:
                        with gil:
                            LOGGER.error('invalid value for n %s' % n)
                    if n == 1:
                        M_local = x
                        S_local = 0.0
                        min_value = x
                        max_value = x
                    else:
                        M_last = M_local
                        M_local = M_local+(x - M_local)/<double>(n)
                        S_local = S_local+(x-M_last)*(x-M_local)
                        if x < min_value:
                            min_value = x
                        elif x > max_value:
                            max_value = x
        except Exception as e:
            LOGGER.exception(
                "exception %s %s %s %s %s", x, M_local, S_local, n, payload)
            raise

    if n > 0:
        stats_work_queue.put(
            (min_value, max_value, M_local,
                (S_local / <double>n) ** 0.5))
    else:
        LOGGER.warning(
            "No valid pixels were received, sending None.")
        stats_work_queue.put(None)


ctypedef long long int64t
ctypedef FastFileIterator[long long]* FastFileIteratorLongLongIntPtr
ctypedef FastFileIterator[double]* FastFileIteratorDoublePtr
ctypedef CoordFastFileIterator[double]* CoordFastFileIteratorPtr

def raster_band_percentile(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size=2**28, ffi_buffer_size=2**10):
    """Calculate percentiles of a raster band.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to a raster
            that is of any integer or real type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements will be stored per
            heap file buffer for iteration.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    raster_type = ecoshard.geoprocessing.get_raster_info(
        base_raster_path_band[0])['datatype']
    if raster_type in (
            gdal.GDT_Byte, gdal.GDT_Int16, gdal.GDT_UInt16, gdal.GDT_Int32,
            gdal.GDT_UInt32):
        return _raster_band_percentile_int(
            base_raster_path_band, working_sort_directory, percentile_list,
            heap_buffer_size, ffi_buffer_size)
    elif raster_type in (gdal.GDT_Float32, gdal.GDT_Float64):
        return _raster_band_percentile_double(
            base_raster_path_band, working_sort_directory, percentile_list,
            heap_buffer_size, ffi_buffer_size)
    else:
        raise ValueError(
            'Cannot process raster type %s (not a known integer nor float '
            'type)', raster_type)


def _raster_band_percentile_int(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size, ffi_buffer_size):
    """Calculate percentiles of a raster band of an integer type.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to a raster that
            is of an integer type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements to store in a file
            buffer at any time.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    cdef FILE *fptr
    cdef FastFileIteratorLongLongIntPtr fast_file_iterator
    cdef vector[FastFileIteratorLongLongIntPtr] fast_file_iterator_vector
    cdef vector[FastFileIteratorLongLongIntPtr].iterator ffiv_iter
    cdef int percentile_index = 0
    cdef long long i, n_elements = 0
    cdef int64t next_val = 0L
    cdef double step_size, current_percentile
    cdef double current_step = 0.0
    result_list = []
    rm_dir_when_done = False
    if not os.path.exists(working_sort_directory):
        os.makedirs(working_sort_directory, exist_ok=True)
        rm_dir_when_done = True

    cdef int64t[:] buffer_data

    heapfile_list = []
    file_index = 0
    raster_info = ecoshard.geoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    cdef long long n_pixels = raster_info['raster_size'][0] * raster_info['raster_size'][1]

    LOGGER.debug('total number of pixels %s (%s)', n_pixels, raster_info['raster_size'])
    cdef long long pixels_processed = 0
    LOGGER.debug('sorting data to heap')
    last_update = time.time()
    for _, block_data in ecoshard.geoprocessing.iterblocks(
            base_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                f'data sort to heap {(100.*pixels_processed)/n_pixels:.1f}% '
                f'complete, {pixels_processed} out of {n_pixels}'),

            last_update = time.time()
        if nodata is not None:
            clean_data = block_data[~numpy.isclose(block_data, nodata)]
        else:
            clean_data = block_data.flatten()
        clean_data = clean_data[numpy.isfinite(clean_data)]
        buffer_data = numpy.sort(clean_data).astype(numpy.int64)
        if buffer_data.size == 0:
            continue
        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        heapfile_list.append(file_path)
        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <int64t*>&buffer_data[0], sizeof(int64t), buffer_data.size,
            fptr)
        fclose(fptr)
        file_index += 1

        fast_file_iterator = new FastFileIterator[int64t](
            (bytes(file_path.encode())), ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[int64t])
    LOGGER.debug('calculating percentiles')
    current_percentile = percentile_list[percentile_index]
    step_size = 0
    if n_elements > 0:
        step_size = 100.0 / n_elements

    for i in range(n_elements):
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'calculating percentiles %.2f%% complete',
                100.0 * i / float(n_elements))
            last_update = time.time()
        current_step = step_size * i
        next_val = fast_file_iterator_vector.front().next()
        if current_step >= current_percentile:
            result_list.append(next_val)
            percentile_index += 1
            if percentile_index >= len(percentile_list):
                break
            current_percentile = percentile_list[percentile_index]
        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[int64t])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                FastFileIteratorCompare[int64t])
        else:
            fast_file_iterator = fast_file_iterator_vector.back()
            del fast_file_iterator
            fast_file_iterator_vector.pop_back()
    if percentile_index < len(percentile_list):
        result_list.append(next_val)

    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()
    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)
    return result_list


def _raster_band_percentile_double(
        base_raster_path_band, working_sort_directory, percentile_list,
        heap_buffer_size, ffi_buffer_size):
    """Calculate percentiles of a raster band of a real type.

    Parameters:
        base_raster_path_band (tuple): raster path band tuple to raster that
            is a real/float type.
        working_sort_directory (str): path to a directory that does not
            exist or is empty. This directory will be used to create heapfiles
            with sizes no larger than ``heap_buffer_size`` which are written in the
            of the pattern N.dat where N is in the numbering 0, 1, 2, ... up
            to the number of files necessary to handle the raster.
        percentile_list (list): sorted list of percentiles to report must
            contain values in the range [0, 100].
        heap_buffer_size (int): defines approximately how many elements to hold in
            a single heap file. This is proportional to the amount of maximum
            memory to use when storing elements before a sort and write to
            disk.
        ffi_buffer_size (int): defines how many elements to store in a file
            buffer at any time.

    Returns:
        A list of len(percentile_list) elements long containing the
        percentile values (ranging from [0, 100]) in ``base_raster_path_band``
        where the interpolation scheme is "higher" (i.e. any percentile splits
        will select the next element higher than the percentile cutoff).

    """
    cdef FILE *fptr
    cdef double[:] buffer_data
    cdef FastFileIteratorDoublePtr fast_file_iterator
    cdef vector[FastFileIteratorDoublePtr] fast_file_iterator_vector
    cdef int percentile_index = 0
    cdef long long i, n_elements = 0
    cdef double next_val = 0.0
    cdef double current_step = 0.0
    cdef double step_size, current_percentile
    result_list = []
    rm_dir_when_done = False
    if not os.path.exists(working_sort_directory):
        os.makedirs(working_sort_directory, exist_ok=True)
        rm_dir_when_done = True
    file_index = 0
    nodata = ecoshard.geoprocessing.get_raster_info(
        base_raster_path_band[0])['nodata'][base_raster_path_band[1]-1]
    heapfile_list = []

    raster_info = ecoshard.geoprocessing.get_raster_info(
        base_raster_path_band[0])
    nodata = raster_info['nodata'][base_raster_path_band[1]-1]
    cdef long long n_pixels = (
        raster_info['raster_size'][0] * raster_info['raster_size'][1])
    cdef long long pixels_processed = 0

    last_update = time.time()
    LOGGER.debug('sorting data to heap')
    for _, block_data in ecoshard.geoprocessing.iterblocks(
            base_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                f'data sort to heap {(100.*pixels_processed)/n_pixels:.1f}% '
                f'complete, {pixels_processed} out of {n_pixels}'),

            last_update = time.time()
        if nodata is not None:
            clean_data = block_data[~numpy.isclose(block_data, nodata)]
        else:
            clean_data = block_data.flatten()
        clean_data = clean_data[numpy.isfinite(clean_data)]
        buffer_data = numpy.sort(clean_data).astype(numpy.double)
        if buffer_data.size == 0:
            continue
        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        heapfile_list.append(file_path)
        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <double*>&buffer_data[0], sizeof(double), buffer_data.size, fptr)
        fclose(fptr)
        file_index += 1

        fast_file_iterator = new FastFileIterator[double](
            (bytes(file_path.encode())), ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])

    current_percentile = percentile_list[percentile_index]
    step_size = 0
    if n_elements > 0:
        step_size = 100.0 / n_elements

    LOGGER.debug('calculating percentiles')
    for i in range(n_elements):
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                'calculating percentiles %.2f%% complete',
                100.0 * i / float(n_elements))
            last_update = time.time()
        current_step = step_size * i
        next_val = fast_file_iterator_vector.front().next()
        if current_step >= current_percentile:
            result_list.append(next_val)
            percentile_index += 1
            if percentile_index >= len(percentile_list):
                break
            current_percentile = percentile_list[percentile_index]
        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            FastFileIteratorCompare[double])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                FastFileIteratorCompare[double])
        else:
            fast_file_iterator_vector.pop_back()
    if percentile_index < len(percentile_list):
        result_list.append(next_val)
    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()
    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)
    LOGGER.debug('here is percentile_list: %s', str(result_list))
    return result_list


def greedy_pixel_pick_by_area(
        base_value_raster_path_band, area_per_pixel_raster_path_band,
        selected_area_report_list, output_dir, output_prefix=None,
        heap_buffer_size=2**28, int ffi_buffer_size=2**20):
    """Select pixel masks with a greedy method.

    Parameters:
        base_value_raster_path_band (tuple): raster path band tuple to raster that
            is a real/float type.
        area_per_pixel_raster_path_band (tuple): path to raster that contains
            the area per pixel in the same units as the
            `base_value_raster_path_band`.
        selected_area_report_list (list): list of sorted postive floating
            point values representing area thresholds to generate reports.
        output_dir (str): path to desired output directory, when complete will
            contain a table called `{base_value_raster_path}_greedy_pick.csv`
            and ``len(selected_area_report_list)`` rasters containing masks
            showing the pixels selected in the greedy optimization.
        output_prefix (str): if not none, this prefix is prepended to any
            output file generated by this call.
        heap_buffer_size (int): defines approximately how many elements to hold
            in a single heap file. This is proportional to the amount of
            maximum memory to use when storing elements before a sort and write
            to disk.

    Returns:
        tuple:
            * path to per step optimization table table created by this call.
            * list of raster mask paths

    """
    LOGGER.debug('starting greedy_pixel_pick_by_area')
    cdef FILE *fptr
    cdef double[:] buffer_data
    cdef long long[:] flat_indexes
    cdef double[:] area_data
    cdef CoordFastFileIteratorPtr fast_file_iterator
    cdef vector[CoordFastFileIteratorPtr] fast_file_iterator_vector

    cdef long long i, n_elements = 0
    cdef long long next_coord
    cdef double total_value = 0.0
    cdef double next_value
    cdef double current_step = 0.0
    cdef double pixel_area
    cdef double step_size, current_percentile
    cdef long n_cols
    cdef long long n_pixels
    cdef long long pixels_processed
    cdef double current_area
    cdef int area_threshold_index
    cdef double area_threshold

    try:
        result_list = []
        rm_dir_when_done = False
        os.makedirs(output_dir, exist_ok=True)
        working_sort_directory = os.path.join(
            output_dir, 'sort_dir')
        os.makedirs(working_sort_directory, exist_ok=True)

        if output_prefix is None:
            output_prefix = ''
        basename = os.path.basename(os.path.splitext(base_value_raster_path_band[0])[0])
        table_path = os.path.join(
            output_dir, f'{output_prefix}{basename}_greedy_pick.csv')

        LOGGER.debug(f'write headers to {table_path}')
        with open(table_path, 'w') as table_file:
            table_file.write('target_area,actual_area,total_value\n')

        base_mask_path = os.path.join(working_sort_directory, 'mask.tif')
        ecoshard.geoprocessing.new_raster_from_base(
            base_value_raster_path_band[0], base_mask_path, gdal.GDT_Byte, [2])
        mask_raster = _ManagedRaster(base_mask_path, 1, 1)

        file_index = 0
        raster_info = ecoshard.geoprocessing.get_raster_info(base_value_raster_path_band[0])
        nodata = raster_info['nodata'][base_value_raster_path_band[1]-1]
        n_cols = raster_info['raster_size'][0]
        heapfile_list = []

        n_pixels = (
            raster_info['raster_size'][0] * raster_info['raster_size'][1])
        pixels_processed = 0

        area_per_pixel_raster = gdal.OpenEx(
            area_per_pixel_raster_path_band[0], gdal.OF_RASTER)
        area_per_pixel_band = area_per_pixel_raster.GetRasterBand(
            area_per_pixel_raster_path_band[1])

        last_update = time.time()
        LOGGER.debug('sorting data to heap')
        for offset_dict, block_data in ecoshard.geoprocessing.iterblocks(
                base_value_raster_path_band, largest_block=heap_buffer_size):
            pixels_processed += block_data.size
            LOGGER.debug(pixels_processed)
            if time.time() - last_update > 5.0:
                LOGGER.debug(
                    f'data sort to heap {(100.*pixels_processed)/n_pixels:.1f}% '
                    f'complete, {pixels_processed} out of {n_pixels}'),

                last_update = time.time()
            if nodata is not None:
                nodata_mask = ~numpy.isclose(block_data, nodata)
                clean_data = block_data[nodata_mask].astype(numpy.float64)
            else:
                clean_data = block_data.flatten().astype(numpy.float64)
                nodata_mask = numpy.ones(block_data.shape, dtype=bool)
            finite_mask = numpy.isfinite(clean_data) & (clean_data > 0)
            clean_data = clean_data[finite_mask]
            # -1 for reverse sort largest to smallest
            sort_indexes = numpy.argsort(-1*clean_data)
            if sort_indexes.size == 0:
                LOGGER.debug('no clean data to sort, going to next block')
                continue
            LOGGER.debug(f'this data will sort: {clean_data}')
            buffer_data = clean_data[sort_indexes]

            area_array = area_per_pixel_band.ReadAsArray(**offset_dict).astype(
                numpy.double)
            area_data = area_array[nodata_mask][finite_mask][sort_indexes]

            # create coordinates
            xx, yy = numpy.meshgrid(
                numpy.arange(0, offset_dict['win_xsize']),
                numpy.arange(0, offset_dict['win_ysize']))
            xx = xx.astype(numpy.int64)
            yy = yy.astype(numpy.int64)
            xx += offset_dict['xoff']
            yy += offset_dict['yoff']

            xx = xx[nodata_mask][finite_mask][sort_indexes]
            yy = yy[nodata_mask][finite_mask][sort_indexes]

            flat_indexes = (yy*n_cols+xx).astype(numpy.int64)

            n_elements += buffer_data.size
            file_path = os.path.join(
                working_sort_directory, '%d.dat' % file_index)
            coord_file_path = os.path.join(
                working_sort_directory, '%dcoord.dat' % file_index)
            area_file_path = os.path.join(
                working_sort_directory, '%darea.dat' % file_index)
            heapfile_list.append(file_path)
            heapfile_list.append(coord_file_path)
            heapfile_list.append(area_file_path)

            fptr = fopen(bytes(file_path.encode()), "wb")
            fwrite(
                <double*>&buffer_data[0], sizeof(double), buffer_data.size, fptr)
            fclose(fptr)
            LOGGER.debug(f'just wrote {buffer_data[0]}')

            fptr = fopen(bytes(coord_file_path.encode()), "wb")
            fwrite(
                <long long*>&flat_indexes[0], sizeof(long long), flat_indexes.size,
                fptr)
            fclose(fptr)

            fptr = fopen(bytes(area_file_path.encode()), "wb")
            fwrite(<double*>&area_data[0], sizeof(double), area_data.size, fptr)
            fclose(fptr)

            file_index += 1
            # test4
            fast_file_iterator = new CoordFastFileIterator[double](
                bytes(file_path.encode()),
                bytes(coord_file_path.encode()),
                bytes(area_file_path.encode()),
                ffi_buffer_size)
            fast_file_iterator_vector.push_back(fast_file_iterator)
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                CoordFastFileIteratorCompare[double])

        LOGGER.info(f'done sorting in {sort_dir}')
        area_per_pixel_raster = None
        area_per_pixel_band = None

        current_area = 0.0
        area_threshold_index = 0
        area_threshold = selected_area_report_list[0]

        gtiff_driver = gdal.GetDriverByName('GTiff')

        LOGGER.info(f'starting greedy selection {selected_area_report_list}')
        i = 0
        mask_path_list = []
        while True:
            if time.time() - last_update > 15.0:
                LOGGER.debug(
                    'greedy optimize %.2f%% complete (%d of %d) on %s',
                    100.0 * i / float(n_elements), i, n_elements,
                    base_value_raster_path_band[0])
                last_update = time.time()
            next_coord = fast_file_iterator_vector.front().coord()
            current_area += fast_file_iterator_vector.front().area()
            next_value = fast_file_iterator_vector.front().next()
            total_value += next_value

            mask_raster.set(next_coord % n_cols, next_coord // n_cols, 1)

            i += 1
            if current_area >= area_threshold:
                LOGGER.info(
                    f'current area threshold met {area_threshold} '
                    f'{current_area} {i} steps')
                with open(table_path, 'a') as table_file:
                    table_file.write(
                        f'{area_threshold},{current_area},{total_value}\n')
                mask_raster.flush()
                mask_path = os.path.join(
                    output_dir,
                    f'{output_prefix}{basename}_mask_{current_area}.tif')
                LOGGER.debug(f'writingh mask to {mask_path}')
                shutil.copy(base_mask_path, mask_path)
                mask_path_list.append(mask_path)
                base_raster = None
                area_threshold_index += 1
                if area_threshold_index == len(selected_area_report_list):
                    break
                area_threshold = selected_area_report_list[
                    area_threshold_index]


            pop_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                CoordFastFileIteratorCompare[double])
            if fast_file_iterator_vector.back().size() > 0:
                push_heap(
                    fast_file_iterator_vector.begin(),
                    fast_file_iterator_vector.end(),
                    CoordFastFileIteratorCompare[double])
            else:
                fast_file_iterator_vector.pop_back()
            if fast_file_iterator_vector.size() == 0:
                break

        if area_threshold_index < len(selected_area_report_list):
            LOGGER.info(f'selection ended before enough area was selected at {area_threshold} {current_area} {i} steps')
            LOGGER.info(
                f'current area threshold met {area_threshold} '
                f'{current_area} {i} steps')
            with open(table_path, 'a') as table_file:
                table_file.write(
                    f'{area_threshold},{current_area},{total_value}\n')
            mask_raster.flush()
            mask_path = os.path.join(
                output_dir,
                f'{output_prefix}{basename}_mask_{current_area}.tif')
            LOGGER.debug(f'writingh mask to {mask_path}')
            shutil.copy(base_mask_path, mask_path)
            mask_path_list.append(mask_path)
            area_threshold = selected_area_report_list[
                area_threshold_index]


        # free all the iterator memory
        ffiv_iter = fast_file_iterator_vector.begin()
        while ffiv_iter != fast_file_iterator_vector.end():
            fast_file_iterator = deref(ffiv_iter)
            del fast_file_iterator
            inc(ffiv_iter)
        fast_file_iterator_vector.clear()

        # delete all the heap files
        for file_path in heapfile_list:
            try:
                os.remove(file_path)
            except OSError:
                # you never know if this might fail!
                LOGGER.warning('unable to remove %s', file_path)

        mask_raster.close()
        if rm_dir_when_done:
            shutil.rmtree(working_sort_directory)

        LOGGER.info('all done')
        return table_path, mask_path_list
    except Exception:
        LOGGER.exception(f'exception on {base_value_raster_path_band}')
        raise


def greedy_pixel_pick_by_area_v2(
        base_value_raster_path_band, area_per_pixel_raster_path_band,
        selected_area_report_list, output_dir, output_prefix=None,
        heap_buffer_size=2**28, int ffi_buffer_size=2**10):
    """Select pixel masks with a greedy method.

    Parameters:
        base_value_raster_path_band (tuple): raster path band tuple to raster that
            is a real/float type.
        area_per_pixel_raster_path_band (tuple): path to raster that contains
            the area per pixel in the same units as the
            `selected_area_report_list`.
        output_dir (str): path to desired output directory, when complete will
            contain a table called `{base_value_raster_path}_greedy_pick.csv`
            and ``len(selected_area_report_list)`` rasters containing masks
            showing the pixels selected in the greedy optimization.
        output_prefix (str): if not none, this prefix is prepended to any
            output file generated by this call.
        heap_buffer_size (int): defines approximately how many elements to hold
            in a single heap file. This is proportional to the amount of
            maximum memory to use when storing elements before a sort and write
            to disk.

    Returns:
        ``None``

    """
    LOGGER.debug('starting greedy_pixel_pick_by_area')
    cdef FILE *fptr
    cdef double[:] buffer_data
    cdef long long[:] flat_indexes
    cdef double[:] area_data
    cdef CoordFastFileIteratorPtr fast_file_iterator
    cdef vector[CoordFastFileIteratorPtr] fast_file_iterator_vector

    cdef long long i, n_elements = 0
    cdef long long next_coord
    cdef double total_value = 0.0
    cdef double next_value
    cdef double current_step = 0.0
    cdef double pixel_area
    cdef double step_size, current_percentile
    result_list = []
    rm_dir_when_done = False
    os.makedirs(output_dir, exist_ok=True)
    working_sort_directory = os.path.join(
        output_dir, 'sort_dir')
    os.makedirs(working_sort_directory, exist_ok=True)

    if output_prefix is None:
        output_prefix = ''
    basename = os.path.basename(os.path.splitext(base_value_raster_path_band[0])[0])
    table_path = os.path.join(
        output_dir, f'{output_prefix}{basename}_greedy_pick.csv')

    LOGGER.debug(f'write headers to {table_path}')
    with open(table_path, 'w') as table_file:
        table_file.write('target_area,actual_area,total_value\n')

    # base_mask_path = os.path.join(working_sort_directory, 'mask.tif')
    #ecoshard.geoprocessing.new_raster_from_base(
    #    base_value_raster_path_band[0], base_mask_path, gdal.GDT_Byte, [2])
    # mask_raster = _ManagedRaster(base_mask_path, 1, 1)

    file_index = 0
    raster_info = ecoshard.geoprocessing.get_raster_info(base_value_raster_path_band[0])
    nodata = raster_info['nodata'][base_value_raster_path_band[1]-1]
    cdef long n_cols = raster_info['raster_size'][0]
    heapfile_list = []

    cdef long long n_pixels = (
        raster_info['raster_size'][0] * raster_info['raster_size'][1])
    cdef long long pixels_processed = 0

    area_per_pixel_raster = gdal.OpenEx(
        area_per_pixel_raster_path_band[0], gdal.OF_RASTER)
    area_per_pixel_band = area_per_pixel_raster.GetRasterBand(
        area_per_pixel_raster_path_band[1])

    last_update = time.time()
    LOGGER.debug('sorting data to heap')
    for offset_dict, block_data in ecoshard.geoprocessing.iterblocks(
            base_value_raster_path_band, largest_block=heap_buffer_size):
        pixels_processed += block_data.size
        if time.time() - last_update > 5.0:
            LOGGER.debug(
                f'data sort to heap {(100.*pixels_processed)/n_pixels:.1f}% '
                f'complete, {pixels_processed} out of {n_pixels}'),

            last_update = time.time()
        if nodata is not None:
            nodata_mask = ~numpy.isclose(block_data, nodata)
            clean_data = block_data[nodata_mask].astype(numpy.float64)
        else:
            clean_data = block_data.flatten().astype(numpy.float64)
            nodata_mask = numpy.ones(block_data.shape, dtype=bool)
        finite_mask = numpy.isfinite(clean_data) & (clean_data > 0)
        clean_data = clean_data[finite_mask]
        # -1 for reverse sort largest to smallest
        sort_indexes = numpy.argsort(-1*clean_data)
        if sort_indexes.size == 0:
            continue
        LOGGER.debug(f'this data will sort: {clean_data}')
        buffer_data = clean_data[sort_indexes]

        area_array = area_per_pixel_band.ReadAsArray(**offset_dict).astype(
            numpy.double)
        area_data = area_array[nodata_mask][finite_mask][sort_indexes]

        # create coordinates
        xx, yy = numpy.meshgrid(
            numpy.arange(0, offset_dict['win_xsize']),
            numpy.arange(0, offset_dict['win_ysize']))
        xx = xx.astype(numpy.int64)
        yy = yy.astype(numpy.int64)
        xx += offset_dict['xoff']
        yy += offset_dict['yoff']

        xx = xx[nodata_mask][finite_mask][sort_indexes]
        yy = yy[nodata_mask][finite_mask][sort_indexes]

        flat_indexes = (yy*n_cols+xx).astype(numpy.int64)

        n_elements += buffer_data.size
        file_path = os.path.join(
            working_sort_directory, '%d.dat' % file_index)
        coord_file_path = os.path.join(
            working_sort_directory, '%dcoord.dat' % file_index)
        area_file_path = os.path.join(
            working_sort_directory, '%darea.dat' % file_index)
        heapfile_list.append(file_path)
        heapfile_list.append(coord_file_path)
        heapfile_list.append(area_file_path)

        fptr = fopen(bytes(file_path.encode()), "wb")
        fwrite(
            <double*>&buffer_data[0], sizeof(double), buffer_data.size, fptr)
        fclose(fptr)
        LOGGER.debug(f'just wrote {buffer_data[0]}')

        fptr = fopen(bytes(coord_file_path.encode()), "wb")
        fwrite(
            <long long*>&flat_indexes[0], sizeof(long long), flat_indexes.size,
            fptr)
        fclose(fptr)

        fptr = fopen(bytes(area_file_path.encode()), "wb")
        fwrite(<double*>&area_data[0], sizeof(double), area_data.size, fptr)
        fclose(fptr)

        file_index += 1
        # test4
        fast_file_iterator = new CoordFastFileIterator[double](
            bytes(file_path.encode()),
            bytes(coord_file_path.encode()),
            bytes(area_file_path.encode()),
            ffi_buffer_size)
        fast_file_iterator_vector.push_back(fast_file_iterator)
        push_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            CoordFastFileIteratorCompare[double])

    area_per_pixel_raster = None
    area_per_pixel_band = None

    cdef double current_area = 0.0
    cdef int area_threshold_index = 0
    cdef double area_threshold = selected_area_report_list[0]

    gtiff_driver = gdal.GetDriverByName('GTiff')

    LOGGER.info(f'starting greedy selection {selected_area_report_list}')
    i = 0
    value_threshold_list = []
    while True:
        if time.time() - last_update > 15.0:
            LOGGER.debug(
                'greedy optimize %.2f%% complete (%d of %d) on %s',
                100.0 * i / float(n_elements), i, n_elements,
                base_value_raster_path_band[0])
            last_update = time.time()
        next_coord = fast_file_iterator_vector.front().coord()
        current_area += fast_file_iterator_vector.front().area()
        next_value = fast_file_iterator_vector.front().next()
        total_value += next_value

        #mask_raster.set(next_coord % n_cols, next_coord // n_cols, 1)

        i += 1
        if current_area >= area_threshold:
            value_threshold_list.append(next_value)
            LOGGER.info(
                f'current area threshold met {area_threshold} '
                f'{current_area} {i} steps')
            with open(table_path, 'a') as table_file:
                table_file.write(
                    f'{current_area},{area_threshold},{total_value}\n')
            #mask_raster.flush()
            #mask_path = os.path.join(
            #    output_dir, f'{output_prefix}mask_{current_area}.tif')
            #shutil.copy(base_mask_path, mask_path)
            #base_raster = None
            area_threshold_index += 1
            if area_threshold_index == len(selected_area_report_list):
                break
            area_threshold = selected_area_report_list[
                area_threshold_index]

        pop_heap(
            fast_file_iterator_vector.begin(),
            fast_file_iterator_vector.end(),
            CoordFastFileIteratorCompare[double])
        if fast_file_iterator_vector.back().size() > 0:
            push_heap(
                fast_file_iterator_vector.begin(),
                fast_file_iterator_vector.end(),
                CoordFastFileIteratorCompare[double])
        else:
            fast_file_iterator_vector.pop_back()
        if fast_file_iterator_vector.size() == 0:
            break

    if area_threshold_index < len(selected_area_report_list):
        # TODO: dump mask
        value_threshold_list.append(next_value)
        LOGGER.info(f'selection ended before enough area was selected at {area_threshold} {current_area} {i} steps')
        LOGGER.info(
            f'current area threshold met {area_threshold} '
            f'{current_area} {i} steps')
        with open(table_path, 'a') as table_file:
            table_file.write(
                f'{current_area},{area_threshold},{total_value}\n')
        #mask_raster.flush()
        #mask_path = os.path.join(
        #   output_dir, f'{output_prefix}mask_{current_area}.tif')
        #shutil.copy(base_mask_path, mask_path)
        area_threshold = selected_area_report_list[
            area_threshold_index]


    # free all the iterator memory
    ffiv_iter = fast_file_iterator_vector.begin()
    while ffiv_iter != fast_file_iterator_vector.end():
        fast_file_iterator = deref(ffiv_iter)
        del fast_file_iterator
        inc(ffiv_iter)
    fast_file_iterator_vector.clear()

    # delete all the heap files
    for file_path in heapfile_list:
        try:
            os.remove(file_path)
        except OSError:
            # you never know if this might fail!
            LOGGER.warning('unable to remove %s', file_path)

    #mask_raster.close()
    if rm_dir_when_done:
        shutil.rmtree(working_sort_directory)

    LOGGER.debug(f'here are the threshold values: {value_threshold_list}')

    # create mask rasters
    for area_threshold, value_threshold in zip(
            selected_area_report_list, value_threshold_list):
        LOGGER.info(f'create mask raster for {area_threshold}')
        mask_path = os.path.join(
            output_dir, f'{output_prefix}mask_{area_threshold}.tif')
        ecoshard.geoprocessing.new_raster_from_base(
            base_value_raster_path_band[0], mask_path, gdal.GDT_Byte, [2])
        mask_raster = gdal.OpenEx(mask_path, gdal.OF_RASTER | gdal.GA_Update)
        mask_band = mask_raster.GetRasterBand(1)

        current_area = 0.0
        for (offset_dict, area_array), (_, value_array) in zip(
                ecoshard.geoprocessing.iterblocks(area_per_pixel_raster_path_band),
                ecoshard.geoprocessing.iterblocks(base_value_raster_path_band)):
            if nodata is not None:
                valid_mask = ~numpy.isclose(value_array, nodata)
            else:
                valid_mask = numpy.ones(value_array.shape, dtype=bool)
            selected_value_mask = numpy.full(
                value_array.shape, 2, dtype=numpy.int8)
            selected_value_mask[valid_mask] = (value_array[valid_mask] > value_threshold)
            current_area += numpy.sum(area_array[valid_mask][selected_value_mask[valid_mask]])
            mask_band.WriteArray(
                selected_value_mask,
                xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])

        # pick up any values that are equal without going over the area
        for (offset_dict, area_array), (_, value_array) in zip(
                ecoshard.geoprocessing.iterblocks(area_per_pixel_raster_path_band),
                ecoshard.geoprocessing.iterblocks(base_value_raster_path_band)):
            if current_area >= area_threshold:
                LOGGER.debug(f'current_area >= area_threshold {current_area} >= {area_threshold}')
                break

            if nodata is not None:
                valid_mask = ~numpy.isclose(value_array, nodata)
            else:
                valid_mask = numpy.ones(value_array.shape, dtype=bool)

            selected_value_mask = numpy.full(
                value_array.shape, 2, dtype=numpy.int8)
            selected_value_mask[valid_mask] = (value_array[valid_mask] == value_threshold)
            #selected_value_mask = (value_array == value_threshold) & valid_mask
            if not numpy.any(selected_value_mask):
                continue
            current_area_list = area_array[selected_value_mask]
            current_area_sum = numpy.sum(current_area_list)
            previous_mask = mask_band.ReadAsArray(**offset_dict)
            if current_area_sum+current_area <= area_threshold:
                previous_mask[selected_value_mask] = 1
                current_area += current_area_sum
            else:
                area_index_array = numpy.where(selected_value_mask)
                offset_index = int((area_index_array[0].size) * (current_area) / (current_area+current_area_sum))
                previous_mask[area_index_array[0][:offset_index], area_index_array[1][:offset_index]] = 1
                current_area = area_threshold

            mask_band.WriteArray(
                previous_mask,
                xoff=offset_dict['xoff'], yoff=offset_dict['yoff'])

        mask_band = None
        mask_raster = None
    LOGGER.info('all done')

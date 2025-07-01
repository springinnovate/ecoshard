# coding=UTF-8
"""A collection of raster and vector algorithms and utilities."""
import faulthandler

faulthandler.enable()
import atexit
import bisect
import concurrent.futures
import collections
import functools
import gc
import logging
import math
import multiprocessing
import multiprocessing.shared_memory
import os
import pprint
import hashlib
import queue
import shutil
import signal
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from shapely.strtree import STRtree
from ecoshard import taskgraph
from retrying import retry
from . import geoprocessing_core
from .geoprocessing_core import DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS
from .geoprocessing_core import DEFAULT_OSR_AXIS_MAPPING_STRATEGY
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import psutil
import numexpr
import numpy
import numpy.ma
import rtree
import scipy.interpolate
import scipy.ndimage
import scipy.signal
import scipy.signal.signaltools
import scipy.sparse
import shapely.ops
import shapely.prepared
import shapely.wkb

numexpr.set_num_threads(multiprocessing.cpu_count())
gdal.UseExceptions()


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any = field(compare=False)


LOGGER = logging.getLogger(__name__)

# Used in joining finished TaskGraph Tasks.
_MAX_TIMEOUT = 60.0

_VALID_GDAL_TYPES = set([getattr(gdal, x) for x in dir(gdal.gdalconst) if "GDT_" in x])

_LOGGING_PERIOD = 10.0
_LARGEST_ITERBLOCK = 2**18  # size determined by experimentation with large rasters

_GDAL_TYPE_TO_NUMPY_LOOKUP = {
    gdal.GDT_Byte: numpy.uint8,
    gdal.GDT_Int16: numpy.int16,
    gdal.GDT_Int32: numpy.int32,
    gdal.GDT_UInt16: numpy.uint16,
    gdal.GDT_UInt32: numpy.uint32,
    gdal.GDT_Float32: numpy.float32,
    gdal.GDT_Float64: numpy.float64,
    gdal.GDT_CFloat32: numpy.csingle,
    gdal.GDT_CFloat64: numpy.complex64,
}

if hasattr(gdal, "GDT_Int64"):
    _GDAL_TYPE_TO_NUMPY_LOOKUP.update(
        {
            gdal.GDT_Int64: numpy.dtype(numpy.int64),
            gdal.GDT_UInt64: numpy.dtype(numpy.uint64),
        }
    )

_BASE_GDAL_TYPE_TO_NUMPY = {v: k for k, v in _GDAL_TYPE_TO_NUMPY_LOOKUP.items()}


def retry_create(
    driver,
    target_path,
    n_cols,
    n_rows,
    n_bands,
    datatype,
    options=None,
    max_retries=5,
):
    for _ in range(max_retries):
        try:
            target_raster = driver.Create(
                target_path, n_cols, n_rows, n_bands, datatype, options=options
            )
            if target_raster is not None:
                return target_raster
        except RuntimeError as e:
            LOGGER.warning(f"create failed on {e} trying again")
        time.sleep(1)
    return None


def _start_thread_to_terminate_when_parent_process_dies(ppid):
    pid = os.getpid()

    def f():
        while True:
            try:
                os.kill(ppid, 0)
            except OSError:
                os.kill(pid, signal.SIGTERM)
            time.sleep(1)

    thread = threading.Thread(target=f, daemon=True)
    thread.start()


def raster_calculator(
    base_raster_path_band_const_list,
    local_op,
    target_raster_path,
    datatype_target,
    nodata_target,
    calc_raster_stats=True,
    largest_block=_LARGEST_ITERBLOCK,
    max_timeout=_MAX_TIMEOUT,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    allow_different_blocksize=False,
    skip_sparse=False,
):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes. Rasters with different
    blocksizes can be processed with an override, but large rasters may incur
    significant cache thrashing runtime performance issues.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing:

            * ``(str, int)`` tuples, referring to a raster path/band index pair
              to use as an input.
            * ``numpy.ndarray`` s of up to two dimensions.  These inputs must
              all be broadcastable to each other AND the size of the raster
              inputs.
            * ``(object, 'raw')`` tuples, where ``object`` will be passed
              directly into the ``local_op``.

            All rasters must have the same raster size. If only arrays are
            input, numpy arrays must be broadcastable to each other and the
            final raster size will be the final broadcast array shape. A value
            error is raised if only "raw" inputs are passed.
        local_op (function): a function that must take in as many parameters as
            there are elements in ``base_raster_path_band_const_list``. The
            parameters in ``local_op`` will map 1-to-1 in order with the values
            in ``base_raster_path_band_const_list``. ``raster_calculator`` will
            call ``local_op`` to generate the pixel values in ``target_raster``
            along memory block aligned processing windows. Note any
            particular call to ``local_op`` will have the arguments from
            ``raster_path_band_const_list`` sliced to overlap that window.
            If an argument from ``raster_path_band_const_list`` is a
            raster/path band tuple, it will be passed to ``local_op`` as a 2D
            numpy array of pixel values that align with the processing window
            that ``local_op`` is targeting. A 2D or 1D array will be sliced to
            match the processing window and in the case of a 1D array tiled in
            whatever dimension is flat. If an argument is a scalar it is
            passed as as scalar.
            The return value must be a 2D array of the same size as any of the
            input parameter 2D arrays and contain the desired pixel values
            for the target raster.
        target_raster_path (string): the path of the output raster.  The
            projection, size, and cell size will be the same as the rasters
            in ``base_raster_path_const_band_list`` or the final broadcast
            size of the constant/ndarray values in the list.
        datatype_target (gdal datatype; int): the desired GDAL output type of
            the target raster.
        nodata_target (numerical value): the desired nodata value of the
            target raster.
        calc_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.
        largest_block (int): Attempts to internally iterate over raster blocks
            with this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        max_timeout (float): amount of time in seconds to wait for stats
            worker thread to join. Default is _MAX_TIMEOUT.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        allow_different_blocksize (bool): If True, allow a mismatch of mixed
            blocksizes of input rasters.
        skip_sparse (bool): If true, completely skips raster blocks that are
            sparse or nodata.

    Return:
        None

    Raises:
        ValueError: invalid input provided

    """
    if not base_raster_path_band_const_list:
        raise ValueError(
            "`base_raster_path_band_const_list` is empty and "
            "should have at least one value."
        )

    # It's a common error to not pass in path/band tuples, so check for that
    # and report error if so
    bad_raster_path_list = False
    if not isinstance(base_raster_path_band_const_list, (list, tuple)):
        bad_raster_path_list = True
    else:
        for value in base_raster_path_band_const_list:
            if (
                not _is_raster_path_band_formatted(value)
                and not isinstance(value, numpy.ndarray)
                and not (
                    isinstance(value, tuple) and len(value) == 2 and value[1] == "raw"
                )
            ):
                bad_raster_path_list = True
                break
    if bad_raster_path_list:
        raise ValueError(
            "Expected a sequence of path / integer band tuples, "
            "ndarrays, or (value, 'raw') pairs for "
            "`base_raster_path_band_const_list`, instead got: "
            "%s" % pprint.pformat(base_raster_path_band_const_list)
        )

    # check that any rasters exist on disk and have enough bands
    not_found_paths = []
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    base_raster_path_band_list = [
        path_band
        for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)
    ]
    for value in base_raster_path_band_list:
        if gdal.OpenEx(value[0], gdal.OF_RASTER) is None:
            not_found_paths.append(value[0])
    gdal.PopErrorHandler()
    if not_found_paths:
        raise ValueError(
            "The following files were expected but do not exist on the "
            "filesystem: " + str(not_found_paths)
        )

    # check that band index exists in raster
    invalid_band_index_list = []
    for value in base_raster_path_band_list:
        raster = gdal.OpenEx(value[0], gdal.OF_RASTER)
        if not (1 <= value[1] <= raster.RasterCount):
            invalid_band_index_list.append(value)
        raster = None
    if invalid_band_index_list:
        raise ValueError(
            "The following rasters do not contain requested band "
            "indexes: %s" % invalid_band_index_list
        )

    # check that the target raster is not also an input raster
    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (target_raster_path, str(base_raster_path_band_const_list))
        )

    # check that raster inputs are all the same dimensions
    raster_info_list = []
    geospatial_info_set = dict()
    for path_band in base_raster_path_band_const_list:
        if _is_raster_path_band_formatted(path_band):
            raster_info = get_raster_info(path_band[0])
            raster_info_list.append(raster_info)
            geospatial_info_set[raster_info["raster_size"]] = path_band

    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(geospatial_info_set)
        )

    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list if isinstance(x, numpy.ndarray)
    ]
    stats_worker_thread = None
    try:
        # numpy.broadcast can only take up to 32 arguments, this loop works
        # around that restriction:
        while len(numpy_broadcast_list) > 1:
            numpy_broadcast_list = [
                numpy.broadcast(*numpy_broadcast_list[:32])
            ] + numpy_broadcast_list[32:]
        if numpy_broadcast_list:
            numpy_broadcast_size = numpy_broadcast_list[0].shape
    except ValueError:
        # this gets raised if numpy.broadcast fails
        raise ValueError(
            "Numpy array inputs cannot be broadcast into a single shape %s"
            % numpy_broadcast_list
        )

    if numpy_broadcast_list and len(numpy_broadcast_list[0].shape) > 2:
        raise ValueError(
            "Numpy array inputs must be 2 dimensions or less %s" % numpy_broadcast_list
        )

    # if there are both rasters and arrays, check the numpy shape will
    # be broadcastable with raster shape
    if raster_info_list and numpy_broadcast_list:
        # geospatial lists x/y order and numpy does y/x so reverse size list
        raster_shape = tuple(reversed(raster_info_list[0]["raster_size"]))
        invalid_broadcast_size = False
        if len(numpy_broadcast_size) == 1:
            # if there's only one dimension it should match the last
            # dimension first, in the raster case this is the columns
            # because of the row/column order of numpy. No problem if
            # that value is ``1`` because it will be broadcast, otherwise
            # it should be the same as the raster.
            if (
                numpy_broadcast_size[0] != raster_shape[1]
                and numpy_broadcast_size[0] != 1
            ):
                invalid_broadcast_size = True
        else:
            for dim_index in range(2):
                # no problem if 1 because it'll broadcast, otherwise must
                # be the same value
                if (
                    numpy_broadcast_size[dim_index] != raster_shape[dim_index]
                    and numpy_broadcast_size[dim_index] != 1
                ):
                    invalid_broadcast_size = True
        if invalid_broadcast_size:
            raise ValueError(
                "Raster size %s cannot be broadcast to numpy shape %s"
                % (raster_shape, numpy_broadcast_size)
            )

    # create target raster
    if raster_info_list:
        # if rasters are passed, the target is the same size as the raster
        n_cols, n_rows = raster_info_list[0]["raster_size"]
    elif numpy_broadcast_list:
        # numpy arrays in args and no raster result is broadcast shape
        # expanded to two dimensions if necessary
        if len(numpy_broadcast_size) == 1:
            n_rows, n_cols = 1, numpy_broadcast_size[0]
        else:
            n_rows, n_cols = numpy_broadcast_size
    else:
        raise ValueError(
            "Only (object, 'raw') values have been passed. Raster "
            "calculator requires at least a raster or numpy array as a "
            "parameter. This is the input list: %s"
            % pprint.pformat(base_raster_path_band_const_list)
        )

    if datatype_target not in _VALID_GDAL_TYPES:
        raise ValueError(
            "Invalid target type, should be a gdal.GDT_* type, received "
            '"%s"' % datatype_target
        )

    # Create target raster
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    try:
        os.makedirs(os.path.dirname(target_raster_path), exist_ok=True)
    except FileNotFoundError:
        # happens when no directory
        pass
    LOGGER.debug(
        f"creating {target_raster_path} with " f"{raster_driver_creation_tuple[1]}"
    )
    target_raster = retry_create(
        raster_driver,
        target_raster_path,
        n_cols,
        n_rows,
        1,
        datatype_target,
        options=raster_driver_creation_tuple[1],
    )
    target_band = target_raster.GetRasterBand(1)
    target_band.FlushCache()
    target_raster.FlushCache()
    if nodata_target is not None:
        target_band.SetNoDataValue(nodata_target)

    if raster_info_list:
        # use the first raster in the list for the projection and geotransform
        base_raster_info = raster_info_list[0]
        projection_wkt = base_raster_info["projection_wkt"]
        if projection_wkt is not None:
            target_raster.SetProjection(projection_wkt)
        target_raster.SetGeoTransform(base_raster_info["geotransform"])

    try:
        LOGGER.debug("build canonical args and block offset list")

        base_canonical_arg_list = []
        canonical_base_raster_path_band_list = []
        for value in base_raster_path_band_const_list:
            # the input has been tested and value is either a raster/path band
            # tuple, 1d ndarray, 2d ndarray, or (value, 'raw') tuple.
            if _is_raster_path_band_formatted(value):
                # it's a raster/path band, keep track of open raster and band
                # for later so we can `None` them.
                canonical_base_raster_path_band_list.append(value)
                base_canonical_arg_list.append(
                    value
                )  # this will be loaded into a raster by a worker
            elif isinstance(value, numpy.ndarray):
                if value.ndim == 1:
                    # easier to process as a 2d array for writing to band
                    base_canonical_arg_list.append(value.reshape((1, value.shape[0])))
                else:  # dimensions are two because we checked earlier.
                    base_canonical_arg_list.append(value)
            elif isinstance(value, tuple):
                base_canonical_arg_list.append(value)
            else:
                raise ValueError(
                    f"An unexpected ``value`` occurred. This should never happen. "
                    f"Value: {value}"
                )
        if len(canonical_base_raster_path_band_list) > 0:
            block_offset_list = list(
                iterblocks(
                    canonical_base_raster_path_band_list,
                    offset_only=True,
                    largest_block=largest_block,
                    skip_sparse=skip_sparse,
                    allow_different_blocksize=allow_different_blocksize,
                )
            )
        else:
            block_offset_list = list(
                iterblocks(
                    (target_raster_path, 1),
                    offset_only=True,
                    largest_block=largest_block,
                    skip_sparse=False,
                    allow_different_blocksize=allow_different_blocksize,
                )
            )

        LOGGER.debug(f"process {len(block_offset_list)} blocks")
        LOGGER.debug(
            f"canonical_base_raster_path_band_list {canonical_base_raster_path_band_list}"
        )

        set_of_blocksizes = set()
        for path_band in canonical_base_raster_path_band_list:
            if not _is_raster_path_band_formatted(path_band):
                continue
            set_of_blocksizes.add(tuple(get_raster_info(path_band[0])["block_size"]))
        set_of_blocksizes.add(tuple(get_raster_info(target_raster_path)["block_size"]))
        if len(set_of_blocksizes) > 1:
            message = (
                "Input raster blocksizes do not match output blocksizes, have "
                f"at least these values: {set_of_blocksizes}."
            )
            if allow_different_blocksize:
                LOGGER.warn(message)
            else:
                raise ValueError(
                    message + " Pass ``allow_different_blocksize=True`` to allow this "
                    "configuration"
                )

        exception_queue = queue.Queue()
        if calc_raster_stats:
            # if this queue is used to send computed valid blocks of
            # the raster to an incremental statistics calculator worker
            stats_worker_queue = queue.Queue()
        else:
            stats_worker_queue = None

        if calc_raster_stats:
            # To avoid doing two passes on the raster to calculate standard
            # deviation, we implement a continuous statistics calculation
            # as the raster is computed. This computational effort is high
            # and benefits from running in parallel. This queue and worker
            # takes a valid block of a raster and incrementally calculates
            # the raster's statistics. When ``None`` is pushed to the queue
            # the worker will finish and return a (min, max, mean, std)
            # tuple.
            LOGGER.debug("starting stats worker thread")
            stats_worker_thread = threading.Thread(
                target=geoprocessing_core.stats_worker,
                args=(stats_worker_queue,),
            )
            stats_worker_thread.daemon = True
            stats_worker_thread.start()

        n_pixels = n_cols * n_rows
        # build up work
        work_queue = queue.Queue()
        for block_offset in block_offset_list:
            work_queue.put(block_offset)
        work_queue.put(None)

        n_workers = min(multiprocessing.cpu_count() // 2, len(block_offset_list))
        active_workers = 0
        target_workers = 1
        last_read_time = 9999
        overtime_lock = threading.Lock()
        raster_workers_complete = threading.Event()

        def _raster_worker(work_queue, result_block_queue, exception_queue):
            """Read from arrays and create blocks.

            Normal behavior involves the worker fetching a block offset
            to process from ``work_queue`` in which it applies ``local_op``
            to the raster/args stack in ``base_canonical_arg_list``.
            The resulting array is pushed to ``result_block_queue``.

            The worker terminates when it recieves a ``None`` from
            ``work_queue`` at which point it also ``put``s a ``None``
            in ``work_queue`` to trigger other workers to terminate.

            If an exception is encountered during processing, this worker
            will drain ``work_queue``, put a ``None`` to trigger other
            workers to quit, push a ``None`` to ``result_block_queue``
            log the exception and push it to ``exception_queue`` before
            raising an exception itself.

            """
            # used to load balance, watching how many overtimes and how many
            # active workers exist.
            nonlocal active_workers
            nonlocal target_workers
            nonlocal last_read_time

            local_arg_list = []
            base_raster_list = []
            # load the raster/bands from the arg list for local processing
            for value in base_canonical_arg_list:
                if _is_raster_path_band_formatted(value):
                    try:
                        base_raster_list.append(gdal.OpenEx(value[0], gdal.OF_RASTER))
                        local_arg_list.append(
                            base_raster_list[-1].GetRasterBand(value[1])
                        )
                    except RuntimeError:
                        # probably the raster wasn't on disk and exceptions are on
                        local_arg_list.append(value)

                else:
                    local_arg_list.append(value)

            try:
                while True:
                    # check to see if new workers need to be made or workers
                    # need to be shut down
                    with overtime_lock:
                        if active_workers > target_workers:
                            # kill current thread
                            # LOGGER.debug(f'{active_workers} > {target_workers} terminating current worker {threading.current_thread()}')
                            active_workers -= 1
                            if active_workers == 0:
                                # This only happens if all the raster work is done
                                result_block_queue.put(None)
                            return
                        while active_workers < target_workers:
                            # spin up new workers
                            # LOGGER.debug(f'{active_workers} < {target_workers} spin up new worker')
                            active_workers += 1
                            raster_worker = threading.Thread(
                                target=_raster_worker,
                                args=(
                                    work_queue,
                                    result_block_queue,
                                    exception_queue,
                                ),
                            )
                            raster_worker.daemon = True
                            raster_worker.start()

                    # read input blocks
                    block_offset = work_queue.get()
                    local_start_time = time.time()
                    if block_offset is None:
                        work_queue.put(None)
                        active_workers -= 1
                        target_workers = 0
                        if active_workers == 0:
                            # This only happens if all the raster work is done
                            LOGGER.debug(
                                "as last raster worker alive, sending None to the result queue to shut down writer"
                            )
                            result_block_queue.put(None)
                        raster_workers_complete.set()
                        return
                    offset_list = (block_offset["yoff"], block_offset["xoff"])
                    blocksize = (
                        block_offset["win_ysize"],
                        block_offset["win_xsize"],
                    )
                    data_blocks = []

                    # process block_offset sized chunks of arrays or local args
                    # for passing to ``local_op``
                    read_start_time = time.time()
                    for value in local_arg_list:
                        if isinstance(value, gdal.Band):
                            data_blocks.append(value.ReadAsArray(**block_offset))
                            # I've encountered the following error when a gdal raster
                            # is corrupt, often from multiple threads writing to the
                            # same file. This helps to catch the error early rather
                            # than lead to confusing values of ``data_blocks`` later.
                            if not isinstance(data_blocks[-1], numpy.ndarray):
                                raise ValueError(
                                    f"got a {data_blocks[-1]} when trying to read "
                                    f"{value.GetDataset().GetFileList()} at "
                                    f"{block_offset}, expected numpy.ndarray."
                                )
                        elif isinstance(value, numpy.ndarray):
                            # must be numpy array and all have been conditioned to be
                            # 2d, so start with 0:1 slices and expand if possible
                            slice_list = [slice(0, 1)] * 2
                            tile_dims = list(blocksize)
                            for dim_index in [0, 1]:
                                if value.shape[dim_index] > 1:
                                    slice_list[dim_index] = slice(
                                        offset_list[dim_index],
                                        offset_list[dim_index] + blocksize[dim_index],
                                    )
                                    tile_dims[dim_index] = 1
                            data_blocks.append(
                                numpy.tile(value[tuple(slice_list)], tile_dims)
                            )
                        else:
                            # must be a raw tuple
                            data_blocks.append(value[0])
                    read_time = time.time() - read_start_time
                    target_block = local_op(*data_blocks)
                    put_time_start = time.time()
                    if target_block is None:
                        # allow for short circuit
                        result_block_queue.put((None, block_offset))
                        continue

                    if (
                        not isinstance(target_block, numpy.ndarray)
                        or target_block.shape != blocksize
                    ):
                        raise ValueError(
                            "Expected `local_op` to return a numpy.ndarray of "
                            "shape %s but got this instead: %s"
                            % (blocksize, target_block)
                        )

                    result_block_queue.put((target_block, block_offset))

                    put_time = time.time() - put_time_start
                    op_time = put_time_start - local_start_time
                    with overtime_lock:
                        # if read_time > last_read_time*4:
                        #     LOGGER.debug(f'{read_time} {last_read_time}')
                        if put_time > op_time or read_time > last_read_time * 4:
                            # exponential backoff if local cycle is
                            # 4 times slower than the average
                            target_workers = max(active_workers // 2, 1)
                        else:
                            # make a new worker if local cycle is so
                            # fast it's 0 or faster than average
                            target_workers = min(n_workers, target_workers + 1)
                        last_read_time = (read_time + last_read_time) / 2

            except Exception as e:
                LOGGER.exception("error in worker")
                # drain the work queue
                try:
                    while True:
                        work_queue.get_nowait()
                except queue.Empty:
                    work_queue.put(None)
                result_block_queue.put(None)
                exception_queue.put(e)
                raster_workers_complete.set()
                raise
            finally:
                base_raster_list[:] = []
                local_arg_list[:] = []
            base_raster_list[:] = []

        # start first worker
        result_block_queue = queue.Queue(n_workers)
        raster_worker = threading.Thread(
            target=_raster_worker,
            args=(work_queue, result_block_queue, exception_queue),
        )
        raster_worker.daemon = True
        active_workers += 1
        raster_worker.start()

        pixels_processed = 0
        last_time = time.time()
        logging_lock = threading.Lock()
        LOGGER.debug("started raster local_op workers")

        def _raster_writer(result_block_queue, target_raster_path, exception_queue):
            """Write incoming blocks to target raster.

            Normal behavior involves fetching a ``target_block, block_offset``
            tuple from ``result_block_queue``, the target block is written
            to the block offset in the already opened ``target_band``.

            If a ``None`` is received, worker puts a ``None`` to the
            ``result_block_queue`` to signal other workers to terminate, then
            terminates normally.

            If an exception occurs during processing, that exception is
            logged, ``put`` to ``exception_queue``, the ``result_block_queue``
            is drained, and a ``None`` is placed to terminate other workers
            then the worker raises the exception locally.

            """
            # create target raster
            try:
                nonlocal last_time
                nonlocal pixels_processed
                while True:
                    with logging_lock:
                        last_time = _invoke_timed_callback(
                            last_time,
                            lambda: LOGGER.info(
                                f"{float(pixels_processed) / n_pixels * 100.0:.2f}% "
                                f"complete on {target_raster_path}, with {active_workers} active workers",
                            ),
                            _LOGGING_PERIOD,
                        )
                    payload = result_block_queue.get()
                    if payload is None:
                        # signal to terminate
                        LOGGER.debug("got a None, terminating writer")
                        return

                    target_block, block_offset = payload

                    if target_block is not None:
                        target_band.WriteArray(
                            target_block,
                            yoff=block_offset["yoff"],
                            xoff=block_offset["xoff"],
                        )
                    pixels_processed += (
                        block_offset["win_xsize"] * block_offset["win_ysize"]
                    )

                    # send result to stats calculator
                    if stats_worker_queue and target_block is not None:
                        # guard against an undefined nodata target
                        if nodata_target is not None:
                            target_block = target_block[target_block != nodata_target]
                        finite_mask = numpy.isfinite(target_block)
                        target_block = target_block[finite_mask]
                        target_block = target_block.astype(numpy.float64).flatten()
                        stats_worker_queue.put(target_block)
                    if pixels_processed == n_pixels:
                        LOGGER.info(f"writer 100.0% complete for {target_raster_path}")
                        return

            except Exception as e:
                LOGGER.exception("error on _raster_writer")
                exception_queue.put(e)
                raise
            finally:
                LOGGER.debug("all done with _raster_writer!")

        raster_writer = threading.Thread(
            target=_raster_writer,
            args=(result_block_queue, target_raster_path, exception_queue),
        )
        raster_writer.daemon = True
        raster_writer.start()

        raster_workers_complete.wait()
        LOGGER.info(
            f"raster workers for {target_raster_path} complete, waiting for writer to finish"
        )
        raster_writer.join()
        LOGGER.info(
            f"raster writers for {target_raster_path} complete, waiting for stats worker to finish"
        )

        if calc_raster_stats:
            LOGGER.info("Waiting for raster stats worker result.")
            stats_worker_queue.put(None)
            stats_worker_thread.join(max_timeout)
            if stats_worker_thread.is_alive():
                LOGGER.error("stats_worker_thread.join() timed out")
                raise RuntimeError("stats_worker_thread.join() timed out")
            payload = stats_worker_queue.get(True, max_timeout)
            if payload is not None:
                target_min, target_max, target_mean, target_stddev = payload
                LOGGER.debug(
                    f"stats payload: {target_min}, {target_max}, "
                    f"{target_mean}, {target_stddev}"
                )
                target_band.SetStatistics(
                    float(target_min),
                    float(target_max),
                    float(target_mean),
                    float(target_stddev),
                )
    except Exception:
        LOGGER.exception("exception encountered in raster_calculator")
        raise
    finally:
        # This block ensures that rasters are destroyed even if there's an
        # exception raised.
        if calc_raster_stats and stats_worker_thread:
            if stats_worker_thread.is_alive():
                stats_worker_queue.put(None, True, max_timeout)
                LOGGER.info("Waiting for raster stats worker result.")
                stats_worker_thread.join(max_timeout)
                if stats_worker_thread.is_alive():
                    LOGGER.error("stats_worker_thread.join() timed out")
                    raise RuntimeError("stats_worker_thread.join() timed out")

            # check for an exception in the workers, otherwise get result
            # and pass to writer
            try:
                exception = exception_queue.get_nowait()
                LOGGER.error("Exception encountered at termination.")
                raise exception
            except queue.Empty:
                pass
        target_raster.FlushCache()
        target_band.FlushCache()
        target_band = None
        target_raster = None


def align_and_resize_raster_stack(
    base_raster_path_list,
    target_raster_path_list,
    resample_method_list,
    target_pixel_size,
    bounding_box_mode,
    base_vector_path_list=None,
    raster_align_index=None,
    base_projection_wkt_list=None,
    target_projection_wkt=None,
    vector_mask_options=None,
    gdal_warp_options=None,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
):
    """Generate rasters from a base such that they align geospatially.

    This function resizes base rasters that are in the same geospatial
    projection such that the result is an aligned stack of rasters that have
    the same cell size, dimensions, and bounding box. This is achieved by
    clipping or resizing the rasters to intersected, unioned, or equivocated
    bounding boxes of all the raster and vector input.

    Args:
        base_raster_path_list (sequence): a sequence of base raster paths that
            will be transformed and will be used to determine the target
            bounding box.
        target_raster_path_list (sequence): a sequence of raster paths that
            will be created to one-to-one map with ``base_raster_path_list``
            as aligned versions of those original rasters. If there are
            duplicate paths in this list, the function will raise a ValueError.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_list`` during
            resizing.  Each element must be one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_pixel_size (list/tuple): the target raster's x and y pixel size
            example: (30, -30).
        bounding_box_mode (string): one of "union", "intersection", or
            a sequence of floats of the form [minx, miny, maxx, maxy] in the
            target projection coordinate system.  Depending
            on the value, output extents are defined as the union,
            intersection, or the explicit bounding box.
        base_vector_path_list (sequence): a sequence of base vector paths
            whose bounding boxes will be used to determine the final bounding
            box of the raster stack if mode is 'union' or 'intersection'.  If
            mode is 'bb=[...]' then these vectors are not used in any
            calculation.
        raster_align_index (int): indicates the index of a
            raster in ``base_raster_path_list`` that the target rasters'
            bounding boxes pixels should align with.  This feature allows
            rasters whose raster dimensions are the same, but bounding boxes
            slightly shifted less than a pixel size to align with a desired
            grid layout.  If ``None`` then the bounding box of the target
            rasters is calculated as the precise intersection, union, or
            bounding box.
        base_projection_wkt_list (sequence): if not None, this is a sequence of
            base projections of the rasters in ``base_raster_path_list``. If a
            value is ``None`` the ``base_sr`` is assumed to be whatever is
            defined in that raster. This value is useful if there are rasters
            with no projection defined, but otherwise known.
        target_projection_wkt (string): if not None, this is the desired
            projection of all target rasters in Well Known Text format. If
            None, the base SRS will be passed to the target.
        vector_mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'`` (str): path to the mask vector file.
              This vector will be automatically projected to the target
              projection if its base coordinate system does not match the
              target.
            * ``'mask_layer_name'`` (str): the layer name to use for masking.
              If this key is not in the dictionary the default is to use
              the layer at index 0.
            * ``'mask_vector_where_filter'`` (str): an SQL WHERE string.
              This will be used to filter the geometry in the mask. Ex: ``'id
              > 10'`` would use all features whose field value of 'id' is >
              10.

        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the `GDAL Warp documentation
            <https://gdal.org/api/gdalwarp_cpp.html#_CPPv415GDALWarpOptions>`_
            for valid options.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Return:
        None

    Raises:
        ValueError
            If any combination of the raw bounding boxes, raster
            bounding boxes, vector bounding boxes, and/or vector_mask
            bounding box does not overlap to produce a valid target.
        ValueError
            If any of the input or target lists are of different
            lengths.
        ValueError
            If there are duplicate paths on the target list which would
            risk corrupted output.
        ValueError
            If some combination of base, target, and embedded source
            reference systems results in an ambiguous target coordinate
            system.
        ValueError
            If ``vector_mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.
        ValueError
            If ``pixel_size`` is not a 2 element sequence of numbers.

    """
    # make sure that the input lists are of the same length
    list_lengths = [
        len(base_raster_path_list),
        len(target_raster_path_list),
        len(resample_method_list),
    ]
    if len(set(list_lengths)) != 1:
        raise ValueError(
            "base_raster_path_list, target_raster_path_list, and "
            "resample_method_list must be the same length "
            " current lengths are %s" % (str(list_lengths))
        )

    unique_targets = set(target_raster_path_list)
    if len(unique_targets) != len(target_raster_path_list):
        seen = set()
        duplicate_list = []
        for path in target_raster_path_list:
            if path not in seen:
                seen.add(path)
            else:
                duplicate_list.append(path)
        raise ValueError(
            "There are duplicated paths on the target list. This is an "
            "invalid state of ``target_path_list``. Duplicates: %s" % (duplicate_list)
        )

    # we can accept 'union', 'intersection', or a 4 element list/tuple
    if bounding_box_mode not in ["union", "intersection"] and (
        not isinstance(bounding_box_mode, (list, tuple)) or len(bounding_box_mode) != 4
    ):
        raise ValueError("Unknown bounding_box_mode %s" % (str(bounding_box_mode)))

    n_rasters = len(base_raster_path_list)
    if (raster_align_index is not None) and (
        (raster_align_index < 0) or (raster_align_index >= n_rasters)
    ):
        raise ValueError(
            "Alignment index is out of bounds of the datasets index: %s"
            " n_elements %s" % (raster_align_index, n_rasters)
        )

    _assert_is_valid_pixel_size(target_pixel_size)

    # used to get bounding box, projection, and possible alignment info
    raster_info_list = [get_raster_info(path) for path in base_raster_path_list]

    # get the literal or intersecting/unioned bounding box
    if isinstance(bounding_box_mode, (list, tuple)):
        # if it's a sequence or tuple, it must be a manual bounding box
        LOGGER.debug("assuming manual bounding box mode of %s", bounding_box_mode)
        target_bounding_box = bounding_box_mode
    else:
        # either intersection or union, get list of bounding boxes, reproject
        # if necessary, and reduce to a single box
        if base_vector_path_list is not None:
            # vectors are only interesting for their bounding boxes, that's
            # this construction is inside an else.
            vector_info_list = [get_vector_info(path) for path in base_vector_path_list]
        else:
            vector_info_list = []

        raster_bounding_box_list = []
        for raster_index, raster_info in enumerate(raster_info_list):
            # this block calculates the base projection of ``raster_info`` if
            # ``target_projection_wkt`` is defined, thus implying a
            # reprojection will be necessary.
            if target_projection_wkt:
                if base_projection_wkt_list and base_projection_wkt_list[raster_index]:
                    # a base is defined, use that
                    base_raster_projection_wkt = base_projection_wkt_list[raster_index]
                else:
                    # otherwise use the raster's projection and there must
                    # be one since we're reprojecting
                    base_raster_projection_wkt = raster_info["projection_wkt"]
                    if not base_raster_projection_wkt:
                        raise ValueError(
                            "no projection for raster %s"
                            % base_raster_path_list[raster_index]
                        )
                # since the base spatial reference is potentially different
                # than the target, we need to transform the base bounding
                # box into target coordinates so later we can calculate
                # accurate bounding box overlaps in the target coordinate
                # system
                raster_bounding_box_list.append(
                    transform_bounding_box(
                        raster_info["bounding_box"],
                        base_raster_projection_wkt,
                        target_projection_wkt,
                    )
                )
            else:
                raster_bounding_box_list.append(raster_info["bounding_box"])

        # include the vector bounding box information to make a global list
        # of target bounding boxes
        bounding_box_list = [
            (
                vector_info["bounding_box"]
                if target_projection_wkt is None
                else transform_bounding_box(
                    vector_info["bounding_box"],
                    vector_info["projection_wkt"],
                    target_projection_wkt,
                )
            )
            for vector_info in vector_info_list
        ] + raster_bounding_box_list

        target_bounding_box = merge_bounding_box_list(
            bounding_box_list, bounding_box_mode
        )

    if vector_mask_options:
        # ensure the mask exists and intersects with the target bounding box
        if "mask_vector_path" not in vector_mask_options:
            raise ValueError(
                "vector_mask_options passed, but no value for "
                '"mask_vector_path": %s',
                vector_mask_options,
            )

        mask_vector_info = get_vector_info(vector_mask_options["mask_vector_path"])

        if "mask_vector_where_filter" in vector_mask_options:
            # the bounding box only exists for the filtered features
            mask_vector = gdal.OpenEx(
                vector_mask_options["mask_vector_path"], gdal.OF_VECTOR
            )
            mask_layer = mask_vector.GetLayer()
            mask_layer.SetAttributeFilter(
                vector_mask_options["mask_vector_where_filter"]
            )
            mask_bounding_box = merge_bounding_box_list(
                [
                    [feature.GetGeometryRef().GetEnvelope()[i] for i in [0, 2, 1, 3]]
                    for feature in mask_layer
                ],
                "union",
            )
            mask_layer = None
            mask_vector = None
        else:
            # if no where filter then use the raw vector bounding box
            mask_bounding_box = mask_vector_info["bounding_box"]

        mask_vector_projection_wkt = mask_vector_info["projection_wkt"]
        if mask_vector_projection_wkt is not None and target_projection_wkt is not None:
            mask_vector_bb = transform_bounding_box(
                mask_bounding_box,
                mask_vector_info["projection_wkt"],
                target_projection_wkt,
            )
        else:
            mask_vector_bb = mask_vector_info["bounding_box"]
        # Calling `merge_bounding_box_list` will raise an ValueError if the
        # bounding box of the mask and the target do not intersect. The
        # result is otherwise not used.
        _ = merge_bounding_box_list(
            [target_bounding_box, mask_vector_bb], "intersection"
        )

    if raster_align_index is not None and raster_align_index >= 0:
        # bounding box needs alignment
        align_bounding_box = raster_info_list[raster_align_index]["bounding_box"]
        align_pixel_size = raster_info_list[raster_align_index]["pixel_size"]
        # adjust bounding box so lower left corner aligns with a pixel in
        # raster[raster_align_index]
        for index in [0, 1]:
            n_pixels = int(
                (target_bounding_box[index] - align_bounding_box[index])
                / float(align_pixel_size[index])
            )
            target_bounding_box[index] = (
                n_pixels * align_pixel_size[index] + align_bounding_box[index]
            )

    job_list = []
    task_graph = taskgraph.TaskGraph(
        os.path.dirname(target_raster_path_list[0]),
        min(len(target_raster_path_list), multiprocessing.cpu_count()),
        parallel_mode="thread",
    )
    for index, (base_path, target_path, resample_method) in enumerate(
        zip(base_raster_path_list, target_raster_path_list, resample_method_list)
    ):
        worker = task_graph.add_task(
            func=warp_raster,
            args=(base_path, target_pixel_size, target_path, resample_method),
            kwargs={
                "target_bb": target_bounding_box,
                "raster_driver_creation_tuple": (raster_driver_creation_tuple),
                "target_projection_wkt": target_projection_wkt,
                "base_projection_wkt": (
                    None
                    if not base_projection_wkt_list
                    else base_projection_wkt_list[index]
                ),
                "vector_mask_options": vector_mask_options,
                "gdal_warp_options": gdal_warp_options,
            },
        )
        job_list.append(worker)
    for index, future in enumerate(job_list):
        future.join()
        LOGGER.info(
            "%d of %d aligned: %s",
            index + 1,
            n_rasters,
            os.path.basename(target_path),
        )
    task_graph.join()
    task_graph.close()
    task_graph = None

    LOGGER.info("aligned all %d rasters.", n_rasters)


def new_raster_from_base(
    base_path,
    target_path,
    datatype,
    band_nodata_list,
    fill_value_list=None,
    n_rows=None,
    n_cols=None,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Create new raster by coping spatial reference/geotransform of base.

    A convenience function to simplify the creation of a new raster from the
    basis of an existing one.  Depending on the input mode, one can create
    a new raster of the same dimensions, geotransform, and georeference as
    the base.  Other options are provided to change the raster dimensions,
    number of bands, nodata values, data type, and core raster creation
    options.

    Args:
        base_path (string): path to existing raster.
        target_path (string): path to desired target raster.
        datatype: the pixel datatype of the output raster, for example
            gdal.GDT_Float32.  See the following header file for supported
            pixel types:
            http://www.gdal.org/gdal_8h.html#22e22ce0a55036a96f652765793fb7a4
        band_nodata_list (sequence): list of nodata values, one for each band,
            to set on target raster.  If value is 'None' the nodata value is
            not set for that band.  The number of target bands is inferred
            from the length of this list.
        fill_value_list (sequence): list of values to fill each band with. If
            None, no filling is done.
        n_rows (int): if not None, defines the number of target raster rows.
        n_cols (int): if not None, defines the number of target raster
            columns.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    base_raster = gdal.OpenEx(base_path, gdal.OF_RASTER)
    if n_rows is None:
        n_rows = base_raster.RasterYSize
    if n_cols is None:
        n_cols = base_raster.RasterXSize
    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])

    local_raster_creation_options = list(raster_driver_creation_tuple[1])
    # PIXELTYPE is sometimes used to define signed vs. unsigned bytes and
    # the only place that is stored is in the IMAGE_STRUCTURE metadata
    # copy it over if it exists and it not already defined by the input
    # creation options. It's okay to get this info from the first band since
    # all bands have the same datatype
    base_band = base_raster.GetRasterBand(1)
    metadata = base_band.GetMetadata("IMAGE_STRUCTURE")
    if "PIXELTYPE" in metadata and not any(
        ["PIXELTYPE" in option for option in local_raster_creation_options]
    ):
        local_raster_creation_options.append("PIXELTYPE=" + metadata["PIXELTYPE"])

    block_size = base_band.GetBlockSize()
    # It's not clear how or IF we can determine if the output should be
    # striped or tiled.  Here we leave it up to the default inputs or if its
    # obviously not striped we tile.
    if not any(["TILED" in option for option in local_raster_creation_options]):
        # TILED not set, so lets try to set it to a reasonable value
        if block_size[0] != n_cols:
            # if x block is not the width of the raster it *must* be tiled
            # otherwise okay if it's striped or tiled, I can't construct a
            # test case to cover this, but there is nothing in the spec that
            # restricts this so I have it just in case.
            local_raster_creation_options.append("TILED=YES")

    if not any(["PREDICTOR" in option for option in local_raster_creation_options]):
        if datatype in [gdal.GDT_Float32, gdal.GDT_Float64]:
            compression_predictor = 3
        else:
            compression_predictor = 2
        local_raster_creation_options.append(f"PREDICTOR={compression_predictor}")

    if not any(["BLOCK" in option for option in local_raster_creation_options]):
        # not defined, so lets copy what we know from the current raster
        local_raster_creation_options.extend(
            ["BLOCKXSIZE=%d" % block_size[0], "BLOCKYSIZE=%d" % block_size[1]]
        )

    # make target directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    except FileNotFoundError:
        # happens when no directory
        pass
    base_band = None
    n_bands = len(band_nodata_list)
    LOGGER.debug(f"about to create {target_path}")
    target_raster = retry_create(
        driver,
        target_path,
        n_cols,
        n_rows,
        n_bands,
        datatype,
        options=local_raster_creation_options,
    )
    target_raster.SetProjection(base_raster.GetProjection())
    target_raster.SetGeoTransform(base_raster.GetGeoTransform())
    base_raster = None

    for index, nodata_value in enumerate(band_nodata_list):
        if nodata_value is None:
            continue
        target_band = target_raster.GetRasterBand(index + 1)
        try:
            target_band.SetNoDataValue(nodata_value.item())
        except AttributeError:
            target_band.SetNoDataValue(nodata_value)

    last_time = time.time()
    pixels_processed = 0
    n_pixels = n_cols * n_rows
    LOGGER.debug(f"about to fill {target_path}")
    target_raster.FlushCache()
    if fill_value_list is not None:
        for index, fill_value in enumerate(fill_value_list):
            if fill_value is None:
                continue
            target_band = target_raster.GetRasterBand(index + 1)
            # some rasters are very large and a fill can appear to cause
            # computation to hang. This block, though possibly slightly less
            # efficient than ``band.Fill`` will give real-time feedback about
            # how the fill is progressing.
            for offsets in iterblocks((target_path, 1), offset_only=True):
                fill_array = numpy.empty((offsets["win_ysize"], offsets["win_xsize"]))
                pixels_processed += offsets["win_ysize"] * offsets["win_xsize"]
                fill_array[:] = fill_value
                target_band.WriteArray(fill_array, offsets["xoff"], offsets["yoff"])

                last_time = _invoke_timed_callback(
                    last_time,
                    lambda: LOGGER.info(
                        f"filling new raster {target_path} with {fill_value} "
                        f"-- {float(pixels_processed)/n_pixels*100.0:.2f}% "
                        f"complete"
                    ),
                    _LOGGING_PERIOD,
                )
            target_band.FlushCache()
            target_band = None

    target_raster.FlushCache()
    target_band = None
    target_raster = None
    LOGGER.debug(f"all done with creating {target_path}")


def create_raster_from_vector_extents(
    base_vector_path,
    target_raster_path,
    target_pixel_size,
    target_pixel_type,
    target_nodata,
    fill_value=None,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Create a blank raster based on a vector file extent.

    Args:
        base_vector_path (string): path to vector shapefile to base the
            bounding box for the target raster.
        target_raster_path (string): path to location of generated geotiff;
            the upper left hand corner of this raster will be aligned with the
            bounding box of the source vector and the extent will be exactly
            equal or contained the source vector's bounding box depending on
            whether the pixel size divides evenly into the source bounding
            box; if not coordinates will be rounded up to contain the original
            extent.
        target_pixel_size (list/tuple): the x/y pixel size as a sequence
            Example::

                [30.0, -30.0]

        target_pixel_type (int): gdal GDT pixel type of target raster
        target_nodata (numeric): target nodata value. Can be None if no nodata
            value is needed.
        fill_value (int/float): value to fill in the target raster; no fill if
            value is None
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    if target_pixel_type not in _VALID_GDAL_TYPES:
        raise ValueError(
            f"Invalid target type, should be a gdal.GDT_* type, received "
            f'"{target_pixel_type}"'
        )
    # Determine the width and height of the tiff in pixels based on the
    # maximum size of the combined envelope of all the features
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    bounding_box = get_vector_info(base_vector_path)["bounding_box"]
    xwidth = numpy.subtract(*[bounding_box[i] for i in (2, 0)])
    ywidth = numpy.subtract(*[bounding_box[i] for i in (3, 1)])
    if numpy.isclose(xwidth, 0) and numpy.isclose(ywidth, 0):
        raise ValueError(
            f"bounding box appears to be empty {bounding_box} suggesting "
            f"vector has no geometry"
        )
    n_cols = abs(round(xwidth / target_pixel_size[0]))
    n_rows = abs(round(ywidth / target_pixel_size[1]))
    n_cols = max(1, n_cols)
    n_rows = max(1, n_rows)

    driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    n_bands = 1
    raster = retry_create(
        driver,
        target_raster_path,
        n_cols,
        n_rows,
        n_bands,
        target_pixel_type,
        options=raster_driver_creation_tuple[1],
    )
    raster.GetRasterBand(1).SetNoDataValue(target_nodata)

    # Set the transform based on the upper left corner and given pixel
    # dimensions
    x_source = bounding_box[0]
    y_source = bounding_box[3]
    raster_transform = [
        x_source,
        target_pixel_size[0],
        0.0,
        y_source,
        0.0,
        target_pixel_size[1],
    ]
    raster.SetGeoTransform(raster_transform)

    # Use the same projection on the raster as the shapefile
    raster.SetProjection(vector.GetLayer(0).GetSpatialRef().ExportToWkt())

    # Initialize everything to nodata
    if fill_value is not None:
        band = raster.GetRasterBand(1)
        band.Fill(fill_value)
        band = None
    vector = None
    raster = None


def interpolate_points(
    base_vector_path,
    vector_attribute_field,
    target_raster_path_band,
    interpolation_mode,
):
    """Interpolate point values onto an existing raster.

    Args:
        base_vector_path (string): path to a shapefile that contains point
            vector layers.
        vector_attribute_field (field): a string in the vector referenced at
            ``base_vector_path`` that refers to a numeric value in the
            vector's attribute table.  This is the value that will be
            interpolated across the raster.
        target_raster_path_band (tuple): a path/band number tuple to an
            existing raster which likely intersects or is nearby the source
            vector. The band in this raster will take on the interpolated
            numerical values  provided at each point.
        interpolation_mode (string): the interpolation method to use for
            scipy.interpolate.griddata, one of 'linear', near', or 'cubic'.

    Return:
        None
    """
    source_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    point_list = []
    value_list = []
    for layer_index in range(source_vector.GetLayerCount()):
        layer = source_vector.GetLayer(layer_index)
        for point_feature in layer:
            value = point_feature.GetField(vector_attribute_field)
            # Add in the numpy notation which is row, col
            # Here the point geometry is in the form x, y (col, row)
            geometry = point_feature.GetGeometryRef()
            point = geometry.GetPoint()
            point_list.append([point[1], point[0]])
            value_list.append(value)

    point_array = numpy.array(point_list)
    value_array = numpy.array(value_list)

    # getting the offsets first before the raster is opened in update mode
    offset_list = list(iterblocks(target_raster_path_band, offset_only=True))
    target_raster = gdal.OpenEx(
        target_raster_path_band[0], gdal.OF_RASTER | gdal.GA_Update
    )
    band = target_raster.GetRasterBand(target_raster_path_band[1])
    nodata = band.GetNoDataValue()
    geotransform = target_raster.GetGeoTransform()
    for offset in offset_list:
        grid_y, grid_x = numpy.mgrid[
            offset["yoff"] : offset["yoff"] + offset["win_ysize"],
            offset["xoff"] : offset["xoff"] + offset["win_xsize"],
        ]
        grid_y = grid_y * geotransform[5] + geotransform[3]
        grid_x = grid_x * geotransform[1] + geotransform[0]

        # this is to be consistent with GDAL 2.0's change of 'nearest' to
        # 'near' for an interpolation scheme that SciPy did not change.
        if interpolation_mode == "near":
            interpolation_mode = "nearest"
        raster_out_array = scipy.interpolate.griddata(
            point_array,
            value_array,
            (grid_y, grid_x),
            interpolation_mode,
            nodata,
        )
        band.WriteArray(raster_out_array, offset["xoff"], offset["yoff"])


def fast_zonal_statistics(
    base_raster_path_band,
    aggregate_vector_path,
    aggregate_layer_name=None,
    ignore_nodata=True,
    polygons_might_overlap=True,
    working_dir=None,
    clean_working_dir=True,
):
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "`base_raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), received %s" % repr(base_raster_path_band)
        )

    raster_info = get_raster_info(base_raster_path_band[0])
    raster_nodata = raster_info["nodata"][base_raster_path_band[1] - 1]
    pixel_width = abs(raster_info["pixel_size"][0])
    tolerance = pixel_width * 0.5

    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    projected_vector_path = os.path.join(temp_working_dir, "projected_vector.gpkg")
    gdal.VectorTranslate(
        projected_vector_path,
        aggregate_vector_path,
        dstSRS=raster_info["projection_wkt"],
        simplifyTolerance=tolerance,
        format="GPKG",
    )

    aggregate_vector = gdal.OpenEx(projected_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % projected_vector_path
        )
    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()
    if aggregate_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s"
            % (aggregate_layer_name, projected_vector_path)
        )

    raster_bbox = raster_info["bounding_box"]
    vec_extent = aggregate_layer.GetExtent()  # (minx, maxx, miny, maxy)
    if (
        vec_extent[1] < raster_bbox[0]
        or vec_extent[0] > raster_bbox[1]
        or vec_extent[3] < raster_bbox[2]
        or vec_extent[2] > raster_bbox[3]
    ):
        LOGGER.error(
            "aggregate vector %s does not intersect with the raster %s",
            aggregate_vector_path,
            base_raster_path_band[0],
        )
        aggregate_stats = collections.defaultdict(
            lambda: {
                "min": None,
                "max": None,
                "count": 0,
                "nodata_count": 0,
                "sum": 0.0,
            }
        )
        for feature in aggregate_layer:
            _ = aggregate_stats[feature.GetFID()]
        if clean_working_dir:
            shutil.rmtree(temp_working_dir)
        return dict(aggregate_stats)

    clipped_raster_path = base_raster_path_band[0]
    clipped_raster = gdal.OpenEx(clipped_raster_path, gdal.OF_RASTER)
    clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])

    local_aggregate_field_name = "original_fid"
    rasterize_layer_args = {
        "options": [
            "ALL_TOUCHED=FALSE",
            "ATTRIBUTE=%s" % local_aggregate_field_name,
        ]
    }

    driver = ogr.GetDriverByName("MEMORY")
    disjoint_vector = driver.CreateDataSource("disjoint_vector")
    spat_ref = aggregate_layer.GetSpatialRef()

    aggregate_layer_fid_set = {feat.GetFID() for feat in aggregate_layer}
    agg_feat = None
    if polygons_might_overlap:
        disjoint_fid_sets = calculate_disjoint_polygon_set(
            projected_vector_path, bounding_box=raster_bbox
        )
    else:
        disjoint_fid_sets = [aggregate_layer_fid_set]

    aggregate_stats = collections.defaultdict(
        lambda: {
            "min": None,
            "max": None,
            "count": 0,
            "nodata_count": 0,
            "sum": 0.0,
        }
    )
    last_time = time.time()
    LOGGER.info("processing %d disjoint polygon sets", len(disjoint_fid_sets))
    for set_index, disjoint_fid_set in enumerate(disjoint_fid_sets):
        last_time = _invoke_timed_callback(
            last_time,
            lambda: LOGGER.info(
                "zonal stats approximately %.1f%% complete on %s",
                100.0 * float(set_index + 1) / len(disjoint_fid_sets),
                os.path.basename(projected_vector_path),
            ),
            _LOGGING_PERIOD,
        )

        agg_fid_raster_path = os.path.join(temp_working_dir, f"agg_fid_{set_index}.tif")
        agg_fid_nodata = -1
        new_raster_from_base(
            clipped_raster_path,
            agg_fid_raster_path,
            gdal.GDT_Int32,
            [agg_fid_nodata],
        )
        agg_fid_offset_list = list(
            iterblocks((agg_fid_raster_path, 1), offset_only=True)
        )
        agg_fid_raster = gdal.OpenEx(
            agg_fid_raster_path, gdal.GA_Update | gdal.OF_RASTER
        )
        agg_fid_band = agg_fid_raster.GetRasterBand(1)

        disjoint_layer = disjoint_vector.CreateLayer(
            "disjoint_vector", spat_ref, ogr.wkbPolygon
        )
        disjoint_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger)
        )
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()
        disjoint_layer.StartTransaction()
        for index, feature_fid in enumerate(disjoint_fid_set):
            last_time = _invoke_timed_callback(
                last_time,
                lambda: LOGGER.info(
                    "polygon set %d of %d approximately %.1f%% processed on %s",
                    set_index + 1,
                    len(disjoint_fid_sets),
                    100.0 * float(index + 1) / len(disjoint_fid_set),
                    os.path.basename(projected_vector_path),
                ),
                _LOGGING_PERIOD,
            )
            agg_feat = aggregate_layer.GetFeature(feature_fid)
            agg_geom_ref = agg_feat.GetGeometryRef()
            disjoint_feat = ogr.Feature(disjoint_layer_defn)
            disjoint_feat.SetGeometry(agg_geom_ref.Clone())
            agg_geom_ref = None
            disjoint_feat.SetField(local_aggregate_field_name, feature_fid)
            disjoint_layer.CreateFeature(disjoint_feat)
        agg_feat = None
        disjoint_layer.CommitTransaction()

        rasterize_callback = _make_logger_callback(
            "rasterizing polygon "
            + str(set_index + 1)
            + " of "
            + str(len(disjoint_fid_set))
            + " set %.1f%% complete %s"
        )
        gdal.RasterizeLayer(
            agg_fid_raster,
            [1],
            disjoint_layer,
            callback=rasterize_callback,
            **rasterize_layer_args,
        )
        agg_fid_raster.FlushCache()
        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        for agg_fid_offset in agg_fid_offset_list:
            agg_fid_block = agg_fid_band.ReadAsArray(**agg_fid_offset)
            clipped_block = clipped_band.ReadAsArray(**agg_fid_offset)
            valid_mask = agg_fid_block != agg_fid_nodata
            valid_agg_fids = agg_fid_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]
            for agg_fid in numpy.unique(valid_agg_fids):
                masked_clipped_block = valid_clipped[valid_agg_fids == agg_fid]
                if raster_nodata is not None:
                    clipped_nodata_mask = numpy.isclose(
                        masked_clipped_block, raster_nodata
                    )
                else:
                    clipped_nodata_mask = numpy.zeros(
                        masked_clipped_block.shape, dtype=bool
                    )
                aggregate_stats[agg_fid]["nodata_count"] += numpy.count_nonzero(
                    clipped_nodata_mask
                )
                if ignore_nodata:
                    masked_clipped_block = masked_clipped_block[~clipped_nodata_mask]
                if masked_clipped_block.size == 0:
                    continue
                if aggregate_stats[agg_fid]["min"] is None:
                    aggregate_stats[agg_fid]["min"] = masked_clipped_block[0]
                    aggregate_stats[agg_fid]["max"] = masked_clipped_block[0]
                aggregate_stats[agg_fid]["min"] = min(
                    numpy.min(masked_clipped_block),
                    aggregate_stats[agg_fid]["min"],
                )
                aggregate_stats[agg_fid]["max"] = max(
                    numpy.max(masked_clipped_block),
                    aggregate_stats[agg_fid]["max"],
                )
                aggregate_stats[agg_fid]["count"] += masked_clipped_block.size
                aggregate_stats[agg_fid]["sum"] += numpy.sum(masked_clipped_block)
        agg_fid_band = None
        agg_fid_raster = None

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    clipped_gt = numpy.array(clipped_raster.GetGeoTransform(), dtype=numpy.float32)
    for unset_fid in unset_fids:
        unset_feat = aggregate_layer.GetFeature(unset_fid)
        unset_geom_ref = unset_feat.GetGeometryRef()
        if unset_geom_ref is None:
            LOGGER.warn(f"no geometry in {projected_vector_path} FID: {unset_fid}")
            continue
        shapely_geom = shapely.wkb.loads(bytes(unset_geom_ref.ExportToWkb()))
        try:
            shapely_geom_list = list(shapely_geom)
        except TypeError:
            shapely_geom_list = [shapely_geom]
        unset_geom_ref = None
        for shapely_geom in shapely_geom_list:
            single_geom = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
            unset_geom_envelope = list(single_geom.GetEnvelope())
            single_geom = None
            if clipped_gt[1] < 0:
                unset_geom_envelope[0], unset_geom_envelope[1] = (
                    unset_geom_envelope[1],
                    unset_geom_envelope[0],
                )
            if clipped_gt[5] < 0:
                unset_geom_envelope[2], unset_geom_envelope[3] = (
                    unset_geom_envelope[3],
                    unset_geom_envelope[2],
                )

            xoff = int((unset_geom_envelope[0] - clipped_gt[0]) / clipped_gt[1])
            yoff = int((unset_geom_envelope[2] - clipped_gt[3]) / clipped_gt[5])
            win_xsize = (
                int(
                    numpy.ceil((unset_geom_envelope[1] - clipped_gt[0]) / clipped_gt[1])
                )
                - xoff
            )
            win_ysize = (
                int(
                    numpy.ceil((unset_geom_envelope[3] - clipped_gt[3]) / clipped_gt[5])
                )
                - yoff
            )
            if xoff < 0:
                win_xsize += xoff
                xoff = 0
            if yoff < 0:
                win_ysize += yoff
                yoff = 0
            if xoff + win_xsize > clipped_band.XSize:
                win_xsize = clipped_band.XSize - xoff
            if yoff + win_ysize > clipped_band.YSize:
                win_ysize = clipped_band.YSize - yoff
            if win_xsize <= 0 or win_ysize <= 0:
                continue

            unset_fid_block = clipped_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize
            )
            if raster_nodata is not None:
                unset_fid_nodata_mask = numpy.isclose(unset_fid_block, raster_nodata)
            else:
                unset_fid_nodata_mask = numpy.zeros(unset_fid_block.shape, dtype=bool)
            valid_unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
            if valid_unset_fid_block.size == 0:
                aggregate_stats[unset_fid]["min"] = 0.0
                aggregate_stats[unset_fid]["max"] = 0.0
                aggregate_stats[unset_fid]["sum"] = 0.0
            else:
                aggregate_stats[unset_fid]["min"] = numpy.min(valid_unset_fid_block)
                aggregate_stats[unset_fid]["max"] = numpy.max(valid_unset_fid_block)
                aggregate_stats[unset_fid]["sum"] = numpy.sum(valid_unset_fid_block)
            aggregate_stats[unset_fid]["count"] = valid_unset_fid_block.size
            aggregate_stats[unset_fid]["nodata_count"] = numpy.count_nonzero(
                unset_fid_nodata_mask
            )

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    for fid in unset_fids:
        _ = aggregate_stats[fid]

    spat_ref = None
    clipped_band = None
    clipped_raster = None
    disjoint_layer = None
    disjoint_vector = None
    aggregate_layer = None
    aggregate_vector = None

    if clean_working_dir:
        shutil.rmtree(temp_working_dir)
    return dict(aggregate_stats)


def zonal_statistics(
    base_raster_path_band,
    aggregate_vector_path,
    aggregate_layer_name=None,
    ignore_nodata=True,
    polygons_might_overlap=True,
    working_dir=None,
    clean_working_dir=True,
):
    """Collect stats on pixel values which lie within polygons.

    This function summarizes raster statistics including min, max,
    mean, and pixel count over the regions on the raster that are
    overlapped by the polygons in the vector layer. Statistics are calculated
    in two passes, where first polygons aggregate over pixels in the raster
    whose centers intersect with the polygon. In the second pass, any polygons
    that are not aggregated use their bounding box to intersect with the
    raster for overlap statistics.

    Note:
        There may be some degenerate cases where the bounding box vs. actual
        geometry intersection would be incorrect, but these are so unlikely as
        to be manually constructed. If you encounter one of these please email
        the description and dataset to richsharp@stanford.edu.

    Args:
        base_raster_path_band (tuple): a str/int tuple indicating the path to
            the base raster and the band index of that raster to analyze.
        aggregate_vector_path (string): a path to a polygon vector whose
            geometric features indicate the areas in
            ``base_raster_path_band`` to calculate zonal statistics.
        aggregate_layer_name (string): name of shapefile layer that will be
            used to aggregate results over.  If set to None, the first layer
            in the DataSource will be used as retrieved by ``.GetLayer()``.
            Note: it is normal and expected to set this field at None if the
            aggregating shapefile is a single layer as many shapefiles,
            including the common 'ESRI Shapefile', are.
        ignore_nodata: if true, then nodata pixels are not accounted for when
            calculating min, max, count, or mean.  However, the value of
            ``nodata_count`` will always be the number of nodata pixels
            aggregated under the polygon.
        polygons_might_overlap (boolean): if True the function calculates
            aggregation coverage close to optimally by rasterizing sets of
            polygons that don't overlap.  However, this step can be
            computationally expensive for cases where there are many polygons.
              this flag to False directs the function rasterize in one
            step.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        clean_working_dir (bool): If false the temporary files used to
            calculate zonal stats are not deleted.

    Return:
        nested dictionary indexed by aggregating feature id, and then by one
        of 'min' 'max' 'sum' 'count' and 'nodata_count'.  Example::

            {0: {'min': 0,
                 'max': 1,
                 'sum': 1.7,
                 'count': 3,
                 'nodata_count': 1
                 }
            }

    Raises:
        ValueError
            if ``base_raster_path_band`` is incorrectly formatted.
        RuntimeError
            if the aggregate vector or layer cannot open.

    """
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "`base_raster_path_band` not formatted as expected.  Expects "
            "(path, band_index), received %s" % repr(base_raster_path_band)
        )
    aggregate_vector = gdal.OpenEx(aggregate_vector_path, gdal.OF_VECTOR)
    if aggregate_vector is None:
        raise RuntimeError(
            "Could not open aggregate vector at %s" % aggregate_vector_path
        )
    if aggregate_layer_name is not None:
        aggregate_layer = aggregate_vector.GetLayerByName(aggregate_layer_name)
    else:
        aggregate_layer = aggregate_vector.GetLayer()
    if aggregate_layer is None:
        raise RuntimeError(
            "Could not open layer %s on %s"
            % (aggregate_layer_name, aggregate_vector_path)
        )

    # create a new aggregate ID field to map base vector aggregate fields to
    # local ones that are guaranteed to be integers.
    local_aggregate_field_name = "original_fid"
    rasterize_layer_args = {
        "options": [
            "ALL_TOUCHED=FALSE",
            "ATTRIBUTE=%s" % local_aggregate_field_name,
        ]
    }

    # clip base raster to aggregating vector intersection
    raster_info = get_raster_info(base_raster_path_band[0])
    # -1 here because bands are 1 indexed
    raster_nodata = raster_info["nodata"][base_raster_path_band[1] - 1]
    temp_working_dir = tempfile.mkdtemp(dir=working_dir)
    clipped_raster_path = os.path.join(temp_working_dir, "clipped_raster.tif")

    try:
        align_and_resize_raster_stack(
            [base_raster_path_band[0]],
            [clipped_raster_path],
            ["near"],
            raster_info["pixel_size"],
            "intersection",
            base_vector_path_list=[aggregate_vector_path],
            target_projection_wkt=raster_info["projection_wkt"],
            raster_align_index=0,
        )
        clipped_raster = gdal.OpenEx(clipped_raster_path, gdal.OF_RASTER)
        clipped_band = clipped_raster.GetRasterBand(base_raster_path_band[1])
    except ValueError as e:
        if "Bounding boxes do not intersect" in repr(e):
            LOGGER.error(
                "aggregate vector %s does not intersect with the raster %s",
                aggregate_vector_path,
                base_raster_path_band,
            )
            aggregate_stats = collections.defaultdict(
                lambda: {
                    "min": None,
                    "max": None,
                    "count": 0,
                    "nodata_count": 0,
                    "sum": 0.0,
                }
            )
            for feature in aggregate_layer:
                _ = aggregate_stats[feature.GetFID()]
            return dict(aggregate_stats)
        else:
            # this would be very unexpected to get here, but if it happened
            # and we didn't raise an exception, execution could get weird.
            raise

    # make a shapefile that non-overlapping layers can be added to
    driver = ogr.GetDriverByName("MEMORY")
    disjoint_vector = driver.CreateDataSource("disjoint_vector")
    spat_ref = aggregate_layer.GetSpatialRef()

    # Initialize these dictionaries to have the shapefile fields in the
    # original datasource even if we don't pick up a value later
    LOGGER.info("build a lookup of aggregate field value to FID")

    aggregate_layer_fid_set = set([agg_feat.GetFID() for agg_feat in aggregate_layer])
    agg_feat = None
    # Loop over each polygon and aggregate
    if polygons_might_overlap:
        LOGGER.info("creating disjoint polygon set")
        disjoint_fid_sets = calculate_disjoint_polygon_set(
            aggregate_vector_path, bounding_box=raster_info["bounding_box"]
        )
    else:
        disjoint_fid_sets = [aggregate_layer_fid_set]

    aggregate_stats = collections.defaultdict(
        lambda: {
            "min": None,
            "max": None,
            "count": 0,
            "nodata_count": 0,
            "sum": 0.0,
        }
    )
    last_time = time.time()
    LOGGER.info("processing %d disjoint polygon sets", len(disjoint_fid_sets))
    for set_index, disjoint_fid_set in enumerate(disjoint_fid_sets):
        last_time = _invoke_timed_callback(
            last_time,
            lambda: LOGGER.info(
                "zonal stats approximately %.1f%% complete on %s",
                100.0 * float(set_index + 1) / len(disjoint_fid_sets),
                os.path.basename(aggregate_vector_path),
            ),
            _LOGGING_PERIOD,
        )

        agg_fid_raster_path = os.path.join(temp_working_dir, f"agg_fid_{set_index}.tif")
        agg_fid_nodata = -1
        new_raster_from_base(
            clipped_raster_path,
            agg_fid_raster_path,
            gdal.GDT_Int32,
            [agg_fid_nodata],
        )
        # fetch the block offsets before the raster is opened for writing
        agg_fid_offset_list = list(
            iterblocks((agg_fid_raster_path, 1), offset_only=True)
        )
        agg_fid_raster = gdal.OpenEx(
            agg_fid_raster_path, gdal.GA_Update | gdal.OF_RASTER
        )
        agg_fid_band = agg_fid_raster.GetRasterBand(1)

        disjoint_layer = disjoint_vector.CreateLayer(
            "disjoint_vector", spat_ref, ogr.wkbPolygon
        )
        disjoint_layer.CreateField(
            ogr.FieldDefn(local_aggregate_field_name, ogr.OFTInteger)
        )
        disjoint_layer_defn = disjoint_layer.GetLayerDefn()
        # add polygons to subset_layer
        disjoint_layer.StartTransaction()
        for index, feature_fid in enumerate(disjoint_fid_set):
            last_time = _invoke_timed_callback(
                last_time,
                lambda: LOGGER.info(
                    "polygon set %d of %d approximately %.1f%% processed " "on %s",
                    set_index + 1,
                    len(disjoint_fid_sets),
                    100.0 * float(index + 1) / len(disjoint_fid_set),
                    os.path.basename(aggregate_vector_path),
                ),
                _LOGGING_PERIOD,
            )
            agg_feat = aggregate_layer.GetFeature(feature_fid)
            agg_geom_ref = agg_feat.GetGeometryRef()
            disjoint_feat = ogr.Feature(disjoint_layer_defn)
            disjoint_feat.SetGeometry(agg_geom_ref.Clone())
            agg_geom_ref = None
            disjoint_feat.SetField(local_aggregate_field_name, feature_fid)
            disjoint_layer.CreateFeature(disjoint_feat)
        agg_feat = None
        disjoint_layer.CommitTransaction()

        LOGGER.info(
            "disjoint polygon set %d of %d 100.0%% processed on %s",
            set_index + 1,
            len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path),
        )

        LOGGER.info(
            "rasterizing disjoint polygon set %d of %d %s",
            set_index + 1,
            len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path),
        )
        rasterize_callback = _make_logger_callback(
            "rasterizing polygon "
            + str(set_index + 1)
            + " of "
            + str(len(disjoint_fid_sets))
            + " set %.1f%% complete %s"
        )
        gdal.RasterizeLayer(
            agg_fid_raster,
            [1],
            disjoint_layer,
            callback=rasterize_callback,
            **rasterize_layer_args,
        )
        agg_fid_raster.FlushCache()

        # Delete the features we just added to the subset_layer
        disjoint_layer = None
        disjoint_vector.DeleteLayer(0)

        # create a key array
        # and parallel min, max, count, and nodata count arrays
        LOGGER.info(
            "summarizing rasterized disjoint polygon set %d of %d %s",
            set_index + 1,
            len(disjoint_fid_sets),
            os.path.basename(aggregate_vector_path),
        )
        for agg_fid_offset in agg_fid_offset_list:
            agg_fid_block = agg_fid_band.ReadAsArray(**agg_fid_offset)
            clipped_block = clipped_band.ReadAsArray(**agg_fid_offset)
            valid_mask = (agg_fid_block != agg_fid_nodata) & (
                numpy.isfinite(clipped_block)
            )
            valid_agg_fids = agg_fid_block[valid_mask]
            valid_clipped = clipped_block[valid_mask]
            for agg_fid in numpy.unique(valid_agg_fids):
                masked_clipped_block = valid_clipped[valid_agg_fids == agg_fid]
                if raster_nodata is not None:
                    clipped_nodata_mask = numpy.isclose(
                        masked_clipped_block, raster_nodata
                    )
                else:
                    clipped_nodata_mask = numpy.zeros(
                        masked_clipped_block.shape, dtype=bool
                    )
                aggregate_stats[agg_fid]["nodata_count"] += numpy.count_nonzero(
                    clipped_nodata_mask
                )
                if ignore_nodata:
                    masked_clipped_block = masked_clipped_block[~clipped_nodata_mask]
                if masked_clipped_block.size == 0:
                    continue

                if aggregate_stats[agg_fid]["min"] is None:
                    aggregate_stats[agg_fid]["min"] = masked_clipped_block[0]
                    aggregate_stats[agg_fid]["max"] = masked_clipped_block[0]

                aggregate_stats[agg_fid]["min"] = min(
                    numpy.min(masked_clipped_block),
                    aggregate_stats[agg_fid]["min"],
                )
                aggregate_stats[agg_fid]["max"] = max(
                    numpy.max(masked_clipped_block),
                    aggregate_stats[agg_fid]["max"],
                )
                aggregate_stats[agg_fid]["count"] += masked_clipped_block.size
                aggregate_stats[agg_fid]["sum"] += numpy.sum(masked_clipped_block)
        agg_fid_band = None
        agg_fid_raster = None
    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug("unset_fids: %s of %s ", len(unset_fids), len(aggregate_layer_fid_set))
    clipped_gt = numpy.array(clipped_raster.GetGeoTransform(), dtype=numpy.float32)
    LOGGER.debug("gt %s for %s", clipped_gt, base_raster_path_band)
    for unset_fid in unset_fids:
        unset_feat = aggregate_layer.GetFeature(unset_fid)
        unset_geom_ref = unset_feat.GetGeometryRef()
        if unset_geom_ref is None:
            LOGGER.warn(f"no geometry in {aggregate_vector_path} FID: {unset_fid}")
            continue
        # fetch a shapely polygon and turn it into a list of polygons in the
        # case that it is a multipolygon
        shapely_geom = shapely.wkb.loads(bytes(unset_geom_ref.ExportToWkb()))
        try:
            # a non multipolygon will raise a TypeError
            shapely_geom_list = list(shapely_geom)
        except TypeError:
            shapely_geom_list = [shapely_geom]
        unset_geom_ref = None
        for shapely_geom in shapely_geom_list:
            single_geom = ogr.CreateGeometryFromWkt(shapely_geom.wkt)
            unset_geom_envelope = list(single_geom.GetEnvelope())
            single_geom = None
            unset_feat = None
            if clipped_gt[1] < 0:
                unset_geom_envelope[0], unset_geom_envelope[1] = (
                    unset_geom_envelope[1],
                    unset_geom_envelope[0],
                )
            if clipped_gt[5] < 0:
                unset_geom_envelope[2], unset_geom_envelope[3] = (
                    unset_geom_envelope[3],
                    unset_geom_envelope[2],
                )

            xoff = int((unset_geom_envelope[0] - clipped_gt[0]) / clipped_gt[1])
            yoff = int((unset_geom_envelope[2] - clipped_gt[3]) / clipped_gt[5])
            win_xsize = (
                int(
                    numpy.ceil((unset_geom_envelope[1] - clipped_gt[0]) / clipped_gt[1])
                )
                - xoff
            )
            win_ysize = (
                int(
                    numpy.ceil((unset_geom_envelope[3] - clipped_gt[3]) / clipped_gt[5])
                )
                - yoff
            )

            # clamp offset to the side of the raster if it's negative
            if xoff < 0:
                win_xsize += xoff
                xoff = 0
            if yoff < 0:
                win_ysize += yoff
                yoff = 0

            # clamp the window to the side of the raster if too big
            if xoff + win_xsize > clipped_band.XSize:
                win_xsize = clipped_band.XSize - xoff
            if yoff + win_ysize > clipped_band.YSize:
                win_ysize = clipped_band.YSize - yoff

            if win_xsize <= 0 or win_ysize <= 0:
                continue

            # here we consider the pixels that intersect with the geometry's
            # bounding box as being the proxy for the intersection with the
            # polygon itself. This is not a bad approximation since the case
            # that caused the polygon to be skipped in the first phase is that it
            # is as small as a pixel. There could be some degenerate cases that
            # make this estimation very wrong, but we do not know of any that
            # would come from natural data. If you do encounter such a dataset
            # please email the description and datset to richsharp@stanford.edu.
            unset_fid_block = clipped_band.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=win_xsize, win_ysize=win_ysize
            )

            if raster_nodata is not None:
                unset_fid_nodata_mask = numpy.isclose(unset_fid_block, raster_nodata)
            else:
                unset_fid_nodata_mask = numpy.zeros(unset_fid_block.shape, dtype=bool)
            # guard against crazy values
            unset_fid_nodata_mask |= ~numpy.isfinite(unset_fid_block)

            valid_unset_fid_block = unset_fid_block[~unset_fid_nodata_mask]
            if valid_unset_fid_block.size == 0:
                aggregate_stats[unset_fid]["min"] = 0.0
                aggregate_stats[unset_fid]["max"] = 0.0
                aggregate_stats[unset_fid]["sum"] = 0.0
            else:
                aggregate_stats[unset_fid]["min"] = numpy.min(valid_unset_fid_block)
                aggregate_stats[unset_fid]["max"] = numpy.max(valid_unset_fid_block)
                aggregate_stats[unset_fid]["sum"] = numpy.sum(valid_unset_fid_block)
            aggregate_stats[unset_fid]["count"] = valid_unset_fid_block.size
            aggregate_stats[unset_fid]["nodata_count"] = numpy.count_nonzero(
                unset_fid_nodata_mask
            )

    unset_fids = aggregate_layer_fid_set.difference(aggregate_stats)
    LOGGER.debug(
        "remaining unset_fids: %s of %s ",
        len(unset_fids),
        len(aggregate_layer_fid_set),
    )
    # fill in the missing polygon fids in the aggregate stats by invoking the
    # accessor in the defaultdict
    for fid in unset_fids:
        _ = aggregate_stats[fid]

    LOGGER.info(
        "all done processing polygon sets for %s",
        os.path.basename(aggregate_vector_path),
    )

    # clean up temporary files
    spat_ref = None
    clipped_band = None
    clipped_raster = None
    disjoint_layer = None
    disjoint_vector = None
    aggregate_layer = None
    aggregate_vector = None

    if clean_working_dir:
        shutil.rmtree(temp_working_dir)
    return dict(aggregate_stats)


def get_vector_info(vector_path, layer_id=0):
    """Get information about an GDAL vector.

    Args:
        vector_path (str): a path to a GDAL vector.
        layer_id (str/int): name or index of underlying layer to analyze.
            Defaults to 0.

    Raises:
        ValueError if ``vector_path`` does not exist on disk or cannot be
        opened as a gdal.OF_VECTOR.

    Return:
        raster_properties (dictionary):
            a dictionary with the following key-value pairs:

            * ``'projection_wkt'`` (string): projection of the vector in Well
              Known Text.
            * ``'bounding_box'`` (sequence): sequence of floats representing
              the bounding box in projected coordinates in the order
              [minx, miny, maxx, maxy].
            * ``'file_list'`` (sequence): sequence of string paths to the files
              that make up this vector.
            * ``'feature_count'`` (int): number of features in the layer.
            * ``'geometry_type'`` (int): OGR geometry type of the layer.

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    if not vector:
        raise ValueError("Could not open %s as a gdal.OF_VECTOR" % vector_path)
    vector_properties = {}
    vector_properties["file_list"] = vector.GetFileList()
    layer = vector.GetLayer(iLayer=layer_id)
    # projection is same for all layers, so just use the first one
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref:
        vector_projection_wkt = spatial_ref.ExportToWkt()
    else:
        vector_projection_wkt = None
    vector_properties["projection_wkt"] = vector_projection_wkt
    layer_bb = layer.GetExtent()
    # convert form [minx,maxx,miny,maxy] to [minx,miny,maxx,maxy]
    vector_properties["bounding_box"] = [layer_bb[i] for i in [0, 2, 1, 3]]
    vector_properties["feature_count"] = layer.GetFeatureCount()
    vector_properties["geometry_type"] = layer.GetGeomType()
    layer = None
    vector = None
    return vector_properties


def get_raster_info(raster_path):
    """Get information about a GDAL raster (dataset).

    Args:
       raster_path (String): a path to a GDAL raster.

    Raises:
        ValueError
            if ``raster_path`` is not a file or cannot be opened as a
            ``gdal.OF_RASTER``.

    Return:
        raster_properties (dictionary):
            a dictionary with the properties stored under relevant keys.

        * ``'pixel_size'`` (tuple): (pixel x-size, pixel y-size)
          from geotransform.
        * ``'raster_size'`` (tuple):  number of raster pixels in (x, y)
          direction.
        * ``'nodata'`` (sequence): a sequence of the nodata values in the bands
          of the raster in the same order as increasing band index.
        * ``'n_bands'`` (int): number of bands in the raster.
        * ``'geotransform'`` (tuple): a 6-tuple representing the geotransform
          of (x orign, x-increase, xy-increase, y origin, yx-increase,
          y-increase).
        * ``'datatype'`` (int): An instance of an enumerated gdal.GDT_* int
          that represents the datatype of the raster.
        * ``'projection_wkt'`` (string): projection of the raster in Well Known
          Text.
        * ``'bounding_box'`` (sequence): sequence of floats representing the
          bounding box in projected coordinates in the order
          [minx, miny, maxx, maxy]
        * ``'block_size'`` (tuple): underlying x/y raster block size for
          efficient reading.
        * ``'numpy_type'`` (numpy type): this is the equivalent numpy datatype
          for the raster bands including signed bytes.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    if not raster:
        raise ValueError("Could not open %s as a gdal.OF_RASTER" % raster_path)
    raster_properties = {}
    raster_properties["file_list"] = raster.GetFileList()
    projection_wkt = raster.GetProjection()
    if not projection_wkt:
        projection_wkt = None
    raster_properties["projection_wkt"] = projection_wkt
    geo_transform = raster.GetGeoTransform()
    raster_properties["geotransform"] = geo_transform
    raster_properties["pixel_size"] = (geo_transform[1], geo_transform[5])
    raster_properties["raster_size"] = (
        raster.GetRasterBand(1).XSize,
        raster.GetRasterBand(1).YSize,
    )
    raster_properties["n_bands"] = raster.RasterCount
    raster_properties["nodata"] = [
        raster.GetRasterBand(index).GetNoDataValue()
        for index in range(1, raster_properties["n_bands"] + 1)
    ]
    # blocksize is the same for all bands, so we can just get the first
    raster_properties["block_size"] = raster.GetRasterBand(1).GetBlockSize()

    # we dont' really know how the geotransform is laid out, all we can do is
    # calculate the x and y bounds, then take the appropriate min/max
    x_bounds = [
        geo_transform[0],
        geo_transform[0]
        + raster_properties["raster_size"][0] * geo_transform[1]
        + raster_properties["raster_size"][1] * geo_transform[2],
    ]
    y_bounds = [
        geo_transform[3],
        geo_transform[3]
        + raster_properties["raster_size"][0] * geo_transform[4]
        + raster_properties["raster_size"][1] * geo_transform[5],
    ]

    raster_properties["bounding_box"] = [
        numpy.min(x_bounds),
        numpy.min(y_bounds),
        numpy.max(x_bounds),
        numpy.max(y_bounds),
    ]

    # datatype is the same for the whole raster, but is associated with band
    band = raster.GetRasterBand(1)
    band_datatype = band.DataType
    raster_properties["datatype"] = band_datatype
    raster_properties["numpy_type"] = _GDAL_TYPE_TO_NUMPY_LOOKUP[band_datatype]
    # this part checks to see if the byte is signed or not
    if band_datatype == gdal.GDT_Byte:
        metadata = band.GetMetadata("IMAGE_STRUCTURE")
        if "PIXELTYPE" in metadata and metadata["PIXELTYPE"] == "SIGNEDBYTE":
            raster_properties["numpy_type"] = numpy.int8
    band = None
    raster = None
    return raster_properties


def reproject_vector(
    base_vector_path,
    target_projection_wkt,
    target_path,
    layer_id=0,
    driver_name="GPKG",
    copy_fields=True,
    geometry_type=None,
    simplify_tol=None,
    where_filter=None,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
):
    """Reproject OGR DataSource (vector).

    Transforms the features of the base vector to the desired output
    projection in a new ESRI Shapefile.

    Args:
        base_vector_path (string): Path to the base shapefile to transform.
        target_projection_wkt (string): the desired output projection in Well
            Known Text (by layer.GetSpatialRef().ExportToWkt())
        target_path (string): the filepath to the transformed shapefile
        layer_id (str/int): name or index of layer in ``base_vector_path`` to
            reproject. Defaults to 0.
        driver_name (string): String to pass to ogr.GetDriverByName, defaults
            to 'ESRI Shapefile'.
        copy_fields (bool or iterable): If True, all the fields in
            ``base_vector_path`` will be copied to ``target_path`` during the
            reprojection step. If it is an iterable, it will contain the
            field names to exclusively copy. An unmatched fieldname will be
            ignored. If ``False`` no fields are copied into the new vector.
        geometry_type (int): enumerated type of target layer, default is
            None which defaults to the base geometry type.
        simplify_tol (float): if not None, a positive value in the target
            projected coordinate units to simplify the underlying
            geometry.
        where_filter (str): if not None, a string that can be baseed to a
            "SetAttributeFilter" call to subset features for reprojecting.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Return:
        None
    """
    base_vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)

    # if this file already exists, then remove it
    if os.path.isfile(target_path):
        LOGGER.warning("%s already exists, removing and overwriting", target_path)
        os.remove(target_path)

    target_sr = osr.SpatialReference(target_projection_wkt)

    # create a new shapefile from the orginal_datasource
    target_driver = ogr.GetDriverByName(driver_name)
    target_vector = target_driver.CreateDataSource(target_path)

    layer = base_vector.GetLayer(layer_id)
    if where_filter is not None:
        LOGGER.debug(where_filter)
        layer.SetAttributeFilter(where_filter)
    layer_dfn = layer.GetLayerDefn()

    if geometry_type is None:
        geometry_type = layer_dfn.GetGeomType()

    target_layer = target_vector.CreateLayer(
        layer_dfn.GetName(), target_sr, geometry_type
    )

    # this will map the target field index to the base index it came from
    # in case we don't need to copy all the fields
    target_to_base_field_id_map = {}
    if copy_fields:
        # Get the number of fields in original_layer
        original_field_count = layer_dfn.GetFieldCount()
        # For every field that's copying, create a duplicate field in the
        # new layer

        for fld_index in range(original_field_count):
            original_field = layer_dfn.GetFieldDefn(fld_index)
            field_name = original_field.GetName()
            if copy_fields is True or field_name in copy_fields:
                target_field = ogr.FieldDefn(field_name, original_field.GetType())
                target_layer.CreateField(target_field)
                target_to_base_field_id_map[fld_index] = len(
                    target_to_base_field_id_map
                )

    # Get the SR of the original_layer to use in transforming
    base_sr = layer.GetSpatialRef()

    base_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_sr.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    # Create a coordinate transformation
    coord_trans = osr.CreateCoordinateTransformation(base_sr, target_sr)

    # Copy all of the features in layer to the new shapefile
    target_layer.StartTransaction()
    error_count = 0
    last_time = time.time()
    LOGGER.info("starting reprojection")
    for feature_index, base_feature in enumerate(layer):
        last_time = _invoke_timed_callback(
            last_time,
            lambda: LOGGER.info(
                "reprojection approximately %.1f%% complete on %s",
                100.0 * float(feature_index + 1) / (layer.GetFeatureCount()),
                os.path.basename(target_path),
            ),
            _LOGGING_PERIOD,
        )

        geom = base_feature.GetGeometryRef()
        if geom is None:
            # we encountered this error occasionally when transforming clipped
            # global polygons.  Not clear what is happening but perhaps a
            # feature was retained that otherwise wouldn't have been included
            # in the clip
            error_count += 1
            continue

        # Transform geometry into format desired for the new projection
        error_code = geom.Transform(coord_trans)
        if error_code != 0:  # error
            # this could be caused by an out of range transformation
            # whatever the case, don't put the transformed poly into the
            # output set
            error_count += 1
            continue

        if simplify_tol is not None:
            geom = geom.Simplify(simplify_tol)

        # Copy original_datasource's feature and set as new shapes feature
        target_feature = ogr.Feature(target_layer.GetLayerDefn())
        target_feature.SetGeometry(geom)

        # For all the fields in the feature set the field values from the
        # source field
        for target_index, base_index in target_to_base_field_id_map.items():
            target_feature.SetField(target_index, base_feature.GetField(base_index))

        target_layer.CreateFeature(target_feature)
        target_feature = None
        base_feature = None
    target_layer.CommitTransaction()
    LOGGER.info("reprojection 100.0%% complete on %s", os.path.basename(target_path))
    if error_count > 0:
        LOGGER.warning(
            "%d features out of %d were unable to be transformed and are"
            " not in the output vector at %s",
            error_count,
            layer.GetFeatureCount(),
            target_path,
        )
    layer = None
    base_vector = None


def reclassify_raster(
    base_raster_path_band,
    value_map,
    target_raster_path,
    target_datatype,
    target_nodata,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Reclassify pixel values in a raster.

    A function to reclassify values in raster to any output type. By default
    the values except for nodata must be in ``value_map``.

    Args:
        base_raster_path_band (tuple): a tuple including file path to a raster
            and the band index to operate over. ex: (path, band_index)
        value_map (dictionary): a dictionary of values of
            {source_value: dest_value, ...} where source_value's type is the
            same as the values in ``base_raster_path`` at band ``band_index``.
            Must contain at least one value.
        target_raster_path (string): target raster output path; overwritten if
            it exists
        target_datatype (gdal type): the numerical type for the target raster
        target_nodata (numerical type): the nodata value for the target raster
            Must be the same type as target_datatype
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    if len(value_map) == 0:
        raise ValueError("value_map must contain at least one value")
    if not _is_raster_path_band_formatted(base_raster_path_band):
        raise ValueError(
            "Expected a (path, band_id) tuple, instead got '%s'" % base_raster_path_band
        )
    raster_info = get_raster_info(base_raster_path_band[0])
    nodata = raster_info["nodata"][base_raster_path_band[1] - 1]
    value_map_copy = value_map.copy()
    # possible that nodata value is not defined, so test for None first
    # otherwise if nodata not predefined, remap it into the dictionary
    if (
        nodata is not None
        and nodata not in value_map_copy
        and target_nodata is not None
    ):
        value_map_copy[nodata] = target_nodata
    keys = sorted(numpy.array(list(value_map_copy.keys())))
    values = numpy.array([value_map_copy[x] for x in keys])

    def _map_dataset_to_value_op(original_values):
        """Convert a block of original values to the lookup values."""
        nonlocal keys
        nonlocal values
        unique = numpy.unique(original_values)
        missing_keys = numpy.setdiff1d(unique, keys)
        if missing_keys.size > 0:
            if target_nodata is None:
                raise ValueError(
                    f'The following raster values "{missing_keys}" are '
                    f"missing from the value_map, but no target nodata value "
                    f"is defined. Either define a mapping for the missing "
                    f"keys or set a nodata value in the input raster."
                )
            else:
                value_map_copy.update({key: target_nodata for key in missing_keys})
                keys = sorted(numpy.array(list(value_map_copy.keys())))
                values = numpy.array([value_map_copy[x] for x in keys])

        index = numpy.digitize(original_values.ravel(), keys, right=True)
        result = values[index].reshape(original_values.shape)
        return result

    single_thread_raster_calculator(
        [base_raster_path_band],
        _map_dataset_to_value_op,
        target_raster_path,
        target_datatype,
        target_nodata,
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )


def warp_raster(
    base_raster_path,
    target_pixel_size,
    target_raster_path,
    resample_method,
    band_id=None,
    target_bb=None,
    base_projection_wkt=None,
    target_projection_wkt=None,
    n_threads=None,
    vector_mask_options=None,
    gdal_warp_options=None,
    gdal_warp_kwargs=None,
    working_dir=None,
    output_type=gdal.GDT_Unknown,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
):
    """Resize/resample raster to desired pixel size, bbox and projection.

    Args:
        base_raster_path (string): path to base raster.
        target_pixel_size (list/tuple): a two element sequence indicating
            the x and y pixel size in projected units.
        target_raster_path (string): the location of the resized and
            resampled raster.
        resample_method (string): the resampling technique, one of
            ``near|bilinear|cubic|cubicspline|lanczos|average|mode|max|min|med|q1|q3``
        band_id (int): if not None, the resulting raster is only the indicated band
            id from `base_raster_path`, otherwise all bands in `base_raster_path`
            are warped.
        target_bb (sequence): if None, target bounding box is the same as the
            source bounding box.  Otherwise it's a sequence of float
            describing target bounding box in target coordinate system as
            [minx, miny, maxx, maxy].
        base_projection_wkt (string): if not None, interpret the projection of
            ``base_raster_path`` as this.
        target_projection_wkt (string): if not None, desired target projection
            in Well Known Text format.
        n_threads (int): optional, if not None this sets the ``N_THREADS``
            option for ``gdal.Warp``.
        vector_mask_options (dict): optional, if not None, this is a
            dictionary of options to use an existing vector's geometry to
            mask out pixels in the target raster that do not overlap the
            vector's geometry. Keys to this dictionary are:

            * ``'mask_vector_path'``: (str) path to the mask vector file. This
              vector will be automatically projected to the target
              projection if its base coordinate system does not match
              the target.
            * ``'mask_layer_id'``: (int/str) the layer index or name to use
              for masking, if this key is not in the dictionary the default
              is to use the layer at index 0.
            * ``'mask_vector_where_filter'``: (str) an SQL WHERE string that
              can be used to filter the geometry in the mask. Ex:
              'id > 10' would use all features whose field value of
              'id' is > 10.
            * ``'all_touched'``: (bool) this value is passed to the
              ALL_TOUCHED option of vector clipping, if not passed it is set
              to False.
            * ``'target_mask_value'``: (numeric), if not None, sets this to be
                the value of masked pixels, useful if nodata is not defined
                on input raster.

        gdal_warp_options (sequence): if present, the contents of this list
            are passed to the ``warpOptions`` parameter of ``gdal.Warp``. See
            the GDAL Warp documentation for valid options.
        working_dir (string): if defined uses this directory to make
            temporary working files for calculation. Otherwise uses system's
            temp directory.
        output_type (gdal type): if set, force the output image bands to have
            a specific GDAL data type supported by the driver.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This parameter
            should not be changed unless you know what you are doing.

    Return:
        None

    Raises:
        ValueError
            if ``pixel_size`` is not a 2 element sequence of numbers.
        ValueError
            if ``vector_mask_options`` is not None but the
            ``mask_vector_path`` is undefined or doesn't point to a valid
            file.

    """
    LOGGER.debug(f"about to warp {base_raster_path}")
    _assert_is_valid_pixel_size(target_pixel_size)

    base_raster_info = get_raster_info(base_raster_path)
    if target_projection_wkt is None:
        target_projection_wkt = base_raster_info["projection_wkt"]

    if target_bb is None:
        # ensure it's a sequence so we can modify it
        working_bb = list(get_raster_info(base_raster_path)["bounding_box"])
        # transform the working_bb if target_projection_wkt is not None
        if target_projection_wkt is not None:
            LOGGER.debug("transforming bounding box from %s ", working_bb)
            working_bb = transform_bounding_box(
                base_raster_info["bounding_box"],
                base_raster_info["projection_wkt"],
                target_projection_wkt,
            )
            LOGGER.debug("transforming bounding to %s ", working_bb)
    else:
        # ensure it's a sequence so we can modify it
        working_bb = list(target_bb)

    # determine the raster size that bounds the input bounding box and then
    # adjust the bounding box to be that size
    target_x_size = int(
        abs(float(working_bb[2] - working_bb[0]) / target_pixel_size[0])
    )
    target_y_size = int(
        abs(float(working_bb[3] - working_bb[1]) / target_pixel_size[1])
    )

    # sometimes bounding boxes are numerically perfect, this checks for that
    x_residual = abs(target_x_size * target_pixel_size[0]) - (
        working_bb[2] - working_bb[0]
    )
    if not numpy.isclose(x_residual, 0.0):
        target_x_size += 1
    y_residual = abs(target_y_size * target_pixel_size[1]) - (
        working_bb[3] - working_bb[1]
    )
    if not numpy.isclose(y_residual, 0.0):
        target_y_size += 1

    if target_x_size == 0:
        LOGGER.warning(
            "bounding_box is so small that x dimension rounds to 0; " "clamping to 1."
        )
        target_x_size = 1
    if target_y_size == 0:
        LOGGER.warning(
            "bounding_box is so small that y dimension rounds to 0; " "clamping to 1."
        )
        target_y_size = 1

    # this ensures the bounding boxes perfectly fit a multiple of the target
    # pixel size
    working_bb[2] = working_bb[0] + abs(target_pixel_size[0] * target_x_size)
    working_bb[3] = working_bb[1] + abs(target_pixel_size[1] * target_y_size)
    reproject_callback = _make_logger_callback("Warp %.1f%% complete %s")

    warp_options = []
    if n_threads:
        warp_options.append("NUM_THREADS=%d" % n_threads)
    if gdal_warp_options:
        warp_options.extend(gdal_warp_options)

    mask_vector_path = None
    mask_layer_id = 0
    mask_vector_where_filter = None
    if vector_mask_options:
        # translate geoprocessing terminology into GDAL warp options.
        if "mask_vector_path" not in vector_mask_options:
            raise ValueError(
                "vector_mask_options passed, but no value for "
                '"mask_vector_path": %s',
                vector_mask_options,
            )
        mask_vector_path = vector_mask_options["mask_vector_path"]
        if not os.path.exists(mask_vector_path):
            raise ValueError("The mask vector at %s was not found.", mask_vector_path)
        if "mask_layer_id" in vector_mask_options:
            mask_layer_id = vector_mask_options["mask_layer_id"]
        if "mask_vector_where_filter" in vector_mask_options:
            mask_vector_where_filter = vector_mask_options["mask_vector_where_filter"]
        if "all_touched" in vector_mask_options:
            all_touched = vector_mask_options["all_touched"]
        else:
            all_touched = False

    if vector_mask_options:
        temp_working_dir = tempfile.mkdtemp(dir=working_dir)
        warped_raster_path = os.path.join(
            temp_working_dir,
            os.path.basename(target_raster_path).replace(".tif", "_nonmasked.tif"),
        )
    else:
        # if there is no vector path the result is the warp
        warped_raster_path = target_raster_path
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)

    raster_creation_options = list(raster_driver_creation_tuple[1])
    if base_raster_info["numpy_type"] == numpy.int8 and "PIXELTYPE" not in " ".join(
        raster_creation_options
    ):
        raster_creation_options.append("PIXELTYPE=SIGNEDBYTE")

    # WarpOptions.this is None when an invalid option is passed, and it's a
    # truthy SWIG proxy object when it's given a valid resample arg.
    if not gdal.WarpOptions(resampleAlg=resample_method)[0].this:
        raise ValueError(f'Invalid resample method: "{resample_method}"')

    if band_id is not None:
        _base_raster = gdal.Translate("", base_raster, bandList=[band_id], format="MEM")
    else:
        _base_raster = base_raster

    LOGGER.debug(
        f"about to call warp on {base_raster} with these kwargs {gdal_warp_kwargs}"
    )
    if gdal_warp_kwargs is None:
        gdal_warp_kwargs = dict()

    gdal.Warp(
        warped_raster_path,
        _base_raster,
        format=raster_driver_creation_tuple[0],
        outputBounds=working_bb,
        xRes=abs(target_pixel_size[0]),
        yRes=abs(target_pixel_size[1]),
        resampleAlg=resample_method,
        outputBoundsSRS=target_projection_wkt,
        srcSRS=base_projection_wkt,
        dstSRS=target_projection_wkt,
        multithread=True if warp_options else False,
        warpOptions=warp_options,
        creationOptions=raster_creation_options,
        callback=reproject_callback,
        callback_data=[target_raster_path],
        overviewLevel=-1,
        warpMemoryLimit=128,
        outputType=output_type,
        **gdal_warp_kwargs,
    )
    _base_raster = None
    base_raster = None
    LOGGER.debug(f"warp complete on {warped_raster_path}")

    if vector_mask_options:
        LOGGER.debug(f"starting vector mask of warp on {warped_raster_path}")
        # Make sure the raster creation options passed to ``mask_raster``
        # reflect any metadata updates
        updated_raster_driver_creation_tuple = (
            raster_driver_creation_tuple[0],
            tuple(raster_creation_options),
        )
        # there was a cutline vector, so mask it out now, otherwise target
        # is already the result.

        target_mask_value = vector_mask_options.get("target_mask_value")
        mask_raster(
            (warped_raster_path, 1),
            vector_mask_options["mask_vector_path"],
            target_raster_path,
            mask_layer_id=mask_layer_id,
            where_clause=mask_vector_where_filter,
            target_mask_value=target_mask_value,
            working_dir=temp_working_dir,
            all_touched=all_touched,
            raster_driver_creation_tuple=updated_raster_driver_creation_tuple,
        )
        shutil.rmtree(temp_working_dir, ignore_errors=True)
    LOGGER.debug(f"finished warping {warped_raster_path}")


def rasterize(
    vector_path,
    target_raster_path,
    burn_values=None,
    option_list=None,
    layer_id=0,
    where_clause=None,
):
    """Project a vector onto an existing raster.

    Burn the layer at ``layer_id`` in ``vector_path`` to an existing
    raster at ``target_raster_path_band``.

    Args:
        vector_path (string): filepath to vector to rasterize.
        target_raster_path (string): path to an existing raster to burn vector
            into.  Can have multiple bands.
        burn_values (list/tuple): optional sequence of values to burn into
            each band of the raster.  If used, should have the same length as
            number of bands at the ``target_raster_path`` raster.  If ``None``
            then ``option_list`` must have a valid value.
        option_list (list/tuple): optional a sequence of burn options, if None
            then a valid value for ``burn_values`` must exist. Otherwise, each
            element is a string of the form:

            * ``"ATTRIBUTE=?"``: Identifies an attribute field on the features
              to be used for a burn in value. The value will be burned into all
              output bands. If specified, ``burn_values`` will not be used and
              can be None.
            * ``"CHUNKYSIZE=?"``: The height in lines of the chunk to operate
              on. The larger the chunk size the less times we need to make a
              pass through all the shapes. If it is not set or set to zero the
              default chunk size will be used. Default size will be estimated
              based on the GDAL cache buffer size using formula:
              ``cache_size_bytes/scanline_size_bytes``, so the chunk will not
              exceed the cache.
            * ``"ALL_TOUCHED=TRUE/FALSE"``: May be set to ``TRUE`` to set all
              pixels touched by the line or polygons, not just those whose
              center is within the polygon or that are selected by Brezenhams
              line algorithm. Defaults to ``FALSE``.
            * ``"BURN_VALUE_FROM"``: May be set to "Z" to use the Z values of
              the geometries. The value from burn_values or the
              attribute field value is added to this before burning. In
              default case dfBurnValue is burned as it is (richpsharp:
              note, I'm not sure what this means, but copied from formal
              docs). This is implemented properly only for points and
              lines for now. Polygons will be burned using the Z value
              from the first point.
            * ``"MERGE_ALG=REPLACE/ADD"``: REPLACE results in overwriting of
              value, while ADD adds the new value to the existing
              raster, suitable for heatmaps for instance.

            Example::

                ["ATTRIBUTE=npv", "ALL_TOUCHED=TRUE"]

        layer_id (str/int): name or index of the layer to rasterize. Defaults
            to 0.
        where_clause (str): If not None, is an SQL query-like string to filter
            which features are used to rasterize, (e.x. where="value=1").

    Return:
        None
    """
    LOGGER.debug(f"starting rasterize {target_raster_path}")
    # gdal.PushErrorHandler('CPLQuietErrorHandler')
    raster = gdal.OpenEx(target_raster_path, gdal.GA_Update | gdal.OF_RASTER)
    gdal.PopErrorHandler()
    if raster is None:
        raise ValueError(
            "%s doesn't exist, but needed to rasterize." % target_raster_path
        )

    rasterize_callback = _make_logger_callback("RasterizeLayer %.1f%% complete %s")

    if burn_values is None:
        burn_values = []
    if option_list is None:
        option_list = []

    if not burn_values and not option_list:
        raise ValueError(
            "Neither `burn_values` nor `option_list` is set. At least "
            "one must have a value."
        )

    if not isinstance(burn_values, (list, tuple)):
        raise ValueError(
            "`burn_values` is not a list/tuple, the value passed is '%s'",
            repr(burn_values),
        )

    if not isinstance(option_list, (list, tuple)):
        raise ValueError(
            "`option_list` is not a list/tuple, the value passed is '%s'",
            repr(option_list),
        )

    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer(layer_id)
    if where_clause:
        layer.SetAttributeFilter(where_clause)

    LOGGER.debug(f"about to rasterize {target_raster_path}")
    try:
        result = gdal.RasterizeLayer(
            raster,
            [1],
            layer,
            burn_values=burn_values,
            options=option_list,
            callback=rasterize_callback,
        )
        raster.FlushCache()
    except Exception:
        # something bad happened, but still clean up
        # this case came out of a flaky test condition where the raster
        # would still be in use by the rasterize layer function
        LOGGER.exception("bad error on rasterizelayer")
        result = -1

    layer = None
    vector = None

    if result != 0:
        # need this __swig_destroy__ because we sometimes encounter a flaky
        # test where the path to the raster cannot be cleaned up because
        # it is still in use somewhere, likely a bug in gdal.RasterizeLayer
        # note it is only invoked if there is a serious error
        gdal.Dataset.__swig_destroy__(raster)
        raise RuntimeError("Rasterize returned a nonzero exit code.")

    raster = None


def calculate_disjoint_polygon_set(vector_path, layer_id=0, bounding_box=None):
    """Create a sequence of sets of polygons that don't overlap.

    Determining the minimal number of those sets is an np-complete problem so
    this is an approximation that builds up sets of maximal subsets.

    Args:
        vector_path (string): a path to an OGR vector.
        layer_id (str/int): name or index of underlying layer in
            ``vector_path`` to calculate disjoint set. Defaults to 0.
        bounding_box (sequence): sequence of floats representing a bounding
            box to filter any polygons by. If a feature in ``vector_path``
            does not intersect this bounding box it will not be considered
            in the disjoint calculation. Coordinates are in the order
            [minx, miny, maxx, maxy].

    Return:
        subset_list (sequence): sequence of sets of FIDs from vector_path

    """
    vector = gdal.OpenEx(vector_path, gdal.OF_VECTOR)
    vector_layer = vector.GetLayer(layer_id)
    feature_count = vector_layer.GetFeatureCount()

    if feature_count == 0:
        raise RuntimeError("Vector must have geometries but does not: %s" % vector_path)

    last_time = time.time()
    LOGGER.info("build shapely polygon list")

    if bounding_box is None:
        bounding_box = get_vector_info(vector_path)["bounding_box"]
    bounding_box = shapely.prepared.prep(shapely.geometry.box(*bounding_box))

    # As much as I want this to be in a comprehension, a comprehension version
    # of this loop causes python 3.6 to crash on linux in GDAL 2.1.2 (which is
    # what's in the debian:stretch repos.)
    shapely_polygon_lookup = {}
    for poly_feat in vector_layer:
        poly_geom_ref = poly_feat.GetGeometryRef()
        if poly_geom_ref is None:
            LOGGER.warn(
                f"no geometry in {vector_path} FID: {poly_feat.GetFID()}, "
                "skipping..."
            )
            continue
        # with GDAL>=3.3.0 ExportToWkb returns a bytearray instead of bytes
        shapely_polygon_lookup[poly_feat.GetFID()] = shapely.wkb.loads(
            bytes(poly_geom_ref.ExportToWkb())
        )
        poly_geom_ref = None
    poly_feat = None

    LOGGER.info("build shapely rtree index")
    r_tree_index_stream = [
        (poly_fid, poly.bounds, None)
        for poly_fid, poly in shapely_polygon_lookup.items()
        if bounding_box.intersects(poly)
    ]
    if r_tree_index_stream:
        poly_rtree_index = rtree.index.Index(r_tree_index_stream)
    else:
        LOGGER.warning("no polygons intersected the bounding box")
        return []

    vector_layer = None
    vector = None
    LOGGER.info(
        "poly feature lookup 100.0%% complete on %s",
        os.path.basename(vector_path),
    )

    LOGGER.info("build poly intersection lookup")
    poly_intersect_lookup = collections.defaultdict(set)
    for poly_index, (poly_fid, poly_geom) in enumerate(shapely_polygon_lookup.items()):
        last_time = _invoke_timed_callback(
            last_time,
            lambda: LOGGER.info(
                "poly intersection lookup approximately %.1f%% complete " "on %s",
                100.0 * float(poly_index + 1) / len(shapely_polygon_lookup),
                os.path.basename(vector_path),
            ),
            _LOGGING_PERIOD,
        )
        possible_intersection_set = list(
            poly_rtree_index.intersection(poly_geom.bounds)
        )
        # no reason to prep the polygon to intersect itself
        if len(possible_intersection_set) > 1:
            polygon = shapely.prepared.prep(poly_geom)
        else:
            polygon = poly_geom
        for intersect_poly_fid in possible_intersection_set:
            if intersect_poly_fid == poly_fid or polygon.intersects(
                shapely_polygon_lookup[intersect_poly_fid]
            ):
                poly_intersect_lookup[poly_fid].add(intersect_poly_fid)
        polygon = None
    LOGGER.info(
        "poly intersection feature lookup 100.0%% complete on %s",
        os.path.basename(vector_path),
    )

    # Build maximal subsets
    subset_list = []
    while len(poly_intersect_lookup) > 0:
        # sort polygons by increasing number of intersections
        intersections_list = [
            (len(poly_intersect_set), poly_fid, poly_intersect_set)
            for poly_fid, poly_intersect_set in poly_intersect_lookup.items()
        ]
        intersections_list.sort()

        # build maximal subset
        maximal_set = set()
        for _, poly_fid, poly_intersect_set in intersections_list:
            last_time = _invoke_timed_callback(
                last_time,
                lambda: LOGGER.info(
                    "maximal subset build approximately %.1f%% complete " "on %s",
                    100.0
                    * float(feature_count - len(poly_intersect_lookup))
                    / feature_count,
                    os.path.basename(vector_path),
                ),
                _LOGGING_PERIOD,
            )
            if not poly_intersect_set.intersection(maximal_set):
                # no intersection, add poly_fid to the maximal set and remove
                # the polygon from the lookup
                maximal_set.add(poly_fid)
                del poly_intersect_lookup[poly_fid]
        # remove all the polygons from intersections once they're computed
        for poly_fid, poly_intersect_set in poly_intersect_lookup.items():
            poly_intersect_lookup[poly_fid] = poly_intersect_set.difference(maximal_set)
        subset_list.append(maximal_set)
    LOGGER.info(
        "maximal subset build 100.0%% complete on %s",
        os.path.basename(vector_path),
    )
    return subset_list


def distance_transform_edt(
    base_region_raster_path_band,
    target_distance_raster_path,
    sampling_distance=(1.0, 1.0),
    working_dir=None,
    clean_working_dir=True,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
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

    Args:
        base_region_raster_path_band (tuple): a tuple including file path to a
            raster and the band index to define the base region pixels. Any
            pixel  that is not 0 and nodata are considered to be part of the
            region.
        target_distance_raster_path (string): path to the target raster that
            is the exact euclidean distance transform from any pixel in the
            base raster that is not nodata and not 0. The units are in
            ``(pixel distance * sampling_distance)``.
        sampling_distance (tuple/list): an optional parameter used to scale
            the pixel distances when calculating the distance transform.
            Defaults to (1.0, 1.0). First element indicates the distance
            traveled in the x direction when changing a column index, and the
            second element in y when changing a row index. Both values must
            be > 0.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        clean_working_dir (bool): If True, delete working directory when
            complete.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    working_raster_paths = {}
    for raster_prefix in ["region_mask_raster", "g_raster"]:
        with tempfile.NamedTemporaryFile(
            prefix=raster_prefix, suffix=".tif", delete=False, dir=working_dir
        ) as tmp_file:
            working_raster_paths[raster_prefix] = tmp_file.name
    nodata = (get_raster_info(base_region_raster_path_band[0])["nodata"])[
        base_region_raster_path_band[1] - 1
    ]
    nodata_out = 255

    def mask_op(base_array):
        """Convert base_array to 1 if not 0 and nodata, 0 otherwise."""
        if nodata is not None:
            return ~numpy.isclose(base_array, nodata) & (base_array != 0)
        else:
            return base_array != 0

    if not isinstance(sampling_distance, (tuple, list)):
        raise ValueError(
            "`sampling_distance` should be a tuple/list, instead it's %s"
            % (type(sampling_distance))
        )

    sample_d_x, sample_d_y = sampling_distance
    if sample_d_x <= 0.0 or sample_d_y <= 0.0:
        raise ValueError(
            "Sample distances must be > 0.0, instead got %s", sampling_distance
        )

    raster_calculator(
        [base_region_raster_path_band],
        mask_op,
        working_raster_paths["region_mask_raster"],
        gdal.GDT_Byte,
        nodata_out,
        calc_raster_stats=False,
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )
    geoprocessing_core._distance_transform_edt(
        working_raster_paths["region_mask_raster"],
        working_raster_paths["g_raster"],
        sampling_distance[0],
        sampling_distance[1],
        target_distance_raster_path,
        raster_driver_creation_tuple,
    )

    if clean_working_dir:
        for path in working_raster_paths.values():
            try:
                os.remove(path)
            except OSError:
                LOGGER.warning("couldn't remove file %s", path)


class PolyEqWrapper:
    def __init__(self, poly):
        self.poly = poly

    def __str__(self):
        hash_str = str(numpy.array(self.poly))
        # print(f'HASH STR: {hash_str}')
        return hash_str
        # hash_str = str(numpy.array(self.poly.exterior.coords))
        # for interior in self.poly.interiors:
        #     hash_str += str(numpy.array(interior.coords))
        # return hash_str

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.poly.equals(other.poly)


def _calculate_convolve_cache_index(predict_bounds_list):
    """Creates a spatial index of unique intersecting boxes for c2d writes

    Args:
        predict_bounds_list (list): list of GDAL offset index dictionaries
            containing, 'xoff', 'yoff', 'win_xsize', 'win_ysize' of expected
            result bounds.

    Returns:
        rtree structure referencing bounds for unique write blocks
        list of shapely boxes indexed by id in rtree structure
        dictionary indexed by box bounds indicating how many expected
            writes to the given bounds box.
    """
    # create spatial index of expected write regions
    LOGGER.debug("build initial r tree")
    y_val_set = set()
    x_val_set = set()
    index_box_list = []
    for r_tree_index, index_dict in enumerate(
        sorted(predict_bounds_list, key=lambda v: (v["yoff"], v["xoff"]))
    ):
        left = index_dict["xoff"]
        bottom = index_dict["yoff"]
        right = index_dict["xoff"] + index_dict["win_xsize"]
        top = index_dict["yoff"] + index_dict["win_ysize"]
        x_val_set = x_val_set.union(set([left, right]))
        y_val_set = y_val_set.union(set([top, bottom]))
        index_box = shapely.geometry.box(left, bottom, right, top)
        index_box_list.append(index_box)

    box_tree = STRtree(index_box_list)

    sorted_x = list(sorted(x_val_set))
    sorted_y = list(sorted(y_val_set))

    finished_box_list = []
    finished_box_count = dict()

    LOGGER.debug("assemble cache boxes")
    finished_box_list = [
        shapely.geometry.box(left, bottom, right, top)
        for bottom, top in zip(sorted_y[:-1], sorted_y[1:])
        for left, right in zip(sorted_x[:-1], sorted_x[1:])
    ]
    LOGGER.debug("count intersecting boxes")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        finished_box_count = dict(
            executor.map(
                lambda cache_box: (
                    cache_box.bounds,
                    sum(
                        [
                            1
                            for v in box_tree.query(cache_box)
                            if not index_box_list[v].touches(cache_box)
                        ]
                    ),
                ),
                finished_box_list,
            )
        )

    # build final r-tree for lookup
    LOGGER.debug(f"build STRtree for cache lookup with {len(finished_box_count)} boxes")

    box_tree = STRtree(finished_box_list)
    box_tree_lookup = {id(box): index for index, box in enumerate(finished_box_list)}
    return (box_tree, box_tree_lookup, finished_box_list, finished_box_count)


def convolve_2d(
    signal_path_band,
    kernel_path_band,
    target_path,
    ignore_nodata_and_edges=False,
    mask_nodata=True,
    normalize_kernel=False,
    target_datatype=gdal.GDT_Float64,
    target_nodata=None,
    working_dir=None,
    set_tol_to_zero=1e-8,
    max_timeout=_MAX_TIMEOUT,
    largest_block=2**24,
    n_workers=multiprocessing.cpu_count(),
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Convolve 2D kernel over 2D signal.

    Convolves the raster in ``kernel_path_band`` over ``signal_path_band``.
    Nodata values are treated as 0.0 during the convolution and masked to
    nodata for the output result where ``signal_path`` has nodata.

    Note with default values, boundary effects can be seen in the result where
    the kernel would hang off the edge of the raster or in regions with
    nodata pixels. The function would treat these areas as values with "0.0"
    by default thus pulling the total convolution down in these areas. This
    is similar to setting ``mode='same'`` in Numpy's ``convolve`` function:
    https://numpy.org/doc/stable/reference/generated/numpy.convolve.html

    This boundary effect can be avoided by setting
    ``ignore_nodata_and_edges=True`` which normalizes the target result by
    dynamically accounting for the number of valid signal pixels the kernel
    overlapped during the convolution step.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index), all pixel values should
            be valid -- output is not well defined if the kernel raster has
            nodata values.
        target_path (string): filepath to target raster that's the convolution
            of signal with kernel.  Output will be a single band raster of
            same size and projection as ``signal_path_band``. Any nodata pixels
            that align with ``signal_path_band`` will be set to nodata.
        ignore_nodata_and_edges (boolean): If true, any pixels that are equal
            to ``signal_path_band``'s nodata value or signal pixels where the
            kernel extends beyond the edge of the raster are not included when
            averaging the convolution filter. This has the effect of
            "spreading" the result as though nodata and edges beyond the
            bounds of the raster are 0s. If set to false this tends to "pull"
            the signal away from nodata holes or raster edges. Set this value
            to ``True`` to avoid distortions signal values near edges for
            large integrating kernels.
                It can be useful to set this value to ``True`` to fill
            nodata holes through distance weighted averaging. In this case
            ``mask_nodata`` must be set to ``False`` so the result does not
            mask out these areas which are filled in. When using this
            technique be careful of cases where the kernel does not extend
            over any areas except nodata holes, in this case the resulting
            values in these areas will be nonsensical numbers, perhaps
            numerical infinity or NaNs.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        mask_nodata (boolean): If true, ``target_path`` raster's output is
            nodata where ``signal_path_band``'s pixels were nodata. Note that
            setting ``ignore_nodata_and_edges`` to ``True`` while setting
            ``mask_nodata`` to ``False`` can allow for a technique involving
            distance weighted averaging to define areas that would otherwise
            be nodata. Be careful in cases where the kernel does not
            extend over any valid non-nodata area since the result can be
            numerical infinity or NaNs.
        target_datatype (GDAL type): a GDAL raster type to set the output
            raster type to, as well as the type to calculate the convolution
            in.  Defaults to GDT_Float64.  Note signed byte is not
            supported.
        target_nodata (int/float): nodata value to set on output raster.
            If ``target_datatype`` is not gdal.GDT_Float64, this value must
            be set.  Otherwise defaults to the minimum value of a float32.
        raster_creation_options (sequence): an argument list that will be
            passed to the GTiff driver for creating ``target_path``.  Useful
            for blocksizes, compression, and more.
        working_dir (string): If not None, indicates where temporary files
            should be created during this run.
        set_tol_to_zero (float): any value within +- this from 0.0 will get
            set to 0.0. This is to handle numerical roundoff errors that
            sometimes result in "numerical zero", such as -1.782e-18 that
            cannot be tolerated by users of this function. If `None` no
            adjustment will be done to output values.
        max_timeout (float): maximum amount of time to wait for worker thread
            to terminate.
        largest_block (int): largest blocksize to attempt to read when
            processing signal and kernel images. Defaults to 2**24 that
            was experimentally determined as an optimal size on random
            convolution blocks.
        n_workers (int): number of parallel workers to use when calculating
            convolution. Reduce to help to reduce memory footprint.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        ``None``

    Raises:
        ValueError:
            if ``ignore_nodata_and_edges`` is ``True`` and ``mask_nodata``
            is ``False``.
        ValueError:
            if ``signal_path_band`` or ``kernel_path_band`` is a row based
            blocksize which would result in slow runtimes due to gdal
            cache thrashing.

    """
    _wait_timeout = 5.0
    if target_datatype is not gdal.GDT_Float64 and target_nodata is None:
        raise ValueError(
            "`target_datatype` is set, but `target_nodata` is None. "
            "`target_nodata` must be set if `target_datatype` is not "
            "`gdal.GDT_Float64`.  `target_nodata` is set to None."
        )
    if target_nodata is None:
        target_nodata = float(numpy.finfo(numpy.float32).min)

    if ignore_nodata_and_edges and not mask_nodata:
        LOGGER.debug(
            "ignore_nodata_and_edges is True while mask_nodata is False -- "
            "this can yield a nonsensical result in areas where the kernel "
            "touches only nodata values."
        )

    bad_raster_path_list = []
    for raster_id, raster_path_band in [
        ("signal", signal_path_band),
        ("kernel", kernel_path_band),
    ]:
        if not _is_raster_path_band_formatted(raster_path_band):
            bad_raster_path_list.append((raster_id, raster_path_band))
    if bad_raster_path_list:
        raise ValueError(
            "Expected raster path band sequences for the following arguments "
            f"but instead got: {bad_raster_path_list}"
        )

    signal_raster_info = get_raster_info(signal_path_band[0])
    kernel_raster_info = get_raster_info(kernel_path_band[0])

    for info_dict in [signal_raster_info, kernel_raster_info]:
        if 1 in info_dict["block_size"]:
            raise ValueError(
                f"{signal_path_band} has a row blocksize which can make this "
                f"function run very slow, create a square blocksize using "
                f"`warp_raster` or `align_and_resize_raster_stack` which "
                f"creates square blocksizes by default: {info_dict}"
            )

    new_raster_from_base(
        signal_path_band[0],
        target_path,
        target_datatype,
        [target_nodata],
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )

    n_cols_signal, n_rows_signal = signal_raster_info["raster_size"]
    n_cols_kernel, n_rows_kernel = kernel_raster_info["raster_size"]
    s_path_band = signal_path_band
    k_path_band = kernel_path_band
    s_nodata = signal_raster_info["nodata"][0]

    cache_row_lookup = collections.defaultdict(lambda: None)
    config = dict()
    config["cache_block_writes"] = 0
    cache_worker_queue_map = dict()

    def _cache_row_worker(memmap_dir, cache_row_tuple, read_queue, write_queue):
        try:
            # initalize cache block
            LOGGER.debug(f"initalize cache block {cache_row_tuple}")
            # we need the original signal raster info because we want the
            # output to be clipped and NODATA masked to it
            signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
            signal_band = signal_raster.GetRasterBand(signal_path_band[1])
            # getting the offset list before it's opened for updating

            cache_filename = os.path.join(
                memmap_dir, f"cache_array_{cache_row_tuple}.npy"
            )
            cache_array = numpy.memmap(
                cache_filename,
                dtype=numpy.float64,
                mode="w+",
                shape=(cache_row_tuple[1] - cache_row_tuple[0], n_cols_signal),
            )

            non_nodata_filename = os.path.join(
                memmap_dir, f"non_nodata_array_{cache_row_tuple}.npy"
            )
            non_nodata_array = numpy.memmap(
                non_nodata_filename,
                dtype=bool,
                mode="w+",
                shape=(cache_row_tuple[1] - cache_row_tuple[0], n_cols_signal),
            )

            valid_mask_filename = None
            mask_array = None
            if ignore_nodata_and_edges:
                valid_mask_filename = os.path.join(
                    memmap_dir, f"mask_array_{cache_row_tuple}.npy"
                )
                mask_array = numpy.memmap(
                    valid_mask_filename,
                    dtype=numpy.float64,
                    mode="w+",
                    shape=(
                        cache_row_tuple[1] - cache_row_tuple[0],
                        n_cols_signal,
                    ),
                )
                mask_array[:] = 0

            cache_array[:] = 0.0

            # initalized non-nodata mask
            all_nodata = False
            if s_nodata is not None and mask_nodata:
                cache_win_ysize = cache_row_tuple[1] - cache_row_tuple[0]
                potential_nodata_signal_array = signal_band.ReadAsArray(
                    xoff=0,
                    yoff=cache_row_tuple[0],
                    win_xsize=n_cols_signal,
                    win_ysize=cache_win_ysize,
                )
                #  from: absolute(a - b) <= (atol + rtol * absolute(b))
                numexpr.evaluate(
                    "abs(a - b) > " "(atol + rtol * abs(b))",
                    out=non_nodata_array,
                    local_dict={
                        "rtol": 1e-05,
                        "atol": 1e-08,
                        "a": potential_nodata_signal_array,
                        "b": s_nodata,
                    },
                )
                del potential_nodata_signal_array

                if not numpy.any(non_nodata_array):
                    # we can ignore all incoming results
                    all_nodata = True
                else:
                    numexpr.evaluate(
                        "where(non_nodata_array, cache_array, target_nodata)",
                        out=cache_array,
                        local_dict={
                            "non_nodata_array": non_nodata_array,
                            "cache_array": cache_array,
                            "target_nodata": target_nodata,
                        },
                    )
            else:
                non_nodata_array[:] = 1

            if not all_nodata:
                cache_row_lookup[cache_row_tuple] = (
                    cache_filename,
                    cache_array,
                    non_nodata_filename,
                    non_nodata_array,
                    valid_mask_filename,
                    mask_array,
                )
            else:
                cache_array._mmap.close()
                non_nodata_array._mmap.close()
                del cache_array
                del non_nodata_array
                if mask_array is not None:
                    mask_array._mmap.close()
                    del mask_array
                gc.collect()
                for filename in [
                    cache_filename,
                    non_nodata_filename,
                    valid_mask_filename,
                ]:
                    if filename is not None:
                        os.remove(filename)

            signal_raster = None
            signal_band = None
            while True:
                # first val is the priority which can be ignored
                attempts = 0
                while True:
                    try:
                        cache_box, local_result, local_mask_result = read_queue.get(
                            timeout=_wait_timeout
                        )
                        cache_row_write_count[cache_row_tuple] -= 1
                        break
                    except queue.Empty:
                        attempts += 1
                        LOGGER.debug(
                            f"_cache_row_worker {cache_row_tuple}: waiting "
                            f"for payload for "
                            f"{attempts*_wait_timeout:.1f}s"
                        )

                if not all_nodata:
                    cache_xmin, cache_ymin, cache_xmax, cache_ymax = cache_box
                    local_slice = (
                        slice(
                            cache_ymin - cache_row_tuple[0],
                            cache_ymax - cache_row_tuple[0],
                        ),
                        slice(cache_xmin, cache_xmax),
                    )

                    # load local slices
                    non_nodata_mask = non_nodata_array[local_slice]
                    if local_result is not None:
                        cache_array[local_slice][non_nodata_mask] += local_result[
                            non_nodata_mask
                        ]
                    if local_mask_result is not None:
                        mask_array[local_slice][non_nodata_mask] += local_mask_result[
                            non_nodata_mask
                        ]
                    del non_nodata_mask
                del local_result
                del local_mask_result

                if cache_row_write_count[cache_row_tuple] == 0:
                    if not all_nodata:
                        cache_array = None
                        del non_nodata_array
                        del mask_array
                    del cache_worker_queue_map[cache_row_tuple]

                    attempts = 0
                    while True:
                        try:
                            write_queue.put(cache_row_tuple, timeout=_wait_timeout)
                            break
                        except queue.Full:
                            attempts += 1
                            LOGGER.debug(
                                f"_cache_row_worker {cache_row_tuple} "
                                f"write_queue has been full for "
                                f"{attempts*_wait_timeout:.1f}s"
                            )
                    config["cache_block_writes"] += 1
                    return
        except Exception:
            LOGGER.exception(f"exception on cache row worker for  {cache_row_tuple}")
            raise

    writer_free = threading.Event()

    def _target_raster_worker_op(expected_writes, target_write_queue):
        """To parallelize writes."""
        target_raster = gdal.OpenEx(target_path, gdal.OF_RASTER | gdal.GA_Update)
        target_band = target_raster.GetRasterBand(1)
        last_target_raster_worker_log_time = time.time()
        row_written_array = numpy.zeros(n_rows_signal, dtype=bool)
        try:
            LOGGER.debug(f"(3) starting _target_raster_worker_op, {target_path}")
            # make numpy rows that can be memory block written
            aligned_row_map = dict()
            memory_row_size = 256
            while True:
                attempts = 0
                last_target_raster_worker_log_time = _invoke_timed_callback(
                    last_target_raster_worker_log_time,
                    lambda: LOGGER.info(
                        f"""(3) _target_raster_worker_op: convolution worker approximately {
                            100.0 * config['cache_block_writes'] / expected_writes:.1f}% """
                        f"""complete on {os.path.basename(target_path)} """
                        f"""{-99999 if config['cache_block_writes'] == 0 else (
                            expected_writes-config['cache_block_writes'])*(
                            (time.time()-start_time)/config['cache_block_writes']):.1f}s """
                        f"""remaining"""
                    ),
                    _LOGGING_PERIOD,
                )

                while True:
                    try:
                        writer_free.set()
                        payload = target_write_queue.get(timeout=_wait_timeout)
                        LOGGER.debug("(3) _target_raster_worker_op got payload")
                        break
                    except queue.Empty:
                        attempts += 1
                        LOGGER.debug(
                            f"(3) _target_raster_worker_op: waiting for payload for "
                            f"{attempts*_wait_timeout:.1f}s"
                        )

                if len(payload) == 1:  # sentinel for end
                    target_band = None
                    target_raster = None
                    target_write_queue.put(payload)
                    if config["cache_block_writes"] != expected_writes:
                        LOGGER.warn(
                            f"this is probably fine because nodata blocks "
                            f"were skipped but, expected block writes to "
                            f"be {expected_writes} but it is "
                            f"{config['cache_block_writes']}"
                        )
                    break
                # the payload is just a row_start/row_end tuple that can be used to index into
                # ``cache_row_lookup``
                cache_row_tuple = payload

                if cache_row_lookup[cache_row_tuple] is not None:
                    # load the arrays and the memory mapped filenames
                    # for those arrays
                    (
                        cache_filename,
                        cache_array,
                        non_nodata_filename,
                        non_nodata_array,
                        valid_mask_filename,
                        mask_array,
                    ) = cache_row_lookup[cache_row_tuple]
                else:
                    # was all nodata so can skip
                    del cache_row_lookup[cache_row_tuple]
                    continue
                del cache_row_lookup[cache_row_tuple]

                if ignore_nodata_and_edges:
                    non_nodata_array &= mask_array > 0
                    cache_array[non_nodata_array] /= mask_array[
                        non_nodata_array
                    ].astype(numpy.float64)

                    # scale by kernel sum if necessary since mask division will
                    # automatically normalize kernel
                    if not normalize_kernel:
                        cache_array[non_nodata_array] *= kernel_sum

                lower_bound = (cache_row_tuple[0] // memory_row_size) * memory_row_size
                upper_bound = int(
                    numpy.ceil(cache_row_tuple[1] / memory_row_size) * memory_row_size
                )
                if upper_bound == lower_bound:
                    upper_bound = lower_bound + memory_row_size
                for global_y_lower_bound in numpy.arange(
                    lower_bound, upper_bound, memory_row_size
                ):
                    # global y start indicates what memory aligned row on the
                    # raster this slice should start at.

                    # local row width is used to determine how wide the local
                    # cache row is, either memory row or less if on bound
                    global_row_width = memory_row_size
                    if global_row_width + global_y_lower_bound >= n_rows_signal:
                        global_row_width = n_rows_signal - global_y_lower_bound

                    # target y start is what row on the local cache_array
                    # this should start at
                    local_block_y_start = cache_row_tuple[0] - global_y_lower_bound
                    if local_block_y_start < 0:
                        # started on a previous block, we start right at 0 now
                        local_block_y_start = 0

                    local_block_y_end = cache_row_tuple[1] - global_y_lower_bound
                    if local_block_y_end > global_row_width:
                        local_block_y_end = global_row_width

                    block_width = local_block_y_end - local_block_y_start

                    # calculate the y bounds in the cache array itself
                    global_y_lower_start = global_y_lower_bound + local_block_y_start
                    cache_y_start = global_y_lower_start - cache_row_tuple[0]
                    cache_y_end = cache_y_start + block_width

                    if global_y_lower_bound not in aligned_row_map:
                        local_block_filename = os.path.join(
                            memmap_dir,
                            f"local_block_{global_y_lower_bound}.npy",
                        )
                        local_block = numpy.memmap(
                            local_block_filename,
                            dtype=cache_array.dtype,
                            mode="w+",
                            shape=(global_row_width, n_cols_signal),
                        )
                        aligned_row_map[global_y_lower_bound] = (
                            local_block,
                            local_block_filename,
                        )

                    local_block = aligned_row_map[global_y_lower_bound][0]
                    local_block[local_block_y_start:local_block_y_end] = cache_array[
                        cache_y_start:cache_y_end, :
                    ]
                    row_written_array[
                        global_y_lower_start : global_y_lower_start + block_width
                    ] = 1
                    ready_to_write = numpy.all(
                        row_written_array[
                            global_y_lower_bound : global_y_lower_bound
                            + global_row_width
                        ]
                    )

                    if ready_to_write:
                        writer_free.clear()
                        target_band.WriteArray(
                            local_block, xoff=0, yoff=int(global_y_lower_bound)
                        )
                        local_block._mmap.close()
                        del local_block
                        gc.collect()
                        os.remove(aligned_row_map[global_y_lower_bound][1])
                        del aligned_row_map[global_y_lower_bound]
                        writer_free.set()

                cache_array._mmap.close()
                non_nodata_array._mmap.close()
                del cache_array
                del non_nodata_array
                if mask_array is not None:
                    mask_array._mmap.close()
                    del mask_array
                gc.collect()
                for filename in [
                    cache_filename,
                    non_nodata_filename,
                    valid_mask_filename,
                ]:
                    if filename is not None:
                        os.remove(filename)
            LOGGER.info("target raster worker quitting")
            if len(aligned_row_map) != 0:
                raise RuntimeError(
                    f"expected aligned_row_map to be empty but still has the following rows: {list(aligned_row_map.keys())}"
                )
        except Exception:
            LOGGER.exception("exception happened on (3)")
            raise

    LOGGER.info("starting convolve")

    # calculate the kernel sum for normalization
    kernel_nodata = kernel_raster_info["nodata"][0]
    kernel_sum = 0.0
    for _, kernel_block in iterblocks(kernel_path_band):
        if kernel_nodata is not None and ignore_nodata_and_edges:
            kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
        kernel_sum += numpy.sum(kernel_block)
    del kernel_block

    # limit the size of the work queue since a large kernel / signal with small
    # block size can have a large memory impact when queuing offset lists.
    work_queue = queue.PriorityQueue()

    signal_offset_list = sorted(
        iterblocks(s_path_band, offset_only=True, largest_block=largest_block),
        key=lambda d: (d["yoff"], d["xoff"]),
    )
    kernel_offset_list = sorted(
        iterblocks(k_path_band, offset_only=True, largest_block=largest_block),
        key=lambda d: (d["yoff"], d["xoff"]),
    )
    n_blocks = len(signal_offset_list) * len(kernel_offset_list)

    LOGGER.debug("start fill work queue thread")

    def _predict_bounds(signal_offset, kernel_offset):
        # Add result to current output to account for overlapping edges
        left_index_raster = (
            signal_offset["xoff"] - n_cols_kernel // 2 + kernel_offset["xoff"]
        )
        right_index_raster = (
            signal_offset["xoff"]
            - n_cols_kernel // 2
            + kernel_offset["xoff"]
            + signal_offset["win_xsize"]
            + kernel_offset["win_xsize"]
            - 1
        )
        top_index_raster = (
            signal_offset["yoff"] - n_rows_kernel // 2 + kernel_offset["yoff"]
        )
        bottom_index_raster = (
            signal_offset["yoff"]
            - n_rows_kernel // 2
            + kernel_offset["yoff"]
            + signal_offset["win_ysize"]
            + kernel_offset["win_ysize"]
            - 1
        )

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
            "xoff": left_index_raster,
            "yoff": top_index_raster,
            "win_xsize": right_index_raster - left_index_raster,
            "win_ysize": bottom_index_raster - top_index_raster,
        }
        if index_dict["win_xsize"] <= 0 or index_dict["win_ysize"] <= 0:
            # this can happen if the kernel just shifts outside of signal
            return None
        return index_dict

    LOGGER.debug("fill work queue")
    predict_bounds_list = []
    for signal_offset in signal_offset_list:
        for kernel_offset in kernel_offset_list:
            output_bounds = _predict_bounds(signal_offset, kernel_offset)
            if output_bounds is not None:
                work_queue.put(
                    PrioritizedItem(
                        output_bounds["yoff"], (signal_offset, kernel_offset)
                    )
                )
                predict_bounds_list.append(output_bounds)
    # sort by increasing y value
    predict_bounds_list = sorted(
        predict_bounds_list, key=lambda d: (d["yoff"], d["xoff"])
    )

    work_queue.put(PrioritizedItem(n_rows_signal + 1, None))
    LOGGER.debug("work queue full")
    LOGGER.debug("calculate cache index")
    (
        box_tree,
        box_tree_lookup,
        cache_box_list,
        cache_block_write_dict,
    ) = _calculate_convolve_cache_index(predict_bounds_list)
    LOGGER.debug("cache index calculated")

    # limit the size of the write queue so we don't accidentally load a whole
    # array into memory
    LOGGER.debug("start worker thread")
    write_queue = queue.PriorityQueue(n_workers)
    worker_list = []
    rtree_lock = threading.Lock()
    n_workers = max(1, min(n_workers, len(predict_bounds_list)))
    LOGGER.debug(f"convolve_2d spinnig up {n_workers} workers")
    for worker_id, worker_id in enumerate(range(n_workers)):
        worker = threading.Thread(
            target=_convolve_2d_worker,
            args=(
                worker_id,
                _wait_timeout,
                signal_path_band,
                kernel_path_band,
                ignore_nodata_and_edges,
                normalize_kernel,
                set_tol_to_zero,
                box_tree,
                box_tree_lookup,
                cache_box_list,
                rtree_lock,
                work_queue,
                write_queue,
            ),
        )
        worker.daemon = True
        worker.start()
        worker_list.append(worker)
    active_workers = len(worker_list)

    LOGGER.info(f"{n_blocks} sent to workers, wait for worker results")

    if not working_dir:
        working_dir = "."
    memmap_dir = tempfile.mkdtemp(prefix="convolve_2d", dir=working_dir)

    start_time = time.time()

    y_offset_list = sorted(
        set(
            [v["yoff"] for v in predict_bounds_list]
            + [v["yoff"] + v["win_ysize"] for v in predict_bounds_list]
        )
    )

    # we want to have cache blocks that take up about half the system memory
    # for the expected workers
    n_elements_per_cache = psutil.virtual_memory().total // 8 // n_workers // 8

    start_row = y_offset_list.pop(0)
    cache_row_list = []

    while y_offset_list:
        next_row = y_offset_list.pop(0)
        n_elements = (next_row - start_row) * n_cols_signal
        if n_elements >= n_elements_per_cache:
            cache_row_list.append(start_row)
            start_row = next_row
    # get the last row
    cache_row_list.append(n_rows_signal)
    if len(cache_row_list) == 1:
        cache_row_list.insert(0, 0)

    cache_row_write_count = collections.defaultdict(int)
    with rtree_lock:
        for y_min, y_max in zip(cache_row_list[:-1], cache_row_list[1:]):
            test_box = shapely.geometry.box(0, y_min, n_cols_signal, y_max)
            try:
                for int_box in box_tree.query(
                    shapely.geometry.box(0, y_min, n_cols_signal, y_max)
                ):
                    if not cache_box_list[int_box].touches(test_box):
                        cache_row_write_count[(y_min, y_max)] += cache_block_write_dict[
                            cache_box_list[int_box].bounds
                        ]
            except Exception:
                LOGGER.exception(
                    f"{(0, y_min, n_cols_signal, y_max)}\n{cache_row_list}"
                )
                raise

    target_write_queue = queue.PriorityQueue()
    target_raster_worker = threading.Thread(
        target=_target_raster_worker_op,
        args=(len(cache_row_list) - 1, target_write_queue),
    )
    target_raster_worker.daemon = True
    target_raster_worker.start()

    cache_row_worker_list = []
    while True:
        # the timeout guards against a worst case scenario where the
        # ``_convolve_2d_worker`` has crashed.
        attempts = 0
        while True:
            try:
                write_payload = write_queue.get(timeout=_wait_timeout)
                break
            except queue.Empty:
                attempts += 1
                LOGGER.debug(
                    f"(1) convolve_2d: waiting for worker payload for "
                    f"{attempts*_wait_timeout:.1f}s"
                )

        if write_payload.item is not None:
            (cache_box, _, _) = write_payload.item
        else:
            active_workers -= 1
            if active_workers == 0:
                LOGGER.debug("last worker to end, joining worker list")
                for worker in worker_list:
                    worker.join(max_timeout)
                LOGGER.debug("workers joined")
                break
            continue

        _, cache_ymin, _, _ = cache_box
        row_index = bisect.bisect_right(cache_row_list, cache_ymin)
        cache_row_tuple = (
            cache_row_list[row_index - 1],
            cache_row_list[row_index],
        )

        if cache_row_tuple not in cache_worker_queue_map:
            # start a new worker
            read_queue = queue.Queue()
            cache_worker_queue_map[cache_row_tuple] = read_queue
            cache_row_worker = threading.Thread(
                target=_cache_row_worker,
                args=(
                    memmap_dir,
                    cache_row_tuple,
                    read_queue,
                    target_write_queue,
                ),
            )
            cache_row_worker.daemon = True
            # no reason to process faster than we can write
            writer_free.wait()
            cache_row_worker.start()
            cache_row_worker_list.append(cache_row_worker)
        cache_worker_queue_map[cache_row_tuple].put(write_payload.item)

    LOGGER.debug("wait for cache row workers to join")
    while cache_row_worker_list:
        worker = cache_row_worker_list.pop()
        worker.join()

    target_write_queue.put((n_rows_signal + 1,))
    LOGGER.debug("wait for writer to join")
    target_raster_worker.join()
    LOGGER.info(
        f"convolution worker 100.0% complete on " f"{os.path.basename(target_path)}"
    )

    shutil.rmtree(memmap_dir, ignore_errors=True)


def iterblocks(
    raster_path_band_list,
    largest_block=_LARGEST_ITERBLOCK,
    offset_only=False,
    skip_sparse=False,
    allow_different_blocksize=False,
):
    """Iterate across all the memory blocks in the input raster.

    Result is a generator of block location information and numpy arrays.

    This is especially useful when a single value needs to be derived from the
    pixel values in a raster, such as the sum total of all pixel values, or
    a sequence of unique raster values.  In such cases, ``raster_local_op``
    is overkill, since it writes out a raster.

    As a generator, this can be combined multiple times with itertools.izip()
    to iterate 'simultaneously' over multiple rasters, though the user should
    be careful to do so only with prealigned rasters.

    Args:
        raster_path_band_list (tuple/list): a path/band index tuple to indicate
            which raster band iterblocks should iterate over or a list of
            such tuples.
        largest_block (int): Attempts to iterate over raster blocks with
            this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        offset_only (boolean): defaults to False, if True ``iterblocks`` only
            returns offset dictionary and doesn't read any binary data from
            the raster.  This can be useful when iterating over writing to
            an output.
        skip_sparse (boolean): defaults to False, if True, any iterblocks that
            cover sparse blocks will be not be included in the iteration of
            this result.
        allow_different_blocksize (boolean): allow processing on a different
            set of blocksizes, if True, uses the blocksize of the first
            raster.

    Yields:
        If ``offset_only`` is false, on each iteration, a tuple containing a
        dict of block data and a 2-dimensional numpy array are
        yielded. The dict of block data has these attributes:

        * ``data['xoff']`` - The X offset of the upper-left-hand corner of the
          block.
        * ``data['yoff']`` - The Y offset of the upper-left-hand corner of the
          block.
        * ``data['win_xsize']`` - The width of the block.
        * ``data['win_ysize']`` - The height of the block.

        If ``offset_only`` is True, the function returns only the block offset
        data and does not attempt to read binary data from the raster.

    """
    if not _is_list_of_raster_path_band(raster_path_band_list):
        if not _is_raster_path_band_formatted(raster_path_band_list):
            raise ValueError(
                "`raster_path_band` not formatted as expected.  Expects "
                "(path, band_index), received %s" % repr(raster_path_band_list)
            )
        else:
            raster_path_band_list = [raster_path_band_list]

    band_list = []
    raster_list = []
    blocksize_set = set()
    for raster_path_band in raster_path_band_list:
        raster = gdal.OpenEx(raster_path_band[0], gdal.OF_RASTER)
        raster_list.append(raster)
        if raster is None:
            raise ValueError("Raster at %s could not be opened." % raster_path_band[0])
        band = raster.GetRasterBand(raster_path_band[1])
        band_list.append(band)
        blocksize = tuple(band.GetBlockSize())
        blocksize_set.add(blocksize)
        band = None
    if len(blocksize_set) > 1:
        if not allow_different_blocksize:
            raise ValueError(f"blocksizes should be identical, got {blocksize_set}")
        else:
            LOGGER.warn(
                f"got different blocksizes: {blocksize_set}, " f"using {blocksize}"
            )
    cols_per_block = blocksize[0]
    rows_per_block = blocksize[1]

    n_cols = raster.RasterXSize
    n_rows = raster.RasterYSize
    raster = None

    block_area = cols_per_block * rows_per_block
    # try to make block wider
    if int(largest_block / block_area) > 0:
        width_factor = int(largest_block / block_area)
        cols_per_block *= width_factor
        if cols_per_block > n_cols:
            cols_per_block = n_cols
        block_area = cols_per_block * rows_per_block
    # try to make block taller
    if int(largest_block / block_area) > 0:
        height_factor = int(largest_block / block_area)
        rows_per_block *= height_factor
        if rows_per_block > n_rows:
            rows_per_block = n_rows

    n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block
        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            offset_dict = {
                "xoff": col_offset,
                "yoff": row_offset,
                "win_xsize": col_block_width,
                "win_ysize": row_block_width,
            }

            if not skip_sparse:
                offset_dict_list = [offset_dict]
            elif skip_sparse:
                offset_dict_list = _non_sparse_offsets(band_list, offset_dict)

            for local_offset_dict in offset_dict_list:
                if offset_only:
                    yield local_offset_dict
                else:
                    if len(raster_path_band_list) == 1:
                        yield (
                            local_offset_dict,
                            band_list[0].ReadAsArray(**local_offset_dict),
                        )
                    else:
                        result_list = []
                        for band in band_list:
                            result_list.append(band.ReadAsArray(**local_offset_dict))
                        band = None
                        yield (local_offset_dict, result_list)

    band_list[:] = []
    raster_list[:] = []

    band = None
    raster = None


# def iterblocks(
#         raster_path_band_list, largest_block=_LARGEST_ITERBLOCK,
#         offset_only=False, skip_sparse=False):
#     """Iterate across all the memory blocks in the input raster.

#     Result is a generator of block location information and numpy arrays.

#     This is especially useful when a single value needs to be derived from the
#     pixel values in a raster, such as the sum total of all pixel values, or
#     a sequence of unique raster values.  In such cases, ``raster_local_op``
#     is overkill, since it writes out a raster.

#     As a generator, this can be combined multiple times with itertools.izip()
#     to iterate 'simultaneously' over multiple rasters, though the user should
#     be careful to do so only with prealigned rasters.

#     Args:
#         raster_path_band (tuple): a path/band index tuple to indicate
#             which raster band iterblocks should iterate over.
#         largest_block (int): Attempts to iterate over raster blocks with
#             this many elements.  Useful in cases where the blocksize is
#             relatively small, memory is available, and the function call
#             overhead dominates the iteration.  Defaults to 2**20.  A value of
#             anything less than the original blocksize of the raster will
#             result in blocksizes equal to the original size.
#         offset_only (boolean): defaults to False, if True ``iterblocks`` only
#             returns offset dictionary and doesn't read any binary data from
#             the raster.  This can be useful when iterating over writing to
#             an output.
#         skip_sparse (boolean): defaults to False, if True, any iterblocks that
#             cover sparse blocks will be not be included in the iteration of
#             this result.

#     Yields:
#         If ``offset_only`` is false, on each iteration, a tuple containing a
#         dict of block data and a 2-dimensional numpy array are
#         yielded. The dict of block data has these attributes:

#         * ``data['xoff']`` - The X offset of the upper-left-hand corner of the
#           block.
#         * ``data['yoff']`` - The Y offset of the upper-left-hand corner of the
#           block.
#         * ``data['win_xsize']`` - The width of the block.
#         * ``data['win_ysize']`` - The height of the block.

#         If ``offset_only`` is True, the function returns only the block offset
#         data and does not attempt to read binary data from the raster.

#     """
#     LOGGER.debug(f'starting iterblocks for {raster_path_band_list}')
#     if not _is_list_of_raster_path_band(raster_path_band_list):
#         if not _is_raster_path_band_formatted(raster_path_band_list):
#             raise ValueError(
#                 "`raster_path_band` not formatted as expected.  Expects "
#                 "(path, band_index), received %s" % repr(
#                     raster_path_band_list))
#         else:
#             raster_path_band_list = [raster_path_band_list]

#     LOGGER.debug(f'will process this {raster_path_band_list}')


#     blocksize_set = set()
#     # #for raster_path_band in raster_path_band_list:
#     raster_path_band = raster_path_band_list[0]
#     # #LOGGER.debug(f'{raster_path_band}')
#     raster = gdal.OpenEx(raster_path_band[0], gdal.OF_RASTER)
#     # #raster_list.append(raster)
#     LOGGER.debug(repr(raster))
#     # if raster is None:
#     #     raise ValueError(
#     #         "Raster at %s could not be opened." % raster_path_band[0])
#     # band = raster.GetRasterBand(raster_path_band_list[0][1])
#     #band_list.append(band)
#     # blocksize = tuple(band.GetBlockSize())
#     # LOGGER.debug(blocksize)
#     # blocksize_set.add(blocksize)
#     #band = None
#     #raster = None

#     #raster = gdal.OpenEx(raster_path_band_list[0][0], gdal.OF_RASTER)
#     if raster is None:
#         raise ValueError(
#             "Raster at %s could not be opened." % raster_path_band_list[0][0])
#     band = raster.GetRasterBand(raster_path_band_list[0][1])
#     block = band.GetBlockSize()
#     cols_per_block = block[0]
#     rows_per_block = block[1]

#     n_cols = raster.RasterXSize
#     n_rows = raster.RasterYSize

#     block_area = cols_per_block * rows_per_block
#     # try to make block wider
#     if int(largest_block / block_area) > 0:
#         width_factor = int(largest_block / block_area)
#         cols_per_block *= width_factor
#         if cols_per_block > n_cols:
#             cols_per_block = n_cols
#         block_area = cols_per_block * rows_per_block
#     # try to make block taller
#     if int(largest_block / block_area) > 0:
#         height_factor = int(largest_block / block_area)
#         rows_per_block *= height_factor
#         if rows_per_block > n_rows:
#             rows_per_block = n_rows

#     n_col_blocks = int(math.ceil(n_cols / float(cols_per_block)))
#     n_row_blocks = int(math.ceil(n_rows / float(rows_per_block)))

#     for row_block_index in range(n_row_blocks):
#         row_offset = row_block_index * rows_per_block
#         row_block_width = n_rows - row_offset
#         if row_block_width > rows_per_block:
#             row_block_width = rows_per_block
#         for col_block_index in range(n_col_blocks):
#             col_offset = col_block_index * cols_per_block
#             col_block_width = n_cols - col_offset
#             if col_block_width > cols_per_block:
#                 col_block_width = cols_per_block

#             offset_dict = {
#                 'xoff': col_offset,
#                 'yoff': row_offset,
#                 'win_xsize': col_block_width,
#                 'win_ysize': row_block_width,
#             }

#             if not skip_sparse:
#                 offset_dict_list = [offset_dict]
#             elif skip_sparse:
#                 offset_dict_list = _non_sparse_offsets(band, offset_dict)

#             for local_offset_dict in offset_dict_list:
#                 if offset_only:
#                     yield local_offset_dict
#                 else:
#                     yield (local_offset_dict,
#                            band.ReadAsArray(**local_offset_dict))
#     band = None
#     raster = None


def _non_sparse_offsets(band_list, offset_dict):
    if not isinstance(band_list, list):
        band_list = [band_list]
    blocksize_set = set([tuple(band.GetBlockSize()) for band in band_list])
    if len(blocksize_set) > 1:
        raise ValueError("need exact block sizes for bands")
    blocksize = next(iter(blocksize_set))
    offset_dict_list = []

    coverage_status, percent_cover = functools.reduce(
        lambda tup_a, tup_b: (tup_a[0] | tup_b[0], min(tup_a[1], tup_b[1])),
        [
            band.GetDataCoverageStatus(
                offset_dict["xoff"],
                offset_dict["yoff"],
                offset_dict["win_xsize"],
                offset_dict["win_ysize"],
            )
            for band in band_list
        ],
    )

    if coverage_status & gdal.GDAL_DATA_COVERAGE_STATUS_UNIMPLEMENTED:
        offset_dict_list.append(offset_dict)
        return offset_dict_list

    if not (coverage_status & gdal.GDAL_DATA_COVERAGE_STATUS_DATA):
        # only skip if no data coverage is present at all
        return offset_dict_list

    if percent_cover < 100.0:
        # do it by blocks
        for local_xoff in range(
            offset_dict["xoff"],
            offset_dict["xoff"] + offset_dict["win_xsize"],
            blocksize[0],
        ):
            for local_yoff in range(
                offset_dict["yoff"],
                offset_dict["yoff"] + offset_dict["win_ysize"],
                blocksize[1],
            ):
                local_winx, local_winy = band_list[0].GetActualBlockSize(
                    local_xoff // blocksize[0], local_yoff // blocksize[1]
                )
                local_offset_dict = {
                    "xoff": local_xoff,
                    "yoff": local_yoff,
                    "win_xsize": local_winx,
                    "win_ysize": local_winy,
                }

                coverage_status, _ = functools.reduce(
                    lambda tup_a, tup_b: (
                        tup_a[0] | tup_b[0],
                        min(tup_a[1], tup_b[1]),
                    ),
                    [
                        band.GetDataCoverageStatus(
                            local_offset_dict["xoff"],
                            local_offset_dict["yoff"],
                            local_offset_dict["win_xsize"],
                            local_offset_dict["win_ysize"],
                        )
                        for band in band_list
                    ],
                )

                if coverage_status & gdal.GDAL_DATA_COVERAGE_STATUS_EMPTY:
                    # local coverage is sparse, skip
                    continue
                else:
                    offset_dict_list.append(local_offset_dict)
    else:
        offset_dict_list.append(offset_dict)
    return offset_dict_list


def transform_bounding_box(
    bounding_box,
    base_projection_wkt,
    target_projection_wkt,
    edge_samples=100,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
    check_finite=True,
    allow_partial_reprojection=True,
):
    """Transform input bounding box to output projection.

    This transform accounts for the fact that the reprojected square bounding
    box might be warped in the new coordinate system.  To account for this,
    the function samples points along the original bounding box edges and
    attempts to make the largest bounding box around any transformed point
    on the edge whether corners or warped edges.

    Args:
        bounding_box (sequence): a sequence of 4 coordinates in ``base_epsg``
            coordinate system describing the bound in the order
            [xmin, ymin, xmax, ymax].
        base_projection_wkt (string): the spatial reference of the input
            coordinate system in Well Known Text.
        target_projection_wkt (string): the spatial reference of the desired
            output coordinate system in Well Known Text.
        edge_samples (int): the number of interpolated points along each
            bounding box edge to sample along. A value of 2 will sample just
            the corners while a value of 3 will also sample the corners and
            the midpoint.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This
            parameter should not be changed unless you know what you are
            doing.
        check_finite (bool): If True, raises ValueError if bounding box
            results in non-finite values.
        allow_partial_reprojection (bool): If True, will attempt partial
            reprojections if coordinates lie outside the area defined by
            a projeciton. If False, will raise error in such cases.

    Return:
        A list of the form [xmin, ymin, xmax, ymax] that describes the largest
        fitting bounding box around the original warped bounding box in
        ``new_epsg`` coordinate system.

    Raises:
        ``ValueError`` if resulting transform yields non-finite coordinates.
        This would indicate an ill posed transform region that the user
        should address.

    """
    base_ref = osr.SpatialReference()
    base_ref.ImportFromWkt(base_projection_wkt)

    target_ref = osr.SpatialReference()
    target_ref.ImportFromWkt(target_projection_wkt)

    base_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)
    target_ref.SetAxisMappingStrategy(osr_axis_mapping_strategy)

    transformer = osr.CreateCoordinateTransformation(base_ref, target_ref)

    # Create a bounding box geometry
    ring = ogr.Geometry(ogr.wkbLinearRing)
    # make a linear interpolation around the polygon for extra transform points
    for start, end in [
        ((0, 1), (2, 1)),
        ((2, 1), (2, 3)),
        ((2, 3), (0, 3)),
        ((0, 3), (0, 1)),
    ]:
        for step in range(edge_samples):
            p = step / edge_samples
            x_coord_start = bounding_box[start[0]]
            y_coord_start = bounding_box[start[1]]
            x_coord_end = bounding_box[end[0]]
            y_coord_end = bounding_box[end[1]]

            x_coord = (1 - p) * x_coord_start + p * x_coord_end
            y_coord = (1 - p) * y_coord_start + p * y_coord_end
            ring.AddPoint(x_coord, y_coord)
    # close the ring by putting a point where we start
    ring.AddPoint(bounding_box[0], bounding_box[1])
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    if allow_partial_reprojection:
        gdal.SetConfigOption("OGR_ENABLE_PARTIAL_REPROJECTION", "TRUE")
    else:
        gdal.SetConfigOption("OGR_ENABLE_PARTIAL_REPROJECTION", "FALSE")
    error_code = poly.Transform(transformer)
    if error_code != 0:
        raise ValueError(
            f"error on transforming {bounding_box} from {base_projection_wkt} "
            f"to {target_projection_wkt}. Error code: {error_code}"
        )
    envelope = poly.GetEnvelope()
    # swizzle from xmin xmax ymin ymax to xmin, ymin, xmax, ymax
    transformed_bounding_box = [envelope[i] for i in [0, 2, 1, 3]]

    if check_finite and not all(numpy.isfinite(numpy.array(transformed_bounding_box))):
        raise ValueError(
            f"Could not transform bounding box from base to target projection."
            f"Some transformed coordinates are not finite: "
            f"{transformed_bounding_box}, base bounding box may not fit into "
            f"target coordinate projection system.\n"
            f"Original bounding box: {bounding_box}\n"
            f"Base projection: {base_projection_wkt}\n"
            f"Target projection: {target_projection_wkt}\n"
        )
    return transformed_bounding_box


def mask_raster(
    base_raster_path_band,
    mask_vector_path,
    target_mask_raster_path,
    mask_layer_id=0,
    target_mask_value=None,
    working_dir=None,
    all_touched=False,
    where_clause=None,
    allow_different_blocksize=False,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Mask a raster band with a given vector.

    Args:
        base_raster_path_band (tuple): a (path, band number) tuple indicating
            the data to mask.
        mask_vector_path (path): path to a vector that will be used to mask
            anything outside of the polygon that overlaps with
            ``base_raster_path_band`` to ``target_mask_value`` if defined or
            else ``base_raster_path_band``'s nodata value.
        target_mask_raster_path (str): path to desired target raster that
            is a copy of ``base_raster_path_band`` except any pixels that do
            not intersect with ``mask_vector_path`` are set to
            ``target_mask_value`` or ``base_raster_path_band``'s nodata value
            if ``target_mask_value`` is None.
        mask_layer_id (str/int): an index or name to identify the mask
            geometry layer in ``mask_vector_path``, default is 0.
        target_mask_value (numeric): If not None, this value is written to
            any pixel in ``base_raster_path_band`` that does not intersect
            with ``mask_vector_path``. Otherwise the nodata value of
            ``base_raster_path_band`` is used.
        working_dir (str): this is a path to a directory that can be used to
            hold temporary files required to complete this operation.
        all_touched (bool): if False, a pixel is only masked if its centroid
            intersects with the mask. If True a pixel is masked if any point
            of the pixel intersects the polygon mask.
        allow_different_blocksize (bool): If false, raises an exception if
            rasters are not the same blocksize.
        where_clause (str): (optional) if not None, it is an SQL compatible
            where clause that can be used to filter the features that are used
            to mask the base raster.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to a GTiff driver tuple
            defined at geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    LOGGER.debug(f"about to mask {base_raster_path_band}")
    if working_dir is not None:
        os.makedirs(working_dir, exist_ok=True)
    else:
        working_dir = ""
    mask_raster_dir = tempfile.mkdtemp(dir=working_dir, prefix="mask_raster")
    mask_raster_path = os.path.join(mask_raster_dir, "mask_raster.tif")

    LOGGER.debug(f"about new raster {mask_raster_path}")

    new_raster_from_base(
        base_raster_path_band[0],
        mask_raster_path,
        gdal.GDT_Byte,
        [255],
        fill_value_list=[0],
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )

    base_raster_info = get_raster_info(base_raster_path_band[0])

    LOGGER.debug(f"about to call rasterize with {mask_vector_path}")

    rasterize(
        mask_vector_path,
        mask_raster_path,
        burn_values=[1],
        layer_id=mask_layer_id,
        option_list=[("ALL_TOUCHED=%s" % all_touched).upper()],
        where_clause=where_clause,
    )

    base_nodata = base_raster_info["nodata"][base_raster_path_band[1] - 1]

    if target_mask_value is None:
        mask_value = base_nodata
        if mask_value is None:
            LOGGER.warning(
                "No mask value was passed and target nodata is undefined, "
                "defaulting to 0 as the target mask value."
            )
            mask_value = 0
    else:
        mask_value = target_mask_value

    def mask_op(base_array, mask_array):
        result = numpy.copy(base_array)
        result[mask_array == 0] = mask_value
        return result

    raster_calculator(
        [base_raster_path_band, (mask_raster_path, 1)],
        mask_op,
        target_mask_raster_path,
        base_raster_info["datatype"],
        base_nodata,
        allow_different_blocksize=allow_different_blocksize,
        raster_driver_creation_tuple=raster_driver_creation_tuple,
    )

    shutil.rmtree(mask_raster_dir, ignore_errors=True)


def _invoke_timed_callback(reference_time, callback_lambda, callback_period):
    """Invoke callback if a certain amount of time has passed.

    This is a convenience function to standardize update callbacks from the
    module.

    Args:
        reference_time (float): time to base ``callback_period`` length from.
        callback_lambda (lambda): function to invoke if difference between
            current time and ``reference_time`` has exceeded
            ``callback_period``.
        callback_period (float): time in seconds to pass until
            ``callback_lambda`` is invoked.

    Return:
        ``reference_time`` if ``callback_lambda`` not invoked, otherwise the
        time when ``callback_lambda`` was invoked.

    """
    current_time = time.time()
    if current_time - reference_time > callback_period:
        callback_lambda()
        return current_time
    return reference_time


def _gdal_to_numpy_type(band):
    """Calculate the equivalent numpy datatype from a GDAL raster band type.

    This function doesn't handle complex or unknown types.  If they are
    passed in, this function will raise a ValueError.

    Args:
        band (gdal.Band): GDAL Band

    Return:
        numpy_datatype (numpy.dtype): equivalent of band.DataType

    """
    if band.DataType in _BASE_GDAL_TYPE_TO_NUMPY:
        return _BASE_GDAL_TYPE_TO_NUMPY[band.DataType]

    if band.DataType != gdal.GDT_Byte:
        raise ValueError("Unsupported DataType: %s" % str(band.DataType))

    # band must be GDT_Byte type, check if it is signed/unsigned
    metadata = band.GetMetadata("IMAGE_STRUCTURE")
    if "PIXELTYPE" in metadata and metadata["PIXELTYPE"] == "SIGNEDBYTE":
        return numpy.int8
    return numpy.uint8


def merge_bounding_box_list(bounding_box_list, bounding_box_mode):
    """Create a single bounding box by union or intersection of the list.

    Args:
        bounding_box_list (sequence): a sequence of bounding box coordinates
            in the order [minx, miny, maxx, maxy].
        mode (string): either ``'union'`` or ``'intersection'`` for the
            corresponding reduction mode.

    Return:
        A four tuple bounding box that is the union or intersection of the
        input bounding boxes.

    Raises:
        ValueError
            if the bounding boxes in ``bounding_box_list`` do not
            intersect if the ``bounding_box_mode`` is 'intersection'.

    """

    def _merge_bounding_boxes(bb1, bb2, mode):
        """Merge two bounding boxes through union or intersection.

        Args:
            bb1, bb2 (sequence): sequence of float representing bounding box
                in the form bb=[minx,miny,maxx,maxy]
            mode (string); one of 'union' or 'intersection'

        Return:
            Reduced bounding box of bb1/bb2 depending on mode.

        """

        def _less_than_or_equal(x_val, y_val):
            return x_val if x_val <= y_val else y_val

        def _greater_than(x_val, y_val):
            return x_val if x_val > y_val else y_val

        if mode == "union":
            comparison_ops = [
                _less_than_or_equal,
                _less_than_or_equal,
                _greater_than,
                _greater_than,
            ]
        if mode == "intersection":
            comparison_ops = [
                _greater_than,
                _greater_than,
                _less_than_or_equal,
                _less_than_or_equal,
            ]

        bb_out = [op(x, y) for op, x, y in zip(comparison_ops, bb1, bb2)]
        return bb_out

    result_bb = functools.reduce(
        functools.partial(_merge_bounding_boxes, mode=bounding_box_mode),
        bounding_box_list,
    )
    if result_bb[0] > result_bb[2] or result_bb[1] > result_bb[3]:
        raise ValueError(
            "Bounding boxes do not intersect. Base list: %s mode: %s "
            " result: %s" % (bounding_box_list, bounding_box_mode, result_bb)
        )
    return result_bb


def get_gis_type(path):
    """Calculate the GIS type of the file located at ``path``.

    Args:
        path (str): path to a file on disk.


    Return:
        A bitwise OR of all GIS types that geoprocessing models, currently
        this is ``geoprocessing.UNKNOWN_TYPE``,
        ``geoprocessing.RASTER_TYPE``, or ``geoprocessing.VECTOR_TYPE``.

    """
    if not os.path.exists(path):
        raise ValueError("%s does not exist", path)
    from ecoshard.geoprocessing import UNKNOWN_TYPE

    gis_type = UNKNOWN_TYPE
    try:
        gis_raster = gdal.OpenEx(path, gdal.OF_RASTER)
        if gis_raster is not None:
            from ecoshard.geoprocessing import RASTER_TYPE

            gis_type |= RASTER_TYPE
            gis_raster = None
    except RuntimeError:
        # GDAL can throw an exception if exceptions are on, okay to skip
        # because it means it's not that gis type
        pass
    try:
        gis_vector = gdal.OpenEx(path, gdal.OF_VECTOR)
        if gis_vector is not None:
            from ecoshard.geoprocessing import VECTOR_TYPE

            gis_type |= VECTOR_TYPE
    except RuntimeError:
        # GDAL can throw an exception if exceptions are on, okay to skip
        # because it means it's not that gis type
        pass
    return gis_type


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Return:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """

    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if (current_time - logger_callback.last_time) > 5.0 or (
                df_complete == 1.0 and logger_callback.total_time >= 5.0
            ):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    LOGGER.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    LOGGER.info(message, df_complete * 100, "")
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0
        except Exception:
            LOGGER.exception(
                "Unhandled error occurred while logging "
                "progress.  df_complete: %s, p_progress_arg: %s",
                df_complete,
                p_progress_arg,
            )

    return logger_callback


def _is_list_of_raster_path_band(raster_path_band_list):
    if (
        isinstance(raster_path_band_list, (list, tuple))
        and (len(raster_path_band_list) > 0)
        and (isinstance(raster_path_band_list[0], (list, tuple)))
    ):
        return all(
            [
                _is_raster_path_band_formatted(raster_path_band)
                for raster_path_band in raster_path_band_list
            ]
        )
    return False


def _is_raster_path_band_formatted(raster_path_band):
    """Return true if raster path band is a (str, int) tuple/list."""
    if not isinstance(raster_path_band, (list, tuple)):
        return False
    elif len(raster_path_band) != 2:
        return False
    elif not isinstance(raster_path_band[0], str):
        return False
    elif not isinstance(raster_path_band[1], int):
        return False
    else:
        return True


@retry(
    wait_exponential_multiplier=1000,
    wait_exponential_max=10000,
    stop_max_attempt_number=10,
)
def _convolve_signal_kernel(
    signal_block,
    kernel_block,
    set_tol_to_zero,
    ignore_nodata,
    signal_nodata_mask,
):
    try:
        if numpy.any(signal_block):
            result = scipy.signal.oaconvolve(signal_block, kernel_block)
            # nix any roundoff error
            if set_tol_to_zero is not None:
                # result[numpy.isclose(result, set_tol_to_zero)] = 0.0
                numexpr.evaluate(
                    "where(abs(a - b) < (atol + rtol * abs(b)), 0, a)",
                    out=result,
                    local_dict={
                        "rtol": 1e-05,
                        "atol": 1e-08,
                        "a": result,
                        "b": set_tol_to_zero,
                    },
                )
        else:
            # this lets us skip any all 0 blocks
            result = None

        # if we're ignoring nodata, we need to make a convolution of the
        # nodata mask too
        mask_result = None
        if ignore_nodata:
            if not numpy.all(signal_nodata_mask):
                mask_result = scipy.signal.oaconvolve(~signal_nodata_mask, kernel_block)
        return result, mask_result
    except Exception:
        LOGGER.exception("error on _convolve_signal_kernel")
        raise


def _convolve_2d_worker(
    worker_id,
    _wait_timeout,
    signal_path_band,
    kernel_path_band,
    ignore_nodata,
    normalize_kernel,
    set_tol_to_zero,
    box_tree,
    box_tree_lookup,
    cache_box_list,
    rtree_lock,
    work_queue,
    write_queue,
):
    """Worker function to be used by ``convolve_2d``.

    Args:
        signal_path_band (tuple): a 2 tuple of the form
            (filepath to signal raster, band index).
        kernel_path_band (tuple): a 2 tuple of the form
            (filepath to kernel raster, band index).
        ignore_nodata (boolean): If true, any pixels that are equal to
            ``signal_path_band``'s nodata value are not included when
            averaging the convolution filter.
        normalize_kernel (boolean): If true, the result is divided by the
            sum of the kernel.
        set_tol_to_zero (float): Value to test close to to determine if values
            are zero, and if so, set to zero.
        box_tree (rtree.Index): rtree datastructure to lookup cache
            blocks for cutting results
        box_tree_lookup (dict): maps boxes from ``box_tree`` to indexes
            into other lists
        cache_box_list (list): maps rtree indexes to actual blocks
        rtree_lock (threading.Lock): used to guard the rtree for multi access
        work_queue (Queue): will contain (signal_offset, kernel_offset)
            tuples that can be used to read raster blocks directly using
            GDAL ReadAsArray(**offset). Indicates the block to operate on.
        write_queue (Queue): mechanism to pass result back to the writer
            contains a (index_dict, result, mask_result) tuple that's used
            to write globally at index_dict either result or mask

    Return:
        None
    """
    try:
        LOGGER.debug(f"_convolve_2d_worker ({worker_id}) loading rasters")
        signal_raster = gdal.OpenEx(signal_path_band[0], gdal.OF_RASTER)
        kernel_raster = gdal.OpenEx(kernel_path_band[0], gdal.OF_RASTER)
        signal_band = signal_raster.GetRasterBand(signal_path_band[1])
        kernel_band = kernel_raster.GetRasterBand(kernel_path_band[1])

        signal_raster_info = get_raster_info(signal_path_band[0])
        kernel_raster_info = get_raster_info(kernel_path_band[0])

        n_cols_signal, n_rows_signal = signal_raster_info["raster_size"]
        n_cols_kernel, n_rows_kernel = kernel_raster_info["raster_size"]
        signal_nodata = signal_raster_info["nodata"][0]
        kernel_nodata = kernel_raster_info["nodata"][0]

        mask_result = None  # in case no mask is needed, variable is still defined

        # calculate the kernel sum for normalization
        kernel_sum = 0.0
        for _, kernel_block in iterblocks(kernel_path_band):
            if kernel_nodata is not None and ignore_nodata:
                kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0
            kernel_sum += numpy.sum(kernel_block)

        while True:
            payload = work_queue.get()
            start_time = time.time()
            if payload.item is None:
                work_queue.put(payload)
                break
            signal_offset, kernel_offset = payload.item

            # ensure signal and kernel are internally float64 precision
            # irrespective of their base type
            signal_block = signal_band.ReadAsArray(**signal_offset).astype(
                numpy.float64
            )
            kernel_block = kernel_band.ReadAsArray(**kernel_offset).astype(
                numpy.float64
            )

            # don't ever convolve the nodata value
            if signal_nodata is not None:
                signal_nodata_mask = numpy.isclose(signal_block, signal_nodata)
                signal_block[signal_nodata_mask] = 0.0
                if not ignore_nodata:
                    signal_nodata_mask[:] = 0
            else:
                signal_nodata_mask = numpy.zeros(signal_block.shape, dtype=bool)

            left_index_raster = (
                signal_offset["xoff"] - n_cols_kernel // 2 + kernel_offset["xoff"]
            )
            right_index_raster = (
                signal_offset["xoff"]
                - n_cols_kernel // 2
                + kernel_offset["xoff"]
                + signal_offset["win_xsize"]
                + kernel_offset["win_xsize"]
                - 1
            )
            top_index_raster = (
                signal_offset["yoff"] - n_rows_kernel // 2 + kernel_offset["yoff"]
            )
            bottom_index_raster = (
                signal_offset["yoff"]
                - n_rows_kernel // 2
                + kernel_offset["yoff"]
                + signal_offset["win_ysize"]
                + kernel_offset["win_ysize"]
                - 1
            )

            # it's possible that the piece of the integrating kernel
            # doesn't affect the final result, if so we should skip
            if (
                right_index_raster < 0
                or bottom_index_raster < 0
                or left_index_raster >= n_cols_signal
                or top_index_raster >= n_rows_signal
            ):
                continue

            if kernel_nodata is not None:
                kernel_block[numpy.isclose(kernel_block, kernel_nodata)] = 0.0

            if normalize_kernel:
                kernel_block /= kernel_sum

            # determine the output convolve shape
            shape = (
                numpy.array(signal_block.shape) + numpy.array(kernel_block.shape) - 1
            )

            # result or mask_result will be none if no data were generated
            result, mask_result = _convolve_signal_kernel(
                signal_block,
                kernel_block,
                set_tol_to_zero,
                ignore_nodata,
                signal_nodata_mask,
            )

            del signal_block
            del kernel_block
            del signal_nodata_mask

            left_index_result = 0
            right_index_result = shape[1]
            top_index_result = 0
            bottom_index_result = shape[0]

            # we might abut the edge of the raster, clip if so
            if left_index_raster < 0:
                left_index_result = -left_index_raster
                left_index_raster = 0
            if top_index_raster < 0:
                top_index_result = -top_index_raster
                top_index_raster = 0
            if right_index_raster > n_cols_signal:
                right_index_result -= right_index_raster - n_cols_signal
                right_index_raster = n_cols_signal
            if bottom_index_raster > n_rows_signal:
                bottom_index_result -= bottom_index_raster - n_rows_signal
                bottom_index_raster = n_rows_signal

            # Add result to current output to account for overlapping edges
            index_dict = {
                "xoff": left_index_raster,
                "yoff": top_index_raster,
                "win_xsize": right_index_raster - left_index_raster,
                "win_ysize": bottom_index_raster - top_index_raster,
            }

            if result is not None:
                result = result[
                    top_index_result:bottom_index_result,
                    left_index_result:right_index_result,
                ]

            if mask_result is not None:
                mask_result = mask_result[
                    top_index_result:bottom_index_result,
                    left_index_result:right_index_result,
                ]

            with rtree_lock:
                write_block_list = [
                    box_tree_lookup[id(cache_box_list[box_id])]
                    for box_id in box_tree.query(
                        shapely.geometry.box(
                            index_dict["xoff"],
                            index_dict["yoff"],
                            index_dict["xoff"] + index_dict["win_xsize"],
                            index_dict["yoff"] + index_dict["win_ysize"],
                        )
                    )
                ]

            for write_block_index in write_block_list:
                # write the sublock from `result` indexed by `write_block_index`
                # into the cache_block

                # result is the array to read from
                # index_dict is the global block to write to

                cache_box = cache_box_list[write_block_index].bounds
                cache_xmin, cache_ymin, cache_xmax, cache_ymax = [
                    round(v) for v in cache_box
                ]

                if (
                    (cache_xmin == index_dict["xoff"] + index_dict["win_xsize"])
                    or (cache_ymin == index_dict["yoff"] + index_dict["win_ysize"])
                    or (cache_xmax == index_dict["xoff"])
                    or (cache_ymax == index_dict["yoff"])
                ):
                    # rtree cannot tell intersection vs touch
                    continue

                local_result = None
                if result is not None:
                    local_result = result[
                        cache_ymin
                        - index_dict["yoff"] : cache_ymax
                        - index_dict["yoff"],
                        cache_xmin
                        - index_dict["xoff"] : cache_xmax
                        - index_dict["xoff"],
                    ]

                local_mask_result = None
                if mask_result is not None:
                    local_mask_result = mask_result[
                        cache_ymin
                        - index_dict["yoff"] : cache_ymax
                        - index_dict["yoff"],
                        cache_xmin
                        - index_dict["xoff"] : cache_xmax
                        - index_dict["xoff"],
                    ]

                attempts = 0
                work_time = time.time() - start_time
                while True:
                    try:
                        write_queue.put(
                            PrioritizedItem(
                                cache_ymin,
                                (
                                    (
                                        cache_xmin,
                                        cache_ymin,
                                        cache_xmax,
                                        cache_ymax,
                                    ),
                                    local_result,
                                    local_mask_result,
                                ),
                            ),
                            timeout=_wait_timeout,
                        )
                        break
                    except queue.Full:
                        attempts += 1
                        LOGGER.debug(
                            f"(2) _convolve_2d_worker ({worker_id}): write queue has been full for "
                            f"{attempts*_wait_timeout:.1f}s, did {work_time:.3f}s of work"
                        )

        # Indicates worker has terminated
        LOGGER.debug(f"write worker ({worker_id}) complete")
        write_queue.put(PrioritizedItem(n_rows_signal + 1, None))
    except Exception:
        LOGGER.exception(f"error on _convolve_2d_worker ({worker_id})")
        raise


def _assert_is_valid_pixel_size(target_pixel_size):
    """Return true if ``target_pixel_size`` is a valid 2 element sequence.

    Raises ValueError if not a two element list/tuple and/or the values in
        the sequence are not numerical.

    """

    def _is_number(x):
        """Return true if x is a number."""
        try:
            if isinstance(x, str):
                return False
            float(x)
            return True
        except (ValueError, TypeError):
            return False

    if not isinstance(target_pixel_size, (list, tuple)):
        raise ValueError(
            "target_pixel_size is not a tuple, its value was '%s'",
            repr(target_pixel_size),
        )

    if len(target_pixel_size) != 2 or not all(
        [_is_number(x) for x in target_pixel_size]
    ):
        raise ValueError(
            "Invalid value for `target_pixel_size`, expected two numerical "
            "elements, got: %s",
            repr(target_pixel_size),
        )
    return True


def shapely_geometry_to_vector(
    shapely_geometry_list,
    target_vector_path,
    projection_wkt,
    vector_format,
    fields=None,
    attribute_list=None,
    ogr_geom_type=ogr.wkbPolygon,
):
    """Convert list of geometry to vector on disk.

    Args:
        shapely_geometry_list (list): a list of Shapely objects.
        target_vector_path (str): path to target vector.
        projection_wkt (str): WKT for target vector.
        vector_format (str): GDAL driver name for target vector.
        fields (dict): a python dictionary mapping string fieldname
            to OGR Fieldtypes, if None no fields are added
        attribute_list (list of dicts): a list of python dictionary mapping
            fieldname to field value for each geometry in
            `shapely_geometry_list`, if None, no attributes are created.
        ogr_geom_type (ogr geometry enumerated type): sets the target layer
            geometry type. Defaults to wkbPolygon.

    Return:
        None
    """
    if fields is None:
        fields = {}

    if attribute_list is None:
        attribute_list = [{} for _ in range(len(shapely_geometry_list))]

    num_geoms = len(shapely_geometry_list)
    num_attrs = len(attribute_list)
    if num_geoms != num_attrs:
        raise ValueError(
            f"Geometry count ({num_geoms}) and attribute count "
            f"({num_attrs}) do not match."
        )

    vector_driver = ogr.GetDriverByName(vector_format)
    target_vector = vector_driver.CreateDataSource(target_vector_path)
    layer_name = os.path.basename(os.path.splitext(target_vector_path)[0])
    projection = osr.SpatialReference()
    projection.ImportFromWkt(projection_wkt)
    target_layer = target_vector.CreateLayer(
        layer_name, srs=projection, geom_type=ogr_geom_type
    )

    for field_name, field_type in fields.items():
        target_layer.CreateField(ogr.FieldDefn(field_name, field_type))
    layer_defn = target_layer.GetLayerDefn()

    for shapely_feature, fields in zip(shapely_geometry_list, attribute_list):
        new_feature = ogr.Feature(layer_defn)
        new_geometry = ogr.CreateGeometryFromWkb(shapely_feature.wkb)
        new_feature.SetGeometry(new_geometry)

        for field_name, field_value in fields.items():
            new_feature.SetField(field_name, field_value)
        target_layer.CreateFeature(new_feature)

    target_layer = None
    target_vector = None


def numpy_array_to_raster(
    base_array,
    target_nodata,
    pixel_size,
    origin,
    projection_wkt,
    target_path,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Create a single band raster of size ``base_array.shape``.

    Args:
        base_array (numpy.array): a 2d numpy array.
        target_nodata (numeric): nodata value of target array, can be None.
        pixel_size (tuple): square dimensions (in ``(x, y)``) of pixel.
        origin (tuple/list): x/y coordinate of the raster origin.
        projection_wkt (str): target projection in wkt.
        target_path (str): path to raster to create that will be of the
            same type of base_array with contents of base_array.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None
    """
    numpy_to_gdal_type = {
        numpy.dtype(bool): gdal.GDT_Byte,
        numpy.dtype(numpy.int8): gdal.GDT_Byte,
        numpy.dtype(numpy.uint8): gdal.GDT_Byte,
        numpy.dtype(numpy.int16): gdal.GDT_Int16,
        numpy.dtype(numpy.int32): gdal.GDT_Int32,
        numpy.dtype(numpy.uint16): gdal.GDT_UInt16,
        numpy.dtype(numpy.uint32): gdal.GDT_UInt32,
        numpy.dtype(numpy.float32): gdal.GDT_Float32,
        numpy.dtype(numpy.float64): gdal.GDT_Float64,
        numpy.dtype(numpy.csingle): gdal.GDT_CFloat32,
        numpy.dtype(numpy.complex64): gdal.GDT_CFloat64,
    }

    if hasattr(gdal, "GDT_Int64"):
        numpy_to_gdal_type.update(
            {
                numpy.dtype(numpy.int64): gdal.GDT_Int64,
                numpy.dtype(numpy.uint64): gdal.GDT_UInt64,
            }
        )
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    ny, nx = base_array.shape
    new_raster = retry_create(
        raster_driver,
        target_path,
        nx,
        ny,
        1,
        numpy_to_gdal_type[base_array.dtype],
        options=raster_driver_creation_tuple[1],
    )
    if projection_wkt is not None:
        new_raster.SetProjection(projection_wkt)
    new_raster.SetGeoTransform(
        [origin[0], pixel_size[0], 0.0, origin[1], 0.0, pixel_size[1]]
    )
    new_band = new_raster.GetRasterBand(1)
    if target_nodata is not None:
        new_band.SetNoDataValue(target_nodata)
    new_band.WriteArray(base_array)
    new_band = None
    new_raster = None


def raster_to_numpy_array(raster_path, band_id=1):
    """Read the entire contents of the raster band to a numpy array.

    Args:
        raster_path (str): path to raster.
        band_id (int): band in the raster to read.

    Return:
        numpy array contents of `band_id` in raster.

    """
    raster = gdal.OpenEx(raster_path, gdal.OF_RASTER)
    band = raster.GetRasterBand(band_id)
    array = band.ReadAsArray()
    band = None
    raster = None
    return array


def stitch_rasters(
    base_raster_path_band_list,
    resample_method_list,
    target_stitch_raster_path_band,
    overlap_algorithm="etch",
    area_weight_m2_to_wgs84=False,
    run_parallel=False,
    working_dir=None,
    stitch_blocksize=2**25,
    osr_axis_mapping_strategy=DEFAULT_OSR_AXIS_MAPPING_STRATEGY,
):
    """Stitch the raster in the base list into the existing target.

    Args:
        base_raster_path_band_list (sequence): sequence of raster path/band
            tuples to stitch into target.
        resample_method_list (sequence): a sequence of resampling methods
            which one to one map each path in ``base_raster_path_band_list``
            during resizing.  Each element must be one of
            "near|bilinear|cubic|cubicspline|lanczos|mode".
        target_stitch_raster_path_band (tuple): raster path/band tuple to an
            existing raster, values in ``base_raster_path_band_list`` will
            be stitched into this raster/band in the order they are in the
            list. The nodata value for the target band must be defined and
            will be written over with values from the base raster. Nodata
            values in the base rasters will not be written into the target.
            If the pixel size or projection are different between base and
            target the base is warped to the target's cell size and target
            with the interpolation method provided. If any part of the
            base raster lies outside of the target, that part of the base
            is ignored. A warning is logged if the entire base raster is
            outside of the target bounds.
        overlap_algorithm (str): this value indicates which algorithm to use
            when a raster is stitched on non-nodata values in the target
            stitch raster. It can be one of the following:
                'etch': write a value to the target raster only if the target
                    raster pixel is nodata. If the target pixel is non-nodata
                    ignore any additional values to write on that pixel.
                'replace': write a value to the target raster irrespective
                    of the value of the target raster
                'add': add the value to be written to the target raster to
                    any existing value that is there. If the existing value
                    is nodata, treat it as 0.0.
        area_weight_m2_to_wgs84 (bool): If ``True`` the stitched raster will
            be converted to a per-area value before reprojection to wgs84,
            then multiplied by the m^2 area per pixel in the wgs84 coordinate
            space. This is useful when the quantity being stitched is a total
            quantity per pixel rather than a per unit area density. Note
            this assumes input rasters are in a projected space of meters,
            if they are not the stitched output will be nonsensical.
        run_parallel (bool): If true, uses all CPUs to do warping if needed.
        working_dir (str): If not None, uses as working directory which is
            kept after run.
        stitch_blocksize (int): max blocksize to read when doing stitch
            default is 2**25.
        osr_axis_mapping_strategy (int): OSR axis mapping strategy for
            ``SpatialReference`` objects. Defaults to
            ``geoprocessing.DEFAULT_OSR_AXIS_MAPPING_STRATEGY``. This
            parameter should not be changed unless you know what you are
            doing.

    Return:
        None.
    """
    valid_overlap_algorithms = ["etch", "replace", "add"]
    if overlap_algorithm not in valid_overlap_algorithms:
        raise ValueError(
            f"overlap algorithm {overlap_algorithm} is not one of "
            f"{valid_overlap_algorithms}"
        )

    if not _is_raster_path_band_formatted(target_stitch_raster_path_band):
        raise ValueError(
            f"Expected raster path/band tuple for "
            f"target_stitch_raster_path_band but got "
            f'"{target_stitch_raster_path_band}"'
        )

    if len(base_raster_path_band_list) != len(resample_method_list):
        raise ValueError(
            f"Expected same number of elements in "
            f"`base_raster_path_band_list` as `resample_method_list` but "
            f"got {len(base_raster_path_band_list)} != "
            f"{len(resample_method_list)} respectively"
        )

    if not os.path.exists(target_stitch_raster_path_band[0]):
        raise ValueError(
            f"Target stitch raster does not exist: "
            f'"{target_stitch_raster_path_band[0]}"'
        )
    gis_type = get_gis_type(target_stitch_raster_path_band[0])
    from ecoshard.geoprocessing import RASTER_TYPE

    if gis_type != RASTER_TYPE:
        raise ValueError(
            f"Target stitch raster is not a raster. "
            f'Location: "{target_stitch_raster_path_band[0]}" '
            f"GIS type: {gis_type}"
        )
    target_raster_info = get_raster_info(target_stitch_raster_path_band[0])
    if target_stitch_raster_path_band[1] > len(target_raster_info["nodata"]):
        raise ValueError(
            f"target_stitch_raster_path_band refers to a band that exceeds "
            f"the number of bands in the raster:\n"
            f"target_stitch_raster_path_band[1]: "
            f"{target_stitch_raster_path_band[1]} "
            f'n bands: {len(target_raster_info["nodata"])}'
        )

    target_nodata = target_raster_info["nodata"][target_stitch_raster_path_band[1] - 1]
    if target_nodata is None:
        raise ValueError(
            f'target stitch raster at "{target_stitch_raster_path_band[0]} "'
            f"nodata value is `None`, expected non-`None` value"
        )

    n_attempts = 10
    while True:
        try:
            target_raster = gdal.OpenEx(
                target_stitch_raster_path_band[0],
                gdal.OF_RASTER | gdal.GA_Update,
            )
            break
        except RuntimeError as e:
            if n_attempts > 0:
                LOGGER.warning(
                    f"attempt {11-n_attempts}: trouble opening {target_stitch_raster_path_band[0]} "
                    f"with this exception : {e}"
                )
                time.sleep((11 - n_attempts) * 0.5)
                n_attempts -= 1
            else:
                for proc in psutil.process_iter(["pid", "name", "open_files"]):
                    try:
                        for f in proc.info["open_files"] or []:
                            if os.path.abspath(f.path) == os.path.abspath(
                                target_stitch_raster_path_band[0]
                            ):
                                LOGGER.error(
                                    f"for file {target_stitch_raster_path_band[0]} -- PID: {proc.info['pid']}, Name: {proc.info['name']}"
                                )
                    except psutil.AccessDenied:
                        pass
                raise
    target_band = target_raster.GetRasterBand(target_stitch_raster_path_band[1])
    target_inv_gt = gdal.InvGeoTransform(target_raster_info["geotransform"])
    target_raster_x_size, target_raster_y_size = target_raster_info["raster_size"]

    if working_dir is None:
        top_workspace_dir = tempfile.mkdtemp(
            dir=os.path.dirname(target_stitch_raster_path_band[0]),
            prefix="stitch_rasters_workspace_",
        )
    else:
        top_workspace_dir = working_dir
    task_graph = taskgraph.TaskGraph(
        top_workspace_dir,
        multiprocessing.cpu_count() if run_parallel else -1,
        15.0,
    )
    empty_task = task_graph.add_task()
    warp_list = []
    for (raster_path, raster_band_id), resample_method in zip(
        base_raster_path_band_list, resample_method_list
    ):
        LOGGER.info(
            f"stitching {(raster_path, raster_band_id)} into "
            f"{target_stitch_raster_path_band}"
        )
        raster_info = get_raster_info(raster_path)

        projected_raster_bounding_box = transform_bounding_box(
            raster_info["bounding_box"],
            raster_info["projection_wkt"],
            target_raster_info["projection_wkt"],
        )

        try:
            # merge the bounding boxes only to see if they don't intersect
            _ = merge_bounding_box_list(
                [
                    projected_raster_bounding_box,
                    target_raster_info["bounding_box"],
                ],
                "intersection",
            )
        except ValueError:
            LOGGER.warning(
                f'the raster at "{raster_path}"" does not intersect the '
                f'stitch raster at "{target_stitch_raster_path_band[0]}", '
                f"skipping..."
            )
            continue

        # use this to determine if we need to warp and delete if we did at
        # the end
        if (
            raster_info["projection_wkt"] == target_raster_info["projection_wkt"]
            and raster_info["pixel_size"] == target_raster_info["pixel_size"]
        ):
            warped_raster = False
            base_stitch_raster_path = raster_path
            task = empty_task
            local_working_dir = None
        else:
            local_working_dir = os.path.join(
                top_workspace_dir,
                hashlib.sha256(raster_path.encode("utf-8")).hexdigest()[:8],
            )
            os.makedirs(local_working_dir, exist_ok=True)
            base_stitch_raster_path = os.path.join(
                local_working_dir, os.path.basename(raster_path)
            )
            task = task_graph.add_task(
                func=warp_raster,
                args=(
                    raster_path,
                    target_raster_info["pixel_size"],
                    base_stitch_raster_path,
                    resample_method,
                ),
                kwargs={
                    "target_projection_wkt": target_raster_info["projection_wkt"],
                    "working_dir": local_working_dir,
                    "osr_axis_mapping_strategy": osr_axis_mapping_strategy,
                },
                target_path_list=[base_stitch_raster_path],
                task_name=f"warp {base_stitch_raster_path}",
            )
            warped_raster = True

            if warped_raster and area_weight_m2_to_wgs84:
                # determine base area per pixel currently and area per pixel
                # once it is projected to wgs84 pixel sizes
                base_pixel_area_m2 = abs(numpy.prod(raster_info["pixel_size"]))
                base_stitch_raster_info = get_raster_info(base_stitch_raster_path)
                _, lat_min, _, lat_max = base_stitch_raster_info["bounding_box"]
                n_rows = base_stitch_raster_info["raster_size"][1]
                # this column is a longitude invariant latitude variant pixel
                # area for scaling area dependent values
                m2_area_per_lat = _create_latitude_m2_area_column(
                    lat_min, lat_max, n_rows
                )

                base_stitch_nodata = base_stitch_raster_info["nodata"][0]
                scaled_raster_path = os.path.join(
                    local_working_dir,
                    f"scaled_{os.path.basename(base_stitch_raster_path)}",
                )
                # multiply the pixels in the resampled raster by the ratio of
                # the pixel area in the wgs84 units divided by the area of the
                # original pixel
                task = task_graph.add_task(
                    func=raster_calculator,
                    args=(
                        [
                            (base_stitch_raster_path, 1),
                            (base_stitch_nodata, "raw"),
                            m2_area_per_lat / base_pixel_area_m2,
                            (
                                _GDAL_TYPE_TO_NUMPY_LOOKUP[
                                    target_raster_info["datatype"]
                                ],
                                "raw",
                            ),
                        ],
                        _scale_mult_op,
                        scaled_raster_path,
                        target_raster_info["datatype"],
                        base_stitch_nodata,
                    ),
                    target_path_list=[scaled_raster_path],
                    dependent_task_list=[task],
                    task_name=f"scale raster {base_stitch_raster_path}",
                )

                # swap the result to base stitch so the rest of the function
                # operates on the area scaled raster
                base_stitch_raster_path = scaled_raster_path

        warp_list.append(
            (
                task,
                warped_raster,
                base_stitch_raster_path,
                raster_info,
                local_working_dir,
            )
        )
    for (
        task,
        warped_raster,
        base_stitch_raster_path,
        raster_info,
        local_working_dir,
    ) in warp_list:
        LOGGER.info(
            f"waiting for {(base_stitch_raster_path, raster_band_id)} " f"to complete"
        )
        task.join()

        base_raster = gdal.OpenEx(base_stitch_raster_path, gdal.OF_RASTER)
        base_gt = base_raster.GetGeoTransform()
        base_band = base_raster.GetRasterBand(raster_band_id)
        base_nodata = base_band.GetNoDataValue()
        # Get the target upper left xoff/yoff w/r/t the stitch raster 0,0
        # coordinates
        target_to_base_xoff, target_to_base_yoff = [
            int(_)
            for _ in gdal.ApplyGeoTransform(
                target_inv_gt, *gdal.ApplyGeoTransform(base_gt, 0, 0)
            )
        ]
        block_list = list(
            iterblocks(
                (base_stitch_raster_path, raster_band_id),
                offset_only=True,
                largest_block=stitch_blocksize,
            )
        )
        last_report_time = time.time()
        for index, offset_dict in enumerate(block_list):
            if time.time() - last_report_time > _LOGGING_PERIOD:
                LOGGER.info(
                    f"{index/len(block_list)*100:.2f}% complete stitching "
                    f"on {target_stitch_raster_path_band[0]}"
                )
                last_report_time = time.time()
            _offset_vars = {}
            overlap = True
            for (
                target_to_base_off,
                off_val,
                target_off_id,
                off_clip_id,
                win_size_id,
                raster_size,
            ) in [
                (
                    target_to_base_xoff,
                    offset_dict["xoff"],
                    "target_xoff",
                    "xoff_clip",
                    "win_xsize",
                    target_raster_x_size,
                ),
                (
                    target_to_base_yoff,
                    offset_dict["yoff"],
                    "target_yoff",
                    "yoff_clip",
                    "win_ysize",
                    target_raster_y_size,
                ),
            ]:
                _offset_vars[target_off_id] = target_to_base_off + off_val
                if _offset_vars[target_off_id] >= raster_size:
                    overlap = False
                    break
                # how far to move right to get in the target raster
                _offset_vars[off_clip_id] = 0
                _offset_vars[win_size_id] = offset_dict[win_size_id]
                if _offset_vars[target_off_id] < 0:
                    # if negative, move the offset so it's in range of the
                    # stitch raster and make the window smaller
                    _offset_vars[off_clip_id] = -_offset_vars[target_off_id]
                    _offset_vars[win_size_id] += _offset_vars[target_off_id]
                if _offset_vars[off_clip_id] >= _offset_vars[win_size_id] or (
                    _offset_vars[win_size_id] < 0
                ):
                    # its too far left/right for the whole window
                    overlap = False
                    break
                # make the _offset_vars[win_size_id] smaller if it shifts
                # off the target window
                if (
                    _offset_vars[off_clip_id]
                    + _offset_vars[target_off_id]
                    + _offset_vars[win_size_id]
                    >= raster_size
                ):
                    _offset_vars[win_size_id] -= (
                        _offset_vars[off_clip_id]
                        + _offset_vars[target_off_id]
                        + _offset_vars[win_size_id]
                        - raster_size
                    )

                # deal with the case where the base_stitch_raster_path is
                # outside of the bounds of the
            if not overlap:
                continue

            target_array = target_band.ReadAsArray(
                xoff=_offset_vars["target_xoff"] + _offset_vars["xoff_clip"],
                yoff=_offset_vars["target_yoff"] + _offset_vars["yoff_clip"],
                win_xsize=_offset_vars["win_xsize"],
                win_ysize=_offset_vars["win_ysize"],
            )
            target_nodata_mask = numpy.isclose(target_array, target_nodata)
            base_array = base_band.ReadAsArray(
                xoff=offset_dict["xoff"] + _offset_vars["xoff_clip"],
                yoff=offset_dict["yoff"] + _offset_vars["yoff_clip"],
                win_xsize=_offset_vars["win_xsize"],
                win_ysize=_offset_vars["win_ysize"],
            )

            if base_nodata is not None:
                base_nodata_mask = numpy.isclose(base_array, base_nodata)
            else:
                base_nodata_mask = numpy.zeros(base_array.shape, dtype=bool)

            if overlap_algorithm == "etch":
                # place values only where target is nodata
                valid_mask = ~base_nodata_mask & target_nodata_mask
                target_array[valid_mask] = base_array[valid_mask]
            elif overlap_algorithm == "replace":
                # write valid values into the target -- disregard any
                # existing values in the target
                valid_mask = ~base_nodata_mask
                target_array[valid_mask] = base_array[valid_mask]
            elif overlap_algorithm == "add":
                # add values to the target and treat target nodata as 0.
                valid_mask = ~base_nodata_mask
                masked_target_array = target_array[valid_mask]
                target_array_nodata_mask = numpy.isclose(
                    masked_target_array, target_nodata
                )
                target_array[valid_mask] = base_array[valid_mask] + numpy.where(
                    target_array_nodata_mask, 0, masked_target_array
                )
            else:
                raise RuntimeError(
                    f"overlap_algorithm {overlap_algorithm} was not defined "
                    f"but also not detected earlier -- this should never "
                    f"happen"
                )

            target_band.WriteArray(
                target_array,
                xoff=_offset_vars["target_xoff"] + _offset_vars["xoff_clip"],
                yoff=_offset_vars["target_yoff"] + _offset_vars["yoff_clip"],
            )

        base_raster = None
        base_band = None
        if warped_raster and working_dir is None:
            # if not user defined working dir then delete
            shutil.rmtree(local_working_dir)

    target_raster = None
    target_band = None
    if working_dir is None:
        shutil.rmtree(top_workspace_dir)


def get_utm_zone(lng, lat):
    """Given lng/lat coordinates return EPSG code of UTM zone.

    Note this only correctly calculates the main longitudnnal UTM zones and
    will incorrectly calcualte the UTM zones for the corner cases in
    very Northern Europe and Russia.

    Args:
        lng/lat (float): longitude and latitude in degrees.

    Returns:
        epsg code for the primary utm zone containing the point (lng/lat)
    """
    utm_code = (math.floor((lng + 180) / 6) % 60) + 1
    lat_code = 6 if lat > 0 else 7
    epsg_code = int("32%d%02d" % (lat_code, utm_code))
    return epsg_code


def _m2_area_of_wg84_pixel(pixel_size, center_lat):
    """Calculate m^2 area of a square wgs84 pixel.

    Adapted from: https://gis.stackexchange.com/a/127327/2397

    Args:
        pixel_size (float): length of side of a square pixel in degrees.
        center_lat (float): latitude of the center of the pixel. Note this
            value +/- half the `pixel-size` must not exceed 90/-90 degrees
            latitude or an invalid area will be calculated.

    Returns:
        Area of square pixel of side length `pixel_size` centered at
        `center_lat` in m^2.

    """
    a = 6378137  # meters
    b = 6356752.3142  # meters
    e = math.sqrt(1 - (b / a) ** 2)
    area_list = []
    for f in [center_lat + pixel_size / 2, center_lat - pixel_size / 2]:
        zm = 1 - e * math.sin(math.radians(f))
        zp = 1 + e * math.sin(math.radians(f))
        area_list.append(
            math.pi
            * b**2
            * (math.log(zp / zm) / (2 * e) + math.sin(math.radians(f)) / (zp * zm))
        )
    return abs(pixel_size / 360.0 * (area_list[0] - area_list[1]))


def get_pixel_area_in_target_projection(raster_path, target_projection_wkt):
    """Calculate average pixel area in raster if transformed to the target.

    Args:
        raster_path (str): path to a raster
        target_projection_wkt (str): a projection coordinate system in
            well known text.

    Returns:
        tuple of base pixel area and area if transformed to target projection.
    """
    # Get geotransform
    raster_info = get_raster_info(raster_path)

    base_sr = osr.SpatialReference(raster_info["projection_wkt"])
    target_sr = osr.SpatialReference(target_projection_wkt)

    center_x = raster_info["bounding_box"][2] - raster_info["bounding_box"][0]
    center_y = raster_info["bounding_box"][3] - raster_info["bounding_box"][1]

    min_x = center_x - raster_info["geotransform"][1] / 2
    max_x = center_x + raster_info["geotransform"][1] / 2

    min_y = center_y - raster_info["geotransform"][5] / 2
    max_y = center_y + raster_info["geotransform"][5] / 2

    # Create a pixel
    ring = ogr.Geometry(ogr.wkbLinearRing)
    ring.AddPoint(min_x, min_y)  # bottom left
    ring.AddPoint(min_x, max_y)  # top left
    ring.AddPoint(max_x, max_y)  # top right
    ring.AddPoint(max_x, min_y)  # bottom right
    ring.AddPoint(min_x, min_y)  # closing the polygon - back to bottom left

    # Create polygon
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)

    coord_trans = osr.CreateCoordinateTransformation(base_sr, target_sr)

    pre_area = poly.GetArea()

    poly.Transform(coord_trans)

    return pre_area, poly.GetArea()


def _create_latitude_m2_area_column(lat_min, lat_max, n_pixels):
    """Create a (n, 1) sized numpy array with m^2 areas in each element.

    Creates a per pixel m^2 area array that varies with changes in latitude.
    This array can be used to scale values by area when converting to or
    from a WGS84 projection to a projected one.

    Args:
        lat_max (float): maximum latitude in the bound
        lat_min (float): minimum latitude in the bound
        n_pixels (int): number of pixels to create for the column. The
            size of the target square pixels are (lat_max-lat_min)/n_pixels
            degrees per side.

    Return:
        A (n, 1) sized numpy array whose elements are the m^2 areas in each
        element estimated by the latitude value at the center of each pixel.
    """
    pixel_size = (lat_max - lat_min) / n_pixels
    center_lat_array = numpy.linspace(
        lat_min + pixel_size / 2, lat_max - pixel_size / 2, n_pixels
    )
    area_array = numpy.array(
        [_m2_area_of_wg84_pixel(pixel_size, lat) for lat in reversed(center_lat_array)]
    ).reshape((n_pixels, 1))
    return area_array


def _scale_mult_op(base_array, base_nodata, scale, datatype):
    """Scale non-nodata by scale."""
    result = base_array.astype(datatype)
    if base_nodata is not None:
        valid_mask = ~numpy.isclose(base_array, base_nodata)
    else:
        valid_mask = numpy.ones(base_array.shape, dtype=bool)
    result[valid_mask] = result[valid_mask] * scale[valid_mask]
    return result


def _unique_in_block(raster_path_band, offset_data):
    """Return set of unique elements in subarray.

    Intermediate function used to parallelize ``get_unique_values``.

    Args:
        raster_path_band (tuple): raster path/band tuple to get unique
            values from
        offset_data (dict): dictionary containing at least "xoff", "yoff",
            "xwin_size", "ywin_size" defining the slice in ``raster_path_band``
            to detect unique values over.

    Returns:
        set of unique non-nodata values in the window specified.

    """
    raster = gdal.OpenEx(raster_path_band[0], gdal.OF_RASTER)
    band = raster.GetRasterBand(raster_path_band[1])
    nodata = band.GetNoDataValue()
    array = band.ReadAsArray(**offset_data)
    band = None
    raster = None
    finite_mask = numpy.isfinite(array)
    if nodata is not None:
        valid_mask = (array != nodata) & finite_mask
        unique_set = set(numpy.unique(array[valid_mask]))
    else:
        unique_set = set(numpy.unique(array[finite_mask]))
    return unique_set


def get_unique_values(raster_path_band):
    """Return a list of non-nodata unique values from `raster_path`.

    Args:
        raster_path_band (tuple): path to raster path/band tuple

    Returns:
        set of unique numerical values foudn in raster_path_band except
        for the nodata value.

    """
    LOGGER.info("starting unique values")
    unique_set = set()
    largest_block = _LARGEST_ITERBLOCK * multiprocessing.cpu_count() * 2**8
    offset_list = list(
        iterblocks((raster_path_band), offset_only=True, largest_block=largest_block)
    )
    offset_list_len = len(offset_list)
    last_time = time.time()
    n_workers = min(multiprocessing.cpu_count(), len(offset_list))

    with concurrent.futures.ProcessPoolExecutor(
        n_workers, initializer=_nice_process
    ) as executor:
        # this forces processpool to terminate if parent dies
        LOGGER.info("registering atexit ")
        atexit.register(lambda: _shutdown_pool(executor))

        if psutil.WINDOWS:
            sig_list = [signal.SIGABRT, signal.SIGINT, signal.SIGTERM]
        else:
            sig_list = [signal.SIGTERM, signal.SIGINT]
        for sig in sig_list:
            signal.signal(sig, lambda: _shutdown_pool(executor))

        LOGGER.info("submitting result")
        result_list = [
            executor.submit(_unique_in_block, raster_path_band, offset_data)
            for offset_data in offset_list
        ]

        for offset_id, future in enumerate(
            concurrent.futures.as_completed(result_list)
        ):
            unique_set |= future.result()

            if time.time() - last_time > 5.0:
                LOGGER.info(
                    f"processing {(offset_id+1)/(offset_list_len)*100:.2f}% "
                    f"({offset_id+1} of "
                    f"{offset_list_len}) complete on "
                    f"{raster_path_band}. set size: {len(unique_set)}"
                )
                last_time = time.time()

    return unique_set


def _shutdown_pool(executor):
    executor.shutdown()


def _nice_process():
    """Make this process nice."""
    if psutil.WINDOWS:
        # Windows' scheduler doesn't use POSIX niceness.
        PROCESS_LOW_PRIORITY = psutil.BELOW_NORMAL_PRIORITY_CLASS
    else:
        # On POSIX, use system niceness.
        # -20 is high priority, 0 is normal priority, 19 is low priority.
        # 10 here is an arbitrary selection that's probably nice enough.
        PROCESS_LOW_PRIORITY = 10
    process = psutil.Process()
    process.nice(PROCESS_LOW_PRIORITY)


def single_thread_raster_calculator(
    base_raster_path_band_const_list,
    local_op,
    target_raster_path,
    datatype_target,
    nodata_target,
    calc_raster_stats=True,
    largest_block=_LARGEST_ITERBLOCK,
    max_timeout=_MAX_TIMEOUT,
    allow_different_blocksize=False,
    raster_driver_creation_tuple=DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS,
):
    """Apply local a raster operation on a stack of rasters.

    This function applies a user defined function across a stack of
    rasters' pixel stack. The rasters in ``base_raster_path_band_list`` must
    be spatially aligned and have the same cell sizes.

    Args:
        base_raster_path_band_const_list (sequence): a sequence containing:

            * ``(str, int)`` tuples, referring to a raster path/band index pair
              to use as an input.
            * ``numpy.ndarray`` s of up to two dimensions.  These inputs must
              all be broadcastable to each other AND the size of the raster
              inputs.
            * ``(object, 'raw')`` tuples, where ``object`` will be passed
              directly into the ``local_op``.

            All rasters must have the same raster size. If only arrays are
            input, numpy arrays must be broadcastable to each other and the
            final raster size will be the final broadcast array shape. A value
            error is raised if only "raw" inputs are passed.
        local_op (function): a function that must take in as many parameters as
            there are elements in ``base_raster_path_band_const_list``. The
            parameters in ``local_op`` will map 1-to-1 in order with the values
            in ``base_raster_path_band_const_list``. ``raster_calculator`` will
            call ``local_op`` to generate the pixel values in ``target_raster``
            along memory block aligned processing windows. Note any
            particular call to ``local_op`` will have the arguments from
            ``raster_path_band_const_list`` sliced to overlap that window.
            If an argument from ``raster_path_band_const_list`` is a
            raster/path band tuple, it will be passed to ``local_op`` as a 2D
            numpy array of pixel values that align with the processing window
            that ``local_op`` is targeting. A 2D or 1D array will be sliced to
            match the processing window and in the case of a 1D array tiled in
            whatever dimension is flat. If an argument is a scalar it is
            passed as as scalar.
            The return value must be a 2D array of the same size as any of the
            input parameter 2D arrays and contain the desired pixel values
            for the target raster.
        target_raster_path (string): the path of the output raster.  The
            projection, size, and cell size will be the same as the rasters
            in ``base_raster_path_const_band_list`` or the final broadcast
            size of the constant/ndarray values in the list.
        datatype_target (gdal datatype; int): the desired GDAL output type of
            the target raster.
        nodata_target (numerical value): the desired nodata value of the
            target raster.
        calc_raster_stats (boolean): If True, calculates and sets raster
            statistics (min, max, mean, and stdev) for target raster.
        largest_block (int): Attempts to internally iterate over raster blocks
            with this many elements.  Useful in cases where the blocksize is
            relatively small, memory is available, and the function call
            overhead dominates the iteration.  Defaults to 2**20.  A value of
            anything less than the original blocksize of the raster will
            result in blocksizes equal to the original size.
        max_timeout (float): amount of time in seconds to wait for stats
            worker thread to join. Default is _MAX_TIMEOUT.
        allow_different_blocksize (bool): If False, raise exception if input
            rasters are different blocksizes.
        raster_driver_creation_tuple (tuple): a tuple containing a GDAL driver
            name string as the first element and a GDAL creation options
            tuple/list as the second. Defaults to
            geoprocessing.DEFAULT_GTIFF_CREATION_TUPLE_OPTIONS.

    Return:
        None

    Raises:
        ValueError: invalid input provided

    """
    if not base_raster_path_band_const_list:
        raise ValueError(
            "`base_raster_path_band_const_list` is empty and "
            "should have at least one value."
        )

    # It's a common error to not pass in path/band tuples, so check for that
    # and report error if so
    bad_raster_path_list = False
    if not isinstance(base_raster_path_band_const_list, (list, tuple)):
        bad_raster_path_list = True
    else:
        for value in base_raster_path_band_const_list:
            if (
                not _is_raster_path_band_formatted(value)
                and not isinstance(value, numpy.ndarray)
                and not (
                    isinstance(value, tuple) and len(value) == 2 and value[1] == "raw"
                )
            ):
                bad_raster_path_list = True
                break
    if bad_raster_path_list:
        raise ValueError(
            "Expected a sequence of path / integer band tuples, "
            "ndarrays, or (value, 'raw') pairs for "
            "`base_raster_path_band_const_list`, instead got: "
            "%s" % pprint.pformat(base_raster_path_band_const_list)
        )

    # check that any rasters exist on disk and have enough bands
    not_found_paths = []
    gdal.PushErrorHandler("CPLQuietErrorHandler")
    base_raster_path_band_list = [
        path_band
        for path_band in base_raster_path_band_const_list
        if _is_raster_path_band_formatted(path_band)
    ]
    for value in base_raster_path_band_list:
        if gdal.OpenEx(value[0], gdal.OF_RASTER) is None:
            not_found_paths.append(value[0])
    gdal.PopErrorHandler()
    if not_found_paths:
        raise ValueError(
            "The following files were expected but do not exist on the "
            "filesystem: " + str(not_found_paths)
        )

    # check that band index exists in raster
    invalid_band_index_list = []
    for value in base_raster_path_band_list:
        raster = gdal.OpenEx(value[0], gdal.OF_RASTER)
        if not (1 <= value[1] <= raster.RasterCount):
            invalid_band_index_list.append(value)
        raster = None
    if invalid_band_index_list:
        raise ValueError(
            "The following rasters do not contain requested band "
            "indexes: %s" % invalid_band_index_list
        )

    # check that the target raster is not also an input raster
    if target_raster_path in [x[0] for x in base_raster_path_band_list]:
        raise ValueError(
            "%s is used as a target path, but it is also in the base input "
            "path list %s" % (target_raster_path, str(base_raster_path_band_const_list))
        )

    # check that raster inputs are all the same dimensions
    raster_info_list = []
    geospatial_info_set = dict()
    for path_band in base_raster_path_band_const_list:
        if _is_raster_path_band_formatted(path_band):
            raster_info = get_raster_info(path_band[0])
            raster_info_list.append(raster_info)
            geospatial_info_set[raster_info["raster_size"]] = path_band

    if len(geospatial_info_set) > 1:
        raise ValueError(
            "Input Rasters are not the same dimensions. The "
            "following raster are not identical %s" % str(geospatial_info_set)
        )

    numpy_broadcast_list = [
        x for x in base_raster_path_band_const_list if isinstance(x, numpy.ndarray)
    ]
    stats_worker_thread = None
    try:
        # numpy.broadcast can only take up to 32 arguments, this loop works
        # around that restriction:
        while len(numpy_broadcast_list) > 1:
            numpy_broadcast_list = [
                numpy.broadcast(*numpy_broadcast_list[:32])
            ] + numpy_broadcast_list[32:]
        if numpy_broadcast_list:
            numpy_broadcast_size = numpy_broadcast_list[0].shape
    except ValueError:
        # this gets raised if numpy.broadcast fails
        raise ValueError(
            "Numpy array inputs cannot be broadcast into a single shape %s"
            % numpy_broadcast_list
        )

    if numpy_broadcast_list and len(numpy_broadcast_list[0].shape) > 2:
        raise ValueError(
            "Numpy array inputs must be 2 dimensions or less %s" % numpy_broadcast_list
        )

    # if there are both rasters and arrays, check the numpy shape will
    # be broadcastable with raster shape
    if raster_info_list and numpy_broadcast_list:
        # geospatial lists x/y order and numpy does y/x so reverse size list
        raster_shape = tuple(reversed(raster_info_list[0]["raster_size"]))
        invalid_broadcast_size = False
        if len(numpy_broadcast_size) == 1:
            # if there's only one dimension it should match the last
            # dimension first, in the raster case this is the columns
            # because of the row/column order of numpy. No problem if
            # that value is ``1`` because it will be broadcast, otherwise
            # it should be the same as the raster.
            if (
                numpy_broadcast_size[0] != raster_shape[1]
                and numpy_broadcast_size[0] != 1
            ):
                invalid_broadcast_size = True
        else:
            for dim_index in range(2):
                # no problem if 1 because it'll broadcast, otherwise must
                # be the same value
                if (
                    numpy_broadcast_size[dim_index] != raster_shape[dim_index]
                    and numpy_broadcast_size[dim_index] != 1
                ):
                    invalid_broadcast_size = True
        if invalid_broadcast_size:
            raise ValueError(
                "Raster size %s cannot be broadcast to numpy shape %s"
                % (raster_shape, numpy_broadcast_size)
            )

    # create a "canonical" argument list that's bands, 2d numpy arrays, or
    # raw values only
    base_canonical_arg_list = []
    base_raster_list = []
    base_band_list = []
    canonical_base_raster_path_band_list = []
    for value in base_raster_path_band_const_list:
        # the input has been tested and value is either a raster/path band
        # tuple, 1d ndarray, 2d ndarray, or (value, 'raw') tuple.
        if _is_raster_path_band_formatted(value):
            # it's a raster/path band, keep track of open raster and band
            # for later so we can `None` them.
            canonical_base_raster_path_band_list.append(value)
            base_raster_list.append(gdal.OpenEx(value[0], gdal.OF_RASTER))
            base_band_list.append(base_raster_list[-1].GetRasterBand(value[1]))
            base_canonical_arg_list.append(base_band_list[-1])
        elif isinstance(value, numpy.ndarray):
            if value.ndim == 1:
                # easier to process as a 2d array for writing to band
                base_canonical_arg_list.append(value.reshape((1, value.shape[0])))
            else:  # dimensions are two because we checked earlier.
                base_canonical_arg_list.append(value)
        elif isinstance(value, tuple):
            base_canonical_arg_list.append(value)
        else:
            raise ValueError(
                "An unexpected ``value`` occurred. This should never happen. "
                "Value: %r" % value
            )

    # create target raster
    if raster_info_list:
        # if rasters are passed, the target is the same size as the raster
        n_cols, n_rows = raster_info_list[0]["raster_size"]
    elif numpy_broadcast_list:
        # numpy arrays in args and no raster result is broadcast shape
        # expanded to two dimensions if necessary
        if len(numpy_broadcast_size) == 1:
            n_rows, n_cols = 1, numpy_broadcast_size[0]
        else:
            n_rows, n_cols = numpy_broadcast_size
    else:
        raise ValueError(
            "Only (object, 'raw') values have been passed. Raster "
            "calculator requires at least a raster or numpy array as a "
            "parameter. This is the input list: %s"
            % pprint.pformat(base_raster_path_band_const_list)
        )

    if datatype_target not in _VALID_GDAL_TYPES:
        raise ValueError(
            "Invalid target type, should be a gdal.GDT_* type, received "
            '"%s"' % datatype_target
        )

    # create target raster
    raster_driver = gdal.GetDriverByName(raster_driver_creation_tuple[0])
    try:
        os.makedirs(os.path.dirname(target_raster_path), exist_ok=True)
    except FileNotFoundError:
        # happens when no directory
        pass
    LOGGER.debug(
        f"creating {target_raster_path} with " f"{raster_driver_creation_tuple[1]}"
    )
    target_raster = retry_create(
        raster_driver,
        target_raster_path,
        n_cols,
        n_rows,
        1,
        datatype_target,
        options=raster_driver_creation_tuple[1],
    )

    target_band = target_raster.GetRasterBand(1)
    if nodata_target is not None:
        target_band.SetNoDataValue(nodata_target)
    if base_raster_list:
        # use the first raster in the list for the projection and geotransform
        target_raster.SetProjection(base_raster_list[0].GetProjection())
        target_raster.SetGeoTransform(base_raster_list[0].GetGeoTransform())
    target_band.FlushCache()
    target_raster.FlushCache()

    try:
        last_time = time.time()
        LOGGER.debug("build block offest list")
        if len(canonical_base_raster_path_band_list) > 0:
            block_offset_list = list(
                iterblocks(
                    canonical_base_raster_path_band_list,
                    offset_only=True,
                    largest_block=largest_block,
                    skip_sparse=True,
                    allow_different_blocksize=allow_different_blocksize,
                )
            )
        else:
            block_offset_list = list(
                iterblocks(
                    (target_raster_path, 1),
                    offset_only=True,
                    largest_block=largest_block,
                    skip_sparse=False,
                    allow_different_blocksize=allow_different_blocksize,
                )
            )

        LOGGER.debug(f"process {len(block_offset_list)} blocks")

        if calc_raster_stats:
            # if this queue is used to send computed valid blocks of
            # the raster to an incremental statistics calculator worker
            stats_worker_queue = queue.Queue()
            exception_queue = queue.Queue()
        else:
            stats_worker_queue = None

        if calc_raster_stats:
            # To avoid doing two passes on the raster to calculate standard
            # deviation, we implement a continuous statistics calculation
            # as the raster is computed. This computational effort is high
            # and benefits from running in parallel. This queue and worker
            # takes a valid block of a raster and incrementally calculates
            # the raster's statistics. When ``None`` is pushed to the queue
            # the worker will finish and return a (min, max, mean, std)
            # tuple.
            LOGGER.info("starting stats_worker")
            stats_worker_thread = threading.Thread(
                target=geoprocessing_core.stats_worker,
                args=(stats_worker_queue,),
            )
            stats_worker_thread.daemon = True
            stats_worker_thread.start()
            LOGGER.info("started stats_worker %s", stats_worker_thread)

        pixels_processed = 0
        n_pixels = n_cols * n_rows

        # iterate over each block and calculate local_op
        for block_offset in block_offset_list:
            # read input blocks
            last_time = _invoke_timed_callback(
                last_time,
                lambda: LOGGER.info(
                    f"{float(pixels_processed) / n_pixels * 100.0:.2f}% "
                    f"complete on {target_raster_path}",
                ),
                _LOGGING_PERIOD,
            )
            offset_list = (block_offset["yoff"], block_offset["xoff"])
            blocksize = (block_offset["win_ysize"], block_offset["win_xsize"])
            data_blocks = []
            for value in base_canonical_arg_list:
                if isinstance(value, gdal.Band):
                    data_blocks.append(value.ReadAsArray(**block_offset))
                    # I've encountered the following error when a gdal raster
                    # is corrupt, often from multiple threads writing to the
                    # same file. This helps to catch the error early rather
                    # than lead to confusing values of ``data_blocks`` later.
                    if not isinstance(data_blocks[-1], numpy.ndarray):
                        raise ValueError(
                            f"got a {data_blocks[-1]} when trying to read "
                            f"{value.GetDataset().GetFileList()} at "
                            f"{block_offset}, expected numpy.ndarray."
                        )
                elif isinstance(value, numpy.ndarray):
                    # must be numpy array and all have been conditioned to be
                    # 2d, so start with 0:1 slices and expand if possible
                    slice_list = [slice(0, 1)] * 2
                    tile_dims = list(blocksize)
                    for dim_index in [0, 1]:
                        if value.shape[dim_index] > 1:
                            slice_list[dim_index] = slice(
                                offset_list[dim_index],
                                offset_list[dim_index] + blocksize[dim_index],
                            )
                            tile_dims[dim_index] = 1
                    data_blocks.append(numpy.tile(value[tuple(slice_list)], tile_dims))
                else:
                    # must be a raw tuple
                    data_blocks.append(value[0])

            target_block = local_op(*data_blocks)
            if target_block is None:
                # allow for short circuit
                pixels_processed += blocksize[0] * blocksize[1]
                continue

            if (
                not isinstance(target_block, numpy.ndarray)
                or target_block.shape != blocksize
            ):
                raise ValueError(
                    "Expected `local_op` to return a numpy.ndarray of "
                    "shape %s but got this instead: %s" % (blocksize, target_block)
                )

            target_band.WriteArray(
                target_block,
                yoff=block_offset["yoff"],
                xoff=block_offset["xoff"],
            )

            # send result to stats calculator
            if stats_worker_queue:
                # guard against an undefined nodata target
                if nodata_target is not None:
                    target_block = target_block[target_block != nodata_target]
                target_block = target_block.astype(numpy.float64).flatten()
                stats_worker_queue.put(target_block)

            pixels_processed += blocksize[0] * blocksize[1]

        LOGGER.info("100.0% complete")

        if calc_raster_stats:
            LOGGER.info("Waiting for raster stats worker result.")
            stats_worker_queue.put(None)
            stats_worker_thread.join(max_timeout)
            if stats_worker_thread.is_alive():
                LOGGER.error("stats_worker_thread.join() timed out")
                raise RuntimeError("stats_worker_thread.join() timed out")
            payload = stats_worker_queue.get(True, max_timeout)
            if payload is not None:
                target_min, target_max, target_mean, target_stddev = payload
                target_band.SetStatistics(
                    float(target_min),
                    float(target_max),
                    float(target_mean),
                    float(target_stddev),
                )
                target_band.FlushCache()
    except Exception:
        LOGGER.exception("exception encountered in raster_calculator")
        raise
    finally:
        # This block ensures that rasters are destroyed even if there's an
        # exception raised.
        base_band_list[:] = []
        base_raster_list[:] = []
        target_band.FlushCache()
        target_band = None
        target_raster.FlushCache()
        target_raster = None

        if calc_raster_stats and stats_worker_thread:
            if stats_worker_thread.is_alive():
                stats_worker_queue.put(None, True, max_timeout)
                LOGGER.info("Waiting for raster stats worker result.")
                stats_worker_thread.join(max_timeout)
                if stats_worker_thread.is_alive():
                    LOGGER.error("stats_worker_thread.join() timed out")
                    raise RuntimeError("stats_worker_thread.join() timed out")

            # check for an exception in the workers, otherwise get result
            # and pass to writer
            try:
                exception = exception_queue.get_nowait()
                LOGGER.error("Exception encountered at termination.")
                raise exception
            except queue.Empty:
                pass

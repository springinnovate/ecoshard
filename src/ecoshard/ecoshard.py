"""Main ecoshard module."""
import datetime
import hashlib
import logging
import json
import os
import re
import requests
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

from osgeo import gdal
import numpy
from .geoprocessing import geoprocessing
import scipy.stats

LOGGER = logging.getLogger(__name__)
gdal.SetCacheMax(2**26)

COG_TUPLE = ('COG', (
    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW',
    'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))


def add_metadata(raster_path, metadata, domain=None):
    """Add metadata to ``raster_path``

    Args:
        raster_path (str): path to existing raster to write medatadata
        metadata (str): metadata string to embed (replace) metadata in raster

    Returns:
        None
    """
    raster = gdal.OpenEx(raster_path, gdal.GA_Update | gdal.OF_RASTER)
    raster.SetMetadata(metadata, domain)
    raster = None


def get_metadata(raster_path, domain=None):
    """Read metadata and return as dict.

    Args:
        raster_path (str): path to existing raster

    Returns:
        dict of metadata embedded in raster
    """
    raster = gdal.OpenEx(raster_path, gdal.GA_ReadOnly | gdal.OF_RASTER)
    metadata = raster.GetMetadata_Dict(domain)
    raster = None
    return metadata


def hash_file(
        base_path, target_token_path=None, target_dir=None, rename=False,
        hash_algorithm='md5', hash_length=None, force=False):
    """Ecoshard file by hashing it and appending hash to filename.

    An EcoShard is the hashing of a file and the rename to the following
    format: [base name]_[hashalg]_[hash][base extension]. If the base path
    already is in this format a ValueError is raised unless `force` is True.

    Args:
        base_path (str): path to base file.
        target_token_path (str): if not None, this file is created and written
            with the timestamp at which the ecoshard was completed. This is
            useful for TaskGraph to note a file being created without a priori
            knowing the filename.
        target_dir (str): if present, the ecoshard is created in this
            directory. This value must be None if `rename` is True.
        rename (bool): if True, `base_path` is renamed to the ecoshard rather
            than a new file being created.
        hash_algorithm (str): a hash function id that exists in
            hashlib.algorithms_available.
        force (bool): if True and the base_path already is in ecoshard format
            the operation proceeds including the possibility that the
            base_path ecoshard file name is renamed to a new hash.
        hash_length (int): if not None, truncate length of hash to this
            many characters.

    Returns:
        Path to the new hashed file, even if it's a rename.

    """
    if target_dir and rename:
        raise ValueError(
            "`target_dir` is defined, but rename is True, either set "
            "`target_dir` to None, or rename to False.")

    if target_dir and not os.path.isdir(target_dir):
        LOGGER.warning('target directory %s does not exist, creating it now')
        os.makedirs(target_dir, exist_ok=True)

    base_filename = os.path.basename(base_path)
    prefix, extension = os.path.splitext(base_filename)
    match_result = re.match(
        '(.+)_(%s)_([0-9a-f])+%s' % (
            '|'.join(hashlib.algorithms_available), extension), base_filename)
    if match_result:
        if not force:
            raise ValueError(
                '%s seems to already be an ecoshard with algorithm %s and '
                'hash %s. Set `force=True` to overwrite.' % (
                    base_path, match_result.group(2), match_result.group(3)))
        else:
            LOGGER.warning(
                '%s is already in ecoshard format, but overriding because '
                '`force` is True.', base_path)
            prefix = match_result.group(1)

    LOGGER.debug('calculating hash for %s', base_path)
    hash_val = calculate_hash(base_path, hash_algorithm)
    if hash_length is not None:
        hash_val = hash_val[:hash_length]

    if target_dir is None:
        target_dir = os.path.dirname(base_path)
    ecoshard_path = os.path.join(target_dir, '%s_%s_%s%s' % (
        prefix, hash_algorithm, hash_val, extension))
    if rename:
        LOGGER.info('renaming %s to %s', base_path, ecoshard_path)
        os.rename(base_path, ecoshard_path)
    else:
        LOGGER.info('copying %s to %s', base_path, ecoshard_path)
        shutil.copyfile(base_path, ecoshard_path)

    if target_token_path:
        with open(target_token_path, 'w') as target_token_file:
            target_token_file.write(str(datetime.datetime.now()))

    return ecoshard_path


def stitch_rasters(base_raster_path_pattern, target_raster_path):
    """Stitch all the rasters in the base pattern to the target.

        base_raster_path_pattern (list):
        target_raster_path (str): target raster path
    """
    vrt_path = os.path.splitext(target_raster_path)[0] + '.vrt'
    subprocess.run(['gdalbuildvrt', vrt_path] + base_raster_path_pattern, check=True)
    compress_raster(vrt_path, target_raster_path, compression_algorithm='LZW')
    os.remove(vrt_path)


def build_overviews(
        base_raster_path, target_token_path=None,
        interpolation_method='near', overview_type='internal',
        rebuild_if_exists=False):
    """Build embedded overviews on raster.

    Args:
        base_raster_path (str): base raster file, must be a GDAL writable
            raster type.
        target_token_path (str): if not None, this file is created and written
            with a timestamp when overviews are successfully completed. This
            file is useful for a library like `taskgraph` that needs to see
            a file to know if an operation is successful.
        interpolation_method (str): one of 'average', 'average_magphase',
            'bilinear', 'cubic', 'cubicspline', 'gauss', 'lanczos', 'mode',
            'near', or 'none'.
        overview_type (str): 'internal' or 'external'
        rebuild_if_exists (bool): If True overviews will be rebuilt even if
            they already exist, otherwise just pass them over.


    Returns:
        None.

    """
    raster_open_mode = gdal.OF_RASTER
    if overview_type == 'internal':
        raster_open_mode |= gdal.GA_Update
    elif overview_type == 'external':
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'LZW')
    else:
        raise ValueError('invalid value for overview_type: %s' % overview_type)
    raster = gdal.OpenEx(base_raster_path, raster_open_mode)
    if not raster:
        raise ValueError(
            'could not open %s as a GDAL raster' % base_raster_path)
    band = raster.GetRasterBand(1)
    overview_count = band.GetOverviewCount()
    if overview_count == 0 or rebuild_if_exists:
        # either no overviews, or we are rebuliding them
        min_dimension = min(raster.RasterXSize, raster.RasterYSize)
        overview_levels = []
        current_level = 2
        while True:
            if min_dimension // current_level == 0:
                break
            overview_levels.append(current_level)
            current_level *= 2

        LOGGER.info(
            'building overviews for %s at the following levels %s' % (
                base_raster_path, overview_levels))
        raster.BuildOverviews(
            interpolation_method, overview_levels,
            callback=_make_logger_callback(
                'build overview for ' + os.path.basename(base_raster_path) +
                '%.2f/1.0 complete %s'))
    else:
        LOGGER.warning(
            'overviews already exist, set rebuild_if_exists=False to rebuild '
            'them anyway')

    if target_token_path:
        with open(target_token_path, 'w') as token_file:
            token_file.write(str(datetime.datetime.now()))


def validate(base_ecoshard_path):
    """Validate ecoshard path, through its filename.

    If `base_ecoshard_path` matches an EcoShard pattern, and the hash matches
    the actual hash, return True. Otherwise raise a ValueError.

    Args:
        base_ecoshard_path (str): path to an ecosharded file.

    Returns:
        True if `base_ecoshard_path` matches .*_[hashalg]_[hash][extension]
        and hashalg(base_ecoshard_path) = hash. Otherwise raise a ValueError.

    """
    base_filename = os.path.basename(base_ecoshard_path)
    prefix, extension = os.path.splitext(base_filename)
    match_result = re.match(
        '.+_([^_]+)_([0-9a-f]+)%s' % extension, base_filename)
    if not match_result:
        raise ValueError("%s does not match an ecoshard" % base_filename)
    hash_algorithm, hash_value = match_result.groups()
    calculated_hash = calculate_hash(
        base_ecoshard_path, hash_algorithm)
    if calculated_hash != match_result.group(2):
        raise ValueError(
            'hash does not match, calculated %s and expected %s '
            'on %s' % (calculated_hash, hash_value, base_filename))
    # if we got here the hash matched the calculated hash
    return True


def calculate_hash(file_path, hash_algorithm, buf_size=2**20):
    """Return a hex digest of `file_path`.

    Args:
        file_path (string): path to file to hash.
        hash_algorithm (string): a hash function id that exists in
            hashlib.algorithms_available.
        buf_size (int): number of bytes to read from `file_path` at a time
            for digesting.

    Returns:
        a hex digest with hash algorithm `hash_algorithm` of the binary
        contents of `file_path`.

    """
    hash_func = hashlib.new(hash_algorithm)
    with open(file_path, 'rb') as f:
        binary_data = f.read(buf_size)
        while binary_data:
            hash_func.update(binary_data)
            binary_data = f.read(buf_size)
    # We return the hash and CRC32 checksum in hexadecimal format
    return hash_func.hexdigest()


def _make_logger_callback(message, timeout=5.0):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.
        timeout (float): number of seconds to wait until print

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        current_time = time.time()
        if ((current_time - logger_callback.last_time) > timeout or
                (df_complete == 1.0 and
                 logger_callback.total_time >= timeout)):
            # In some multiprocess applications I was encountering a
            # ``p_progress_arg`` of None. This is unexpected and I suspect
            # was an issue for some kind of GDAL race condition. So I'm
            # guarding against it here and reporting an appropriate log
            # if it occurs.
            progress_arg = ''
            if p_progress_arg is not None:
                progress_arg = p_progress_arg[0]

            LOGGER.info(message, df_complete * 100, progress_arg)
            logger_callback.last_time = current_time
            logger_callback.total_time += current_time
    logger_callback.last_time = time.time()
    logger_callback.total_time = 0.0

    return logger_callback


def compress_raster(
        base_raster_path, target_compressed_path, compression_algorithm='LZW',
        compression_predictor=None):
    """Compress base raster to target.

    Args:
        base_raster_path (str): the original GIS raster file, presumably
            uncompressed.
        target_compressed_path (str): the desired output raster path with the
            defined compression algorithm applied to it.
        compression_algorithm (str): a valid GDAL compression algorithm eg
            'LZW', 'DEFLATE', and others defined in GDAL.
        compression_predictor (int): if defined uses the predictor in whatever
            compression algorithm is used. In most cases this only applies to
            LZW or DEFLATE.

    Returns:
        None.

    """
    gtiff_driver = gdal.GetDriverByName('GTiff')
    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    LOGGER.info('compress %s to %s' % (
        base_raster_path, target_compressed_path))
    if compression_predictor is None:
        datatype = geoprocessing.get_raster_info(base_raster_path)['numpy_type']
        if datatype in [numpy.float32, numpy.float64, float]:
            compression_predictor = 3
        else:
            compression_predictor = 2
    compressed_raster = gtiff_driver.CreateCopy(
        target_compressed_path, base_raster, options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=%s' % compression_algorithm,
            f'PREDICTOR={compression_predictor}',
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256'),
        callback=geoprocessing._make_logger_callback(
            f"copying {target_compressed_path} %.1f%% complete %s"))
    del compressed_raster


def download_url(url, target_path, skip_if_target_exists=False):
    """Download `url` to `target_path`.

    Args:
        url (str): url path to a file.
        target_path (str): desired output target path.
        skip_if_target_exists (bool): if True will not download a file if the
            path already exists on disk.

    Returns:
        None.

    """
    try:
        if skip_if_target_exists and os.path.exists(target_path):
            return
        with open(target_path, 'wb') as target_file:
            last_download_size = 0
            start_time = time.time()
            with urllib.request.urlopen(url) as url_stream:
                meta = url_stream.info()
                file_size = int(meta["Content-Length"])
                LOGGER.info(
                    "Downloading: %s Bytes: %s" % (target_path, file_size))

                downloaded_so_far = 0
                block_size = 2**20
                last_log_time = 0
                while True:
                    data_buffer = url_stream.read(block_size)
                    if not data_buffer:
                        break
                    downloaded_so_far += len(data_buffer)
                    target_file.write(data_buffer)
                    time_since_last_log = time.time() - last_log_time
                    if time_since_last_log > 5.0:
                        download_rate = (
                            (downloaded_so_far - last_download_size)/2**20) / (
                                float(time_since_last_log))
                        last_download_size = downloaded_so_far
                        status = r"%10dMB  [%3.2f%% @ %5.2fMB/s]" % (
                            downloaded_so_far/2**20, downloaded_so_far * 100. /
                            file_size, download_rate)
                        LOGGER.info(status)
                        last_log_time = time.time()
            total_time = time.time() - start_time + 1e-6  # just in case 0
            final_download_rate = downloaded_so_far/2**20 / float(
                total_time)
            status = r"%10dMB  [%3.2f%% @ %5.2fMB/s]" % (
                downloaded_so_far/2**20, downloaded_so_far * 100. /
                file_size, final_download_rate)
            LOGGER.info(status)
            target_file.flush()
            os.fsync(target_file.fileno())
    except Exception:
        LOGGER.exception(f'unable to download {url}')
        raise


def download_and_unzip(url, target_dir, target_token_path=None):
    """Download `url` to `target_dir` and touch `target_token_path`.

    Args:
        url (str): url to file to download
        target_dir (str): path to a local directory to download and unzip the
            file to. The contents will be unzipped into the same directory as
            the zipfile.
        target_token_path (str): If not None, a path a file to touch when
            the unzip is complete. This parameter is added to work well with
            the ecoshard library that expects a file to be created after
            an operation is complete. It may be complicated to list the files
            that are unzipped, so instead this file is created and contains
            the timestamp of when this function completed.

    Returns:
        None.

    """
    zipfile_path = os.path.join(target_dir, os.path.basename(url))
    LOGGER.info('download %s, to: %s', url, zipfile_path)
    download_url(url, zipfile_path)

    LOGGER.info('unzip %s', zipfile_path)
    with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    if target_token_path:
        with open(target_token_path, 'w') as touchfile:
            touchfile.write(f'unzipped {zipfile_path}')
    LOGGER.info('download an unzip for %s complete', zipfile_path)


def convolve_layer(
        base_raster_path, integer_factor, method, target_raster_path):
    """Convolve a raster to a lower size.

    Args:
        base_raster_path (str): base raster.
        integer_factor (int): integer number of pixels to aggregate by.
            i.e. 2 -- makes 2x2 into a 1x1, 3-- 3x3 to a 1x1.
        method (str): one of 'max', 'min', 'sum', 'average', 'mode'.
        target_raster_path (str): based off of `base_raster_path` with size
            reduced by `integer_factor`.

    Return:
        None.

    """
    base_raster_info = geoprocessing.get_raster_info(base_raster_path)
    n_cols, n_rows = numpy.ceil(base_raster_info['raster_size']).astype(
        numpy.int)
    n_cols_reduced = int(numpy.ceil(n_cols / integer_factor))
    n_rows_reduced = int(numpy.ceil(n_rows / integer_factor))
    nodata = base_raster_info['nodata'][0]
    geoprocessing.new_raster_from_base(
        base_raster_path, target_raster_path, base_raster_info['datatype'],
        [nodata], n_rows=n_rows_reduced, n_cols=n_cols_reduced)

    base_raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER)
    base_band = base_raster.GetRasterBand(1)
    base_geotransform = base_raster.GetGeoTransform()
    target_raster = gdal.OpenEx(
        target_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    target_geotransform = [
        base_geotransform[0],
        base_geotransform[1]*integer_factor,
        base_geotransform[2]*integer_factor,
        base_geotransform[3],
        base_geotransform[4]*integer_factor,
        base_geotransform[5]*integer_factor]
    target_raster.SetGeoTransform(target_geotransform)

    target_band = target_raster.GetRasterBand(1)

    block = base_band.GetBlockSize()
    cols_per_block = min(
        n_cols, max(1, block[0] // integer_factor) * integer_factor * 10)
    rows_per_block = min(
        n_rows, max(1, block[1] // integer_factor) * integer_factor * 10)
    n_col_blocks = int(numpy.ceil(n_cols / float(cols_per_block)))
    n_row_blocks = int(numpy.ceil(n_rows / float(rows_per_block)))
    for row_block_index in range(n_row_blocks):
        row_offset = row_block_index * rows_per_block
        row_block_width = n_rows - row_offset
        LOGGER.info('step %d of %d', row_block_index+1, n_row_blocks)
        if row_block_width > rows_per_block:
            row_block_width = rows_per_block
        for col_block_index in range(n_col_blocks):
            col_offset = col_block_index * cols_per_block
            col_block_width = n_cols - col_offset
            if col_block_width > cols_per_block:
                col_block_width = cols_per_block

            offset_dict = {
                'xoff': int(col_offset),
                'yoff': int(row_offset),
                'win_xsize': int(col_block_width),
                'win_ysize': int(row_block_width),
            }

            target_offset_x = offset_dict['xoff'] // integer_factor
            target_offset_y = offset_dict['yoff'] // integer_factor

            block_data = base_band.ReadAsArray(**offset_dict)

            rw = int(numpy.ceil(
                col_block_width / integer_factor) * integer_factor)
            rh = int(numpy.ceil(
                row_block_width / integer_factor) * integer_factor)
            w_pad = rw - col_block_width
            h_pad = rh - row_block_width
            j = rw // integer_factor
            k = rh // integer_factor
            if method == 'max':
                block_data_pad = numpy.pad(
                    block_data, ((0, h_pad), (0, w_pad)), mode='edge')
                reduced_block_data = block_data_pad.reshape(
                    k, integer_factor, j, integer_factor).max(axis=(-1, -3))
            elif method == 'min':
                block_data_pad = numpy.pad(
                    block_data, ((0, h_pad), (0, w_pad)), mode='edge')
                reduced_block_data = block_data_pad.reshape(
                    k, integer_factor, j, integer_factor).min(axis=(-1, -3))
            elif method == 'mode':
                block_data_pad = numpy.pad(
                    block_data, ((0, h_pad), (0, w_pad)), mode='edge')
                reduced_block_data = scipy.stats.mode(
                    block_data_pad.reshape(
                        k, integer_factor, j, integer_factor).swapaxes(
                            1, 2).reshape(k, j, integer_factor**2),
                    axis=2).mode.reshape(k, j)
            elif method == 'average':
                block_data_pad = numpy.pad(
                    block_data, ((0, h_pad), (0, w_pad)), mode='edge')
                block_data_pad_copy = block_data_pad.copy()
                # set any nodata to 0 so we don't average it strangely
                block_data_pad[numpy.isclose(block_data_pad, nodata)] = 0.0
                # straight average
                reduced_block_data = block_data_pad.reshape(
                    k, integer_factor, j, integer_factor).mean(
                    axis=(-1, -3))
                # this one is used to restore any nodata areas because they'll
                # still be nodata when it's done
                min_block_data = block_data_pad_copy.reshape(
                    k, integer_factor, j, integer_factor).min(
                    axis=(-1, -3))
                reduced_block_data[
                    numpy.isclose(min_block_data, nodata)] = nodata
            elif method == 'sum':
                block_data_pad = numpy.pad(
                    block_data, ((0, h_pad), (0, w_pad)), mode='edge')
                nodata_mask = numpy.isclose(block_data_pad, nodata)
                block_data_pad_copy = block_data_pad.copy()
                # set any nodata to 0 so we don't sum it strangely
                block_data_pad[nodata_mask] = 0.0
                # straight sum
                reduced_block_data = block_data_pad.reshape(
                    k, integer_factor, j, integer_factor).sum(
                    axis=(-1, -3))
                # this one is used to restore any nodata areas because they'll
                # still be nodata when it's done
                max_block_data = block_data_pad_copy.reshape(
                    k, integer_factor, j, integer_factor).max(
                    axis=(-1, -3))
                reduced_block_data[
                    numpy.isclose(max_block_data, nodata)] = nodata
            else:
                raise ValueError("unknown method: %s" % method)

            target_band.WriteArray(
                reduced_block_data, xoff=target_offset_x, yoff=target_offset_y)
            continue


def search(
        host_port, api_key, bounding_box, description, datetime, asset_id,
        catalog_list):
    """Search EcoServer.

    Args:
        host_port (str): `host:port` string pair to identify server to post
            publish request to.
        api_key (str): an api key that as write access to the catalog on the
            server.
        bounding_box (list): a float list of xmin,ymin,xmax,ymax to indicate
            the search area in lng/lat coordinates.
        description (str): description to partially search for
        datetime (str): utc range or open range to search for times like
            '2020-04-20 04:20:17.866142/2020-04-20 19:49:17.866142, '
            '../2020-04-20 19:49:17.866142', or
            '2020-04-20 04:20:17.866142/..'
        asset_id (str): to search for a substring match on ids in the catalog
        catalog_list (str): comma separated string of catalogs to search ex:
            'salo,nasa,joe'

    Returns:
        None

    """
    post_url = f'http://{host_port}/api/v1/search'

    if bounding_box:
        bounding_box_str = ','.join([str(val) for val in bounding_box])
    else:
        bounding_box_str = None

    LOGGER.debug('search posting to here: %s' % post_url)
    search_response = requests.post(
        post_url,
        params={'api_key': api_key},
        json=json.dumps({
            'bounding_box': bounding_box_str,
            'description': description,
            'datetime': datetime,
            'asset_id': asset_id,
            'catalog_list': catalog_list
        }))
    if not search_response:
        LOGGER.error(f'response from server: {search_response.text}')
        raise RuntimeError(search_response.text)

    response_dict = search_response.json()
    LOGGER.debug(response_dict)
    for index, feature in enumerate(response_dict['features']):
        LOGGER.info(
            f"{index}: {feature['id']}, "
            f"bbox: {feature['bbox']}, "
            f"utc_datetime: {feature['utc_datetime']}, "
            f"description: {feature['description']}")


def process_worker(file_path, args):
    """Do the ecoshard process commands to the given file path."""
    working_file_path = file_path
    LOGGER.info('processing %s', file_path)
    if args.cog:
        # create copy with COG
        cog_driver = gdal.GetDriverByName('COG')
        base_raster = gdal.OpenEx(file_path, gdal.OF_RASTER)
        cog_file_path = os.path.join(
            f'cog_{os.path.basename(file_path)}')
        options = ('COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES')
        LOGGER.info(f'convert {file_path} to COG {cog_file_path} with {options}')
        cog_raster = cog_driver.CreateCopy(
            cog_file_path, base_raster, options=options,
            callback=geoprocessing._make_logger_callback(
                f"COGing {cog_file_path} %.1f%% complete %s"))
        del cog_raster
        return

    if args.reduce_factor:
        method = args.reduce_factor[1]
        valid_methods = ["max", "min", "sum", "average", "mode"]
        if method not in valid_methods:
            LOGGER.error(
                '--reduce_method must be one of %s' % valid_methods)
            sys.exit(-1)
        convolve_layer(
            file_path, int(args.reduce_factor[0]),
            args.reduce_factor[1],
            args.reduce_factor[2])
        return

    if args.strip_hash:
        working_file_path = _remove_hash_from_filename(
            file_path, args.strip_hash)
        os.rename(file_path, working_file_path)
    else:
        working_file_path = file_path
    if args.ndv is not None:
        raster_info = geoprocessing.get_raster_info(working_file_path)
        current_nodata = raster_info['nodata'][0]
        if current_nodata is not None and not args.force:
            error_message = (
                f'--ndv flag is passed but {working_file_path} already has a '
                f'nodata value of {raster_info["nodata"]}, pass --force flag '
                f'to override this.')
            LOGGER.error(error_message)
            return error_message
        basename = os.path.basename(working_file_path)
        target_file_path = f'ndv_{args.ndv}_{basename}'
        LOGGER.info(
            f"replacing nodata value of {raster_info['nodata']} with "
            f"{args.ndv} on {working_file_path}")
        geoprocessing.raster_calculator(
            [(working_file_path, 1), (current_nodata, 'raw'),
             (args.ndv, 'raw')], _reclass_op, target_file_path,
            raster_info['datatype'], args.ndv)
        working_file_path = target_file_path

    if args.compress:
        prefix, suffix = os.path.splitext(working_file_path)
        compressed_filename = '%s_compressed%s' % (prefix, suffix)
        compress_raster(
            working_file_path, compressed_filename,
            compression_algorithm='DEFLATE')
        working_file_path = compressed_filename

    if args.buildoverviews:
        build_overviews(
            working_file_path, interpolation_method=args.interpolation_method,
            rebuild_if_exists=args.force)

    if args.validate:
        try:
            is_valid = validate(working_file_path)
            if is_valid:
                LOGGER.info('VALID ECOSHARD: %s', working_file_path)
            else:
                LOGGER.error(
                    'got a False, but no ValueError on validate? '
                    'that is not impobipible?')
        except ValueError:
            error_message = 'INVALID ECOSHARD: %s', working_file_path
            LOGGER.error(error_message)
            return error_message
    elif args.hash_file:
        hash_file(
            working_file_path, rename=args.rename, hash_algorithm=args.hashalg,
            hash_length=args.hash_length,
            force=args.force)


def _reclass_op(data_array, current_nodata, target_nodata):
    """Replace data array non-finte and current nodata to target."""
    result = numpy.copy(data_array)
    replace_block = ~numpy.isfinite(result)
    if current_nodata is not None:
        replace_block |= data_array == current_nodata
    result[replace_block] = target_nodata
    return result


def _remove_hash_from_filename(file_path, hash_id):
    """Returns new filename without hash.

    Assumes filename is of the form [prefix]_[hash_id]_[hash_value].ext

    Args:
        file_path (str): any filename
        hash_id (str): the value of the hash id ex md5

    Returns:
        [prefix].ext
    """
    prefix, suffix = os.path.splitext(file_path)
    file_match = re.match(
        f'(.*?)_{hash_id}_.*$', prefix)
    if file_match:
        rename_file_path = f'{file_match.group(1)}{suffix}'
        return rename_file_path
    else:
        raise ValueError(
            f"could not find a hash matching '{hash_id}'' in the filename "
            f"'{file_path}'")

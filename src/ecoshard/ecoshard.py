"""Main ecoshard module."""
import time
import datetime
import shutil
import hashlib
import re
import os
import logging
import urllib.request

from osgeo import gdal

LOGGER = logging.getLogger(__name__)


def hash_file(
        base_path, target_token_path=None, target_dir=None, rename=False,
        hash_algorithm='md5', force=False):
    """Ecoshard file by hashing it and appending hash to filename.

    An EcoShard is the hashing of a file and the rename to the following
    format: [base name]_[hashalg]_[hash][base extension]. If the base path
    already is in this format a ValueError is raised unless `force` is True.

    Parameters:
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

    Returns:
        None.

    """
    if target_dir and rename:
        raise ValueError(
            "`target_dir` is defined, but rename is True, either set "
            "`target_dir` to None, or rename to False.")

    if target_dir and not os.path.isdir(target_dir):
        LOGGER.warning('target directory %s does not exist, creating it now')
        try:
            os.makedirs(target_dir)
        except OSError:
            # this would never happen unless there was some concurrency that
            # created the target dir after the test, this guards against it.
            LOGGER.exception('failed to make %s', target_dir)

    base_filename = os.path.basename(base_path)
    prefix, extension = os.path.splitext(base_filename)
    match_result = re.match(
        '(.+)_([^_]+)_([0-9a-f]+)%s' % extension, base_filename)
    if match_result:
        if not force:
            raise ValueError(
                '%s seems to already be an ecoshard with algorithm %s and '
                'hash %s. Set `force=True` to overwrite.' % (
                    base_path, match_result.group(1), match_result.group(2)))
        else:
            LOGGER.warning(
                '%s is already in ecoshard format, but overriding because '
                '`force` is True.', base_path)
            prefix = match_result.group(1)

    LOGGER.debug('calculating hash for %s', base_path)
    hash_val = calculate_hash(base_path, hash_algorithm)

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


def build_overviews(
        base_raster_path, target_token_path=None,
        interpolation_method='near'):
    """Build embedded overviews on raster.

    Parameters:
        base_raster_path (str): base raster file, must be a GDAL writable
            raster type.
        target_token_path (str): if not None, this file is created and written
            with a timestamp when overviews are successfully completed. This
            file is useful for a library like `taskgraph` that needs to see
            a file to know if an operation is successful.
        interpolation_method (str): one of 'average', 'average_magphase',
            'bilinear', 'cubic', 'cubicspline', 'gauss', 'lanczos', 'mode',
            'near', or 'none'.

    Returns:
        None.

    """
    raster = gdal.OpenEx(base_raster_path, gdal.OF_RASTER | gdal.GA_Update)
    if not raster:
        raise ValueError(
            'could not open %s as a GDAL raster' % base_raster_path)
    band = raster.GetRasterBand(1)
    overview_count = band.GetOverviewCount()
    if overview_count != 0:
        LOGGER.warn(
            '%d overviews already exist for %s still creating anyway',
            overview_count, base_raster_path)
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
        'average', overview_levels, callback=_make_logger_callback(
            'build overview for ' + os.path.basename(base_raster_path) +
            '%.2f/1.0 complete'))


def validate(base_ecoshard_path):
    """Validate ecoshard path, through its filename.

    If `base_ecoshard_path` matches an EcoShard pattern, and the hash matches
    the actual hash, return True. Otherwise raise a ValueError.

    Parameters:
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

    Parameters:
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


def _make_logger_callback(message):
    """Build a timed logger callback that prints ``message`` replaced.

    Parameters:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        try:
            current_time = time.time()
            if ((current_time - logger_callback.last_time) > 5.0 or
                    (df_complete == 1.0 and
                     logger_callback.total_time >= 5.0)):
                # In some multiprocess applications I was encountering a
                # ``p_progress_arg`` of None. This is unexpected and I suspect
                # was an issue for some kind of GDAL race condition. So I'm
                # guarding against it here and reporting an appropriate log
                # if it occurs.
                if p_progress_arg:
                    LOGGER.info(message, df_complete * 100, p_progress_arg[0])
                else:
                    LOGGER.info(
                        'p_progress_arg is None df_complete: %s, message: %s',
                        df_complete, message)
                logger_callback.last_time = current_time
                logger_callback.total_time += current_time
        except AttributeError:
            logger_callback.last_time = time.time()
            logger_callback.total_time = 0.0

    return logger_callback


def compress_raster(
        base_raster_path, target_compressed_path, compression_algorithm='LZW',
        compression_predictor=None):
    """Compress base raster to target.

    Parameters:
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
    compressed_raster = gtiff_driver.CreateCopy(
        target_compressed_path, base_raster, options=(
            'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=%s' % compression_algorithm,
            'BLOCKXSIZE=256', 'BLOCKYSIZE=256',
            'PREDICTOR=%s' % str(compression_predictor)))
    del compressed_raster


def download_url(url, target_path, skip_if_target_exists=False):
    """Download `url` to `target_path`.

    Parameters:
        url (str): url to a file that can be retrieved.
        target_path (str): desired target location of the file fetched from
            `url`. Note any directory structure referenced in `target_path`
            must already exist.
        skip_if_target_exists (bool): If True and `target_path` is a file,
            this function logs that status and does not download a new copy.

    Returns:
        None.

    """
    if skip_if_target_exists and os.path.exists(target_path):
        LOGGER.info(
            'target exists and `skip_if_target_exists` is True: (%s->%s)',
            url, target_path)
        return
    with open(target_path, 'wb') as target_file:
        last_time = time.time()
        with urllib.request.urlopen(url) as url_stream:
            meta = url_stream.info()
            file_size = int(meta["Content-Length"])
            LOGGER.info(
                "Downloading: %s Bytes: %s" % (target_path, file_size))

            downloaded_so_far = 0
            block_size = 2**20
            last_download_size = 0
            while True:
                data_buffer = url_stream.read(block_size)
                if not data_buffer:
                    break
                downloaded_so_far += len(data_buffer)
                target_file.write(data_buffer)
                # display download so far in kilobytes
                current_time = time.time()
                if current_time - last_time > 5.0:  # log every 5 seconds
                    status = r"%10dMb  [%3.2f%%] [%4.2fMb/sec]" % (
                        downloaded_so_far/(2**20),
                        downloaded_so_far * 100./file_size,
                        ((downloaded_so_far-last_download_size)/2**20)/(
                            current_time-last_time))
                    LOGGER.info(status)
                    last_time = time.time()
                    last_download_size = downloaded_so_far

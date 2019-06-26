"""Main ecoshard module."""
import datetime
import shutil
import hashlib
import re
import os
import logging

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
    pass


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

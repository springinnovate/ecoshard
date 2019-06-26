"""Entry point for hashandrename."""
import shutil
import hashlib
import argparse
import os
import sys
import logging

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hash and rename files.')
    parser.add_argument(
        'filepath', nargs='+', help='Files to hash and rename.')
    parser.add_argument(
        '--hashalg', nargs=1, default='md5',
        help='hash algorithm in hashlib.algorithms_available')
    parser.add_argument(
        '--rename', action='store_true',
        help='rename the files, rather than copy new ones.')
    args = parser.parse_args()
    for file_path in args.filepath:
        hash_val = calculate_hash(file_path, args.hashalg)
        path_pre, path_suf = os.path.splitext(file_path)
        target_path = '%s_%s_%s%s' % (
            path_pre, args.hashalg, hash_val, path_suf)
        if os.path.exists(target_path):
            LOGGER.warn('%s already exists, not replacing')
            continue
        if args.rename:
            LOGGER.info('renaming %s to %s', file_path, target_path)
            os.rename(file_path, target_path)
        else:
            LOGGER.info('copying %s to %s', file_path, target_path)
            shutil.copyfile(file_path, target_path)

"""Entry point for ecoshard."""
import shutil
import argparse
import os
import sys
import logging
import glob

import ecoshard

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ecoshard files.')
    parser.add_argument(
        'filepath', nargs='+', help='Files/patterns to ecoshard.')
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

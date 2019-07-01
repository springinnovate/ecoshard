"""Entry point for ecoshard."""
import os
import argparse
import sys
import logging
import glob

import ecoshard

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard').setLevel(logging.WARN)
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ecoshard files.')
    parser.add_argument(
        'filepath', nargs='+', help='Files/patterns to ecoshard.')
    parser.add_argument(
        '--hashalg', nargs=1, default='md5',
        help='hash algorithm in hashlib.algorithms_available')

    parser.add_argument(
        '--compress', action='store_true', help='compress the raster files.')
    parser.add_argument(
        '--buildoverviews', action='store_true',
        help='build overviews on the raster files.')
    parser.add_argument(
        '--rename', action='store_true', help=(
            'if not compressing and hashing only, rename files rather than '
            'copy new ones.'))
    parser.add_argument(
        '--interpolation_method', help=(
            'used when building overviews, can be one of '
            '"average|near|mode|min|max|..."'), default='near')
    parser.add_argument(
        '--validate', action='store_true', help=(
            'validate the ecoshard rather than hash it'))
    parser.add_argument(
        '--hash_file', action='store_true', help=(
            'hash the file and and rename/copy depending on if --rename is '
            'set'))
    parser.add_argument(
        '--force', action='store_true', help=(
            'force an ecoshard to rehash even if it is already an ecoshard.'))

    args = parser.parse_args()
    for glob_pattern in args.filepath:
        for file_path in glob.glob(glob_pattern):
            working_file_path = file_path
            LOGGER.info('processing %s', file_path)
            if args.compress:
                prefix, suffix = os.path.splitext(file_path)
                compressed_filename = '%s_compressed%s' % (prefix, suffix)
                ecoshard.compress_raster(
                    file_path, compressed_filename,
                    compression_algorithm='DEFLATE')
                working_file_path = compressed_filename

            if args.buildoverviews:
                overview_token_path = '%s.OVERVIEWCOMPLETE' % (
                    working_file_path)
                ecoshard.build_overviews(
                    working_file_path, target_token_path=overview_token_path,
                    interpolation_method=args.interpolation_method)

            if args.validate:
                try:
                    is_valid = ecoshard.validate(working_file_path)
                    if is_valid:
                        LOGGER.info('VALID ECOSHARD: %s', working_file_path)
                    else:
                        LOGGER.error(
                            'got a False, but no ValueError on validate? '
                            'that is not impobipible?')
                except ValueError:
                    LOGGER.error('INVALID ECOSHARD: %s', working_file_path)
            elif args.hash_file:
                hash_token_path = '%s.ECOSHARDCOMPLETE' % (
                    working_file_path)
                ecoshard.hash_file(
                    working_file_path, target_token_path=hash_token_path,
                    rename=args.rename, hash_algorithm=args.hashalg,
                    force=args.force)

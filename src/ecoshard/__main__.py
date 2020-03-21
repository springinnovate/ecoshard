"""Entry point for ecoshard."""
import argparse
import glob
import hashlib
import logging
import os
import sys

import ecoshard

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=('%(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)


if __name__ == '__main__':
    return_code = 0
    parser = argparse.ArgumentParser(description='Ecoshard files.')
    parser.add_argument(
        'filepath', nargs='+', help='Files/patterns to ecoshard.',
        default=None)
    parser.add_argument(
        '--version', action='version', version='ecoshard version ' +
        ecoshard.__version__)
    parser.add_argument(
        '--hashalg', nargs=1, default='md5',
        help='Choose one of: "%s"' % '|'.join(hashlib.algorithms_available))
    parser.add_argument(
        '--compress', action='store_true', help='Compress the raster files.')
    parser.add_argument(
        '--buildoverviews', action='store_true',
        help='Build overviews on the raster files.')
    parser.add_argument(
        '--rename', action='store_true', help=(
            'If not compressing and hashing only, rename files rather than '
            'copy new ones.'))
    parser.add_argument(
        '--interpolation_method', help=(
            'Used when building overviews, can be one of '
            '"average|near|mode|min|max".'), default='near')
    parser.add_argument(
        '--validate', action='store_true', help=(
            'Validate the ecoshard rather than hash it. Returns non-zero '
            'exit code if failed.'))
    parser.add_argument(
        '--hash_file', action='store_true', help=(
            'Hash the file and and rename/copy depending on if --rename is '
            'set.'))
    parser.add_argument(
        '--force', action='store_true', help=(
            'force an ecoshard hash if the filename looks like an ecoshard. '
            'The new hash will be appended to the filename.'))
    parser.add_argument(
        '--reduce_factor', help=(
            "Reduce size by [factor] to with [method] to [target]. "
            "[method] must be one of 'max', 'min', 'sum', 'average', 'mode'"),
        nargs=3)

    args = parser.parse_args()
    for glob_pattern in args.filepath:
        for file_path in glob.glob(glob_pattern):
            working_file_path = file_path
            LOGGER.info('processing %s', file_path)

            if args.reduce_factor:
                method = args.reduce_factor[1]
                valid_methods = ["max", "min", "sum", "average", "mode"]
                if method not in valid_methods:
                    LOGGER.error(
                        '--reduce_method must be one of %s' % valid_methods)
                    sys.exit(-1)
                ecoshard.convolve_layer(
                    file_path, int(args.reduce_factor[0]),
                    args.reduce_factor[1],
                    args.reduce_factor[2])
                continue

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
                    return_code = -1
            elif args.hash_file:
                hash_token_path = '%s.ECOSHARDCOMPLETE' % (
                    working_file_path)
                ecoshard.hash_file(
                    working_file_path, target_token_path=hash_token_path,
                    rename=args.rename, hash_algorithm=args.hashalg,
                    force=args.force)
    sys.exit(return_code)
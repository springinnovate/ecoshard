"""Entry point for ecoshard."""
import argparse
import configparser
import glob
import hashlib
import logging
import os
import sys
import threading

import ecoshard

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard').setLevel(logging.DEBUG)
LOGGER = logging.getLogger(__name__)

CONFIG_PATH = os.path.expanduser(os.path.join('~', 'ecoshard.ini'))


def main():
    """Execute main and return valid return code "0 if fine"."""
    return_code = 0
    parser = argparse.ArgumentParser(description='Ecoshard files.')
    subparsers = parser.add_subparsers(dest='command')

    fetch_subparser = subparsers.add_parser('fetch', help='fetch ecoshards')
    fetch_subparser.add_argument(
        '--host_port', required=True, help='host:port of the ecoshard server')
    fetch_subparser.add_argument(
        '--api_key', required=True, help='api key to access ecoshard server.')
    fetch_subparser.add_argument(
        '--catalog', required=True, help='catalog to fetch from.')
    fetch_subparser.add_argument(
        '--asset_id', required=True, help='asset id to fetch')
    fetch_subparser.add_argument(
        '--asset_type', required=True, help='one of "WMS_preview|uri"')

    search_subparser = subparsers.add_parser(
        'search', help='search ecoshards')
    search_subparser.add_argument(
        '--host_port', required=True, help='host:port of the ecoshard server')
    search_subparser.add_argument(
        '--api_key', required=True, help='api key to access ecoshard server.')
    search_subparser.add_argument(
        '--bounding_box', type=float, nargs=4,
        help='list of xmin ymin xmax ymax')
    search_subparser.add_argument(
        '--description', help='search descriptions for this substring')
    search_subparser.add_argument(
        '--asset_id', help='to search ids for substring of this')
    search_subparser.add_argument(
        '--datetime', help=(
            'either range or open range, ex:\n'
            '2020-04-20 04:20:17.866142/2020-04-20 19:49:17.866142, '
            '../2020-04-20 19:49:17.866142, or '
            '2020-04-20 04:20:17.866142/.., '))
    search_subparser.add_argument(
        '--catalog_list', help=(
            'comma separated list of catalogs to limit search through, ex: '
            'salo,natcap,nci'))

    publish_subparser = subparsers.add_parser(
        'publish', help='publish ecoshards')
    publish_subparser.add_argument(
        '--host_port', required=True, help='host:port of the ecoshard server')
    publish_subparser.add_argument(
        '--path_to_file', required=True, help='path to file to publish')
    publish_subparser.add_argument(
        '--gs_root', help=(
            'root gs:// path to upload to the STAC `uri` parameter'))
    publish_subparser.add_argument(
        '--catalog', default='public', help='catalog to publish asset to')
    publish_subparser.add_argument(
        '--api_key', help='api key to access ecoshard server.')
    publish_subparser.add_argument(
        '--description', default='no description provided',
        help='description of the asset')
    publish_subparser.add_argument(
        '--mediatype', default='GeoTIFF',
        help='Currently only GeoTIFF is supported.')
    publish_subparser.add_argument(
        '--force', action='store_true', help=(
            'force a raster to be republished.'))

    process_subparser = subparsers.add_parser(
        'process', help='process files/ecoshards')
    process_subparser.add_argument(
        'filepath', nargs='+', help='Files/patterns to ecoshard.',
        default=None)
    process_subparser.add_argument(
        '--version', action='version', version='ecoshard version ' +
        ecoshard.__version__)
    process_subparser.add_argument(
        '--hashalg', nargs=1, default='md5',
        help='Choose one of: "%s"' % '|'.join(hashlib.algorithms_available))
    process_subparser.add_argument(
        '--compress', action='store_true', help='Compress the raster files.')
    process_subparser.add_argument(
        '--buildoverviews', action='store_true',
        help='Build overviews on the raster files.')
    process_subparser.add_argument(
        '--rename', action='store_true', help=(
            'If not compressing and hashing only, rename files rather than '
            'copy new ones.'))
    process_subparser.add_argument(
        '--interpolation_method', help=(
            'Used when building overviews, can be one of '
            '"average|near|mode|min|max".'), default='near')
    process_subparser.add_argument(
        '--validate', action='store_true', help=(
            'Validate the ecoshard rather than hash it. Returns non-zero '
            'exit code if failed.'))
    process_subparser.add_argument(
        '--hash_file', action='store_true', help=(
            'Hash the file and and rename/copy depending on if --rename is '
            'set.'))
    process_subparser.add_argument(
        '--force', action='store_true', help=(
            'force an ecoshard hash if the filename looks like an ecoshard. '
            'The new hash will be appended to the filename.'))
    process_subparser.add_argument(
        '--reduce_factor', help=(
            "Reduce size by [factor] to with [method] to [target]. "
            "[method] must be one of 'max', 'min', 'sum', 'average', 'mode'"),
        nargs=3)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        config.read(CONFIG_PATH)

    if args.command == 'fetch':
        ecoshard.fetch(
            args.host_port, args.api_key, args.catalog, args.asset_id,
            args.asset_type)
        return 0

    if args.command == 'publish':
        # publish an ecoshard
        if 'gs_root' in config['publish']:
            gs_root = config['publish']['gs_root']
        if args.gs_root:
            # prefer locally defined gs root
            gs_root = args.gs_root

        publish_thread_list = []
        for file_path in glob.glob(args.path_to_file):
            LOGGER.info(f'calculating hash for {file_path}')
            hash_val = ecoshard.calculate_hash(file_path, 'md5')
            LOGGER.info(f'hash val for {file_path}: {hash_val}')
            basename, ext = os.path.splitext(
                os.path.basename(file_path))
            target_gs_path = f'{gs_root}/{basename}_md5_{hash_val}{ext}'
            LOGGER.info(f'copying {file_path} to {target_gs_path}')
            ecoshard.copy_to_bucket(file_path, target_gs_path)
            asset_id = f'{basename}_md5_{hash_val}'

            if 'api_key' in config['publish']:
                api_key = config['publish']['api_key']

            # prefer locally defined api key if present
            if args.api_key:
                api_key = args.api_key

            publish_thread = threading.Thread(
                target=ecoshard.publish,
                args=(
                    target_gs_path, args.host_port, api_key, asset_id,
                    args.catalog, args.mediatype, args.description,
                    args.force))
            publish_thread.start()
            publish_thread_list.append((publish_thread, asset_id))

        fetch_payload_list = []
        for publish_thread, asset_id in publish_thread_list:
            publish_thread.join()

            fetch_payload = ecoshard.fetch(
                args.host_port, api_key, args.catalog, asset_id,
                'WMS_preview')
            fetch_payload_list.append((fetch_payload, asset_id))

        for fetch_payload, asset_id in fetch_payload_list:
            LOGGER.info(f"fetch url:\n{fetch_payload['link']}")
        return 0

    if args.command == 'search':
        # search for ecoshards
        ecoshard.search(
            args.host_port, args.api_key, args.bounding_box, args.description,
            args.datetime, args.asset_id, args.catalog_list)
        return 0

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
    return return_code


if __name__ == '__main__':
    sys.exit(main())

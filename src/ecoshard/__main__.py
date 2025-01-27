"""Entry point for ecoshard."""
import argparse
import configparser
import glob
import hashlib
import logging
import multiprocessing
import os
import sys

from ecoshard import taskgraph
from osgeo import gdal
import ecoshard

gdal.SetCacheMax(2**26)

LOGGER = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(processName)s %(levelname)s '
        '%(name)s [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
logging.getLogger('ecoshard').setLevel(logging.DEBUG)
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
LOGGER = logging.getLogger(__name__)

CONFIG_PATH = os.path.expanduser(os.path.join('~', 'ecoshard.ini'))
HASH_LEN_DEFAULT = 6


def main():
    """Execute main and return valid return code "0 if fine"."""
    return_code = 0
    parser = argparse.ArgumentParser(description='Ecoshard files.')
    subparsers = parser.add_subparsers(dest='command')

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
        '--hash_length', type=int, default=HASH_LEN_DEFAULT, help=(
            'Limit the length of the hash to this many characters. '
            f'Default is {HASH_LEN_DEFAULT}.'))

    process_subparser.add_argument(
        '--force', action='store_true', help=(
            'force an ecoshard hash if the filename looks like an ecoshard. '
            'The new hash will be appended to the filename.'))
    process_subparser.add_argument(
        '--reduce_factor', help=(
            "Reduce size by [factor] to with [method] to [target]. "
            "[method] must be one of 'max', 'min', 'sum', 'average', 'mode'"),
        nargs=3)
    process_subparser.add_argument(
        '--stitch', action='store_true', help='first argument is target, second is wildcard')
    process_subparser.add_argument(
        '--ndv', type=float, help=(
            'Set the nodata value to this value and replaces any non-finite '
            'values in the raster to this value. If a nodata value is '
            'already defined, this will raise an exception unless the '
            '--force flag is passed in which case any current nodata values '
            'will be replaced by this value'))

    process_subparser.add_argument(
        '--strip_hash', type=str, help=(
            "Removes the hash value from a filename of the form "
            "[prefix]_[REMOVE_HASH].ext choosing one of the hash algorithms "
            "from above"))

    process_subparser.add_argument(
        '--cog', action='store_true', help=(
            'Convert to Cloud Optimized GeoTiff, overrides any other flags.'))

    process_subparser.add_argument(
        '--transient_run', action='store_true', help=(
            "Do not use unless you know what you are doing."))

    args = parser.parse_args()

    config = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        config.read(CONFIG_PATH)

    if args.stitch:
        target_path = args.filepath[0]
        file_list = [
            file_path
            for glob_pattern in args.filepath[1:]
            for file_path in glob.glob(glob_pattern)]
        ecoshard.stitch_rasters(file_list, args.filepath[0])
        return

    file_list = [
        file_path
        for glob_pattern in args.filepath
        for file_path in glob.glob(glob_pattern)]


    n_workers = min(multiprocessing.cpu_count(), len(file_list))
    taskgraph_dir = '_ecoshard_taskgraph_dir_ok_to_delete'
    os.makedirs(taskgraph_dir, exist_ok=True)
    task_graph = taskgraph.TaskGraph(taskgraph_dir, n_workers, 15.0)

    result_list = []
    error_messages = []
    for file_path in file_list:
        if args.transient_run:
            LOGGER.info('TRANSIENT RUN')
        result = task_graph.add_task(
            func=ecoshard.process_worker,
            args=(file_path, args,),
            transient_run=args.transient_run,
            task_name=f'processing {file_path}')
        result_list.append(result)

    for result in result_list:
        # ensure no bad result
        try:
            result.join()
        except Exception as e:
            error_messages.append(str(e))

    for message in error_messages:
        LOGGER.error(message)
        return_code = -1

    task_graph.join()
    task_graph.close()
    return return_code


if __name__ == '__main__':
    sys.exit(main())

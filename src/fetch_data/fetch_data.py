"""See `python scriptname.py --help"""
import configparser
import csv
import datetime
import logging
import os

from ecoshard import geoprocessing
from sqlalchemy import create_engine
from sqlalchemy import Integer
from sqlalchemy import Text
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import and_
import boto3
import botocore.exceptions
import sqlalchemy

LOGGER = logging.getLogger(__name__)

GLOBAL_INI_PATH = os.path.join(os.path.dirname(__file__), 'defaults.ini')
GLOBAL_CONFIG = None
DB_ENGINE = None
DATA_CLASS_MAP = {}


def _initalize():
    """Initalize ini location and database file."""
    global DB_ENGINE
    global GLOBAL_CONFIG
    global DATA_CACHE_DIR
    global DATA_CLASS_MAP
    GLOBAL_CONFIG = configparser.ConfigParser(allow_no_value=False)
    GLOBAL_CONFIG.read(GLOBAL_INI_PATH)

    config_found = False
    possible_path_list = [
        os.path.expanduser(path) for path in
        GLOBAL_CONFIG['defaults']['default_user_ini_path'].split(',')]
    for fetch_dirpath in possible_path_list:
        if os.path.exists(fetch_dirpath):
            GLOBAL_CONFIG.read(fetch_dirpath)
            config_found = True
            break
    if not config_found:
        message = (
            f'Could not find a custom file registry config file at any of the '
            f'expected locations: {possible_path_list}\n'
            f'To resolve this error, create a file at one of the expected '
            f'locations and populate it with this template that can be '
            f'customized:\n'
            f'[defaults]\nCACHE_DIR='
            f'{os.path.dirname(possible_path_list[0])}/cache_dir\n'
            f'\nCREDENTIALS_DIR={os.path.dirname(possible_path_list[0])}/access_keys')
        raise RuntimeError(message)
    elif any(key_val not in GLOBAL_CONFIG['defaults']
             for key_val in ['cache_dir', 'credentials_dir']):
        message = (
            'expected a fields CACHE_DIR=PATH and CREDENTIALS_DIR=PATH but '
            'one or both are not found, please update these lines to '
            f'{fetch_dirpath}')
        raise RuntimeError(message)

    DATA_CACHE_DIR = GLOBAL_CONFIG['defaults']['cache_dir']

    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    for section_id in GLOBAL_CONFIG.sections():
        if section_id == 'defaults':
            continue
        DatasetBaseClass = sqlalchemy_base_factory(
            SqlalchemyBase, section_id,
            GLOBAL_CONFIG[section_id]['expected_args'].split(','))
        DATA_CLASS_MAP[section_id] = DatasetBaseClass

    # create the table if it doesn't exist
    db_path = os.path.join(
        DATA_CACHE_DIR, 'local_file_registry.sqlite')
    DB_ENGINE = create_engine(f"sqlite:///{db_path}", echo=False)
    SqlalchemyBase.metadata.create_all(DB_ENGINE)


# Need this because we can't subclass it directly
class SqlalchemyBase(DeclarativeBase):
    pass


def sqlalchemy_base_factory(BaseClass, dataset_id, arg_list):
    class NewClass(BaseClass):
        __tablename__ = f'{dataset_id}_file_to_location'
        id_val = mapped_column(Integer, primary_key=True)
        file_path = mapped_column(Text, index=True)
        remote_path = mapped_column(Text, index=True)

        def __repr__(self) -> str:
            return (
                f'{dataset_id}(id_val={self.id_val!r},\n'
                f'file_path={self.file_path!r},\n' +
                ',\n'.join([
                    f'{arg}={getattr(self, arg)!r}']))

    NewClass.__name__ = f'{dataset_id}'
    for arg in arg_list:
        setattr(NewClass, arg, mapped_column(Text, index=True))
    return NewClass


# # Table to store local file locations
# class File(Base):
#     __tablename__ = "file_to_location"
#     id_val = mapped_column(Integer, primary_key=True)
#     dataset_id = mapped_column(Text, index=True)
#     variable_id = mapped_column(Text, index=True)
#     date_str = mapped_column(Text, index=True)
#     file_path = mapped_column(Text, index=True)
#     remote_path = mapped_column(Text, index=True)

#     def __repr__(self) -> str:
#         return (
#             f'File(id_val={self.id_val!r}, '
#             f'dataset_id={self.dataset_id!r}, '
#             f'variable_id={self.variable_id!r}, '
#             f'date_str={self.date_str!r}, '
#             f'file_path={self.file_path!r}')


def _construct_filepath(dataset_id, args):
    """Form consistent local cache path for given file parameters.

    Returns: local path, bucket path to file if it exists on the file system.
    """
    file_format = GLOBAL_CONFIG[dataset_id]['file_format']
    bucket_path = file_format.format(**args)
    target_path = os.path.join(DATA_CACHE_DIR, dataset_id, bucket_path)
    return target_path, bucket_path


def _create_s3_bucket_obj(dataset_id):
    """Create S3 object given dataset_id."""
    credentials_dir = GLOBAL_CONFIG['defaults']['credentials_dir']
    access_key_path = os.path.join(
        credentials_dir,
        GLOBAL_CONFIG[dataset_id]['access_key'])
    if not os.path.exists(access_key_path):
        raise ValueError(
            f'expected a keyfile to access the S3 bucket at {access_key_path} '
            'but not found')
    reader = csv.DictReader(open(access_key_path))
    bucket_access_dict = next(reader)
    s3 = boto3.resource(
        's3',
        endpoint_url=GLOBAL_CONFIG[dataset_id]['base_uri'],
        aws_access_key_id=bucket_access_dict['Access Key Id'],
        aws_secret_access_key=bucket_access_dict['Secret Access Key'],
    )
    dataset_bucket = s3.Bucket(GLOBAL_CONFIG[dataset_id]['bucket_id'])
    return dataset_bucket


def fetch_remote_dataset(dataset_id):
    """Fetch a copy of the remote file database."""
    local_config = GLOBAL_CONFIG[dataset_id]
    database_path = os.path.join(
        local_config['database_dir'], local_config['database'])
    os.makedirs(os.path.dirname(database_path))


def file_exists(dataset_id, variable_id, date_str):
    """Return true if file exists.

    Args:
        dataset_id (str): dataset defined by config
        variable_id (str): variable id that's consistent with dataset
        date_str (str): date to query that's consistent with the dataset

    Return: true if file exists in bucket."""
    local_path, bucket_path = _construct_filepath(
        dataset_id, variable_id, date_str)
    LOGGER.info(f'checking if {dataset_id}/{bucket_path} exists')
    if os.path.exists(local_path):
        return True
    dataset_bucket = _create_s3_bucket_obj(dataset_id)
    objects = list(dataset_bucket.objects.filter(Prefix=bucket_path))
    return len(objects) == 1


def fetch_and_clip(
        dataset_id, variable_id, date_str, pixel_size, clip_vector_path,
        target_raster_path, all_touched=True, target_mask_value=None):
    """Download and clip a file to a vector coverage.

    Args:
        dataset_id (str): dataset defined by config
        variable_id (str): variable id that's consistent with dataset
        date_str (str): date to query that's consistent with the dataset
        pixel_size (tuple): target clip pixel size in units of the vector
        clip_vector_path (str): path to vector to clip against
        all_touched (bool): if True, clip keeps any pixels whose edge is
            contained in the vector
        target_mask_value (numeric): if not None, set the value of pixels
            outside the vector coverage to this value.
    """
    local_raster_path = fetch_file(dataset_id, variable_id, date_str)
    vector_info = geoprocessing.get_vector_info(clip_vector_path)
    geoprocessing.warp_raster(
        local_raster_path, pixel_size, target_raster_path, 'near',
        target_projection_wkt=vector_info['projection_wkt'],
        target_bb=vector_info['bounding_box'],
        vector_mask_options={
            'mask_vector_path': clip_vector_path,
            'all_touched': all_touched,
            'target_mask_value': target_mask_value})


def put_file(dataset_id, variable_id, date_str, local_path):
    """Put a local file to remote bucket.

    Args:
        dataset_id (str): dataset defined by config
        variable_id (str): variable id that's consistent with dataset
        date_str (str): date to query that's consistent with the dataset

    Returns:
        (str) path to local downloaded file.
    """
    pass


def _download_file_from_s3(
        bucket_id, bucket_path, target_path, overwrite_existing=False):
    """Download file from s3 to local.

    Args:
        bucket_id (str): name of bucket
        bucket_path (str): path of file on bucket
        target_path (str): desired local target path.
        overwrite_existing (bool): If true, overwrites existing files, if
            false, raises a ValueError if file already exists.

    Returns:
        None
    """
    dataset_bucket = _create_s3_bucket_obj(bucket_id)
    if not os.path.exists(target_path):
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        LOGGER.info(f'downloading {bucket_path}')
        try:
            dataset_bucket.download_file(bucket_path, target_path)
        except botocore.exceptions.ClientError:
            message = (
                f'file not found for {bucket_path}')
            LOGGER.error(message)
            raise FileNotFoundError(message)
    else:
        LOGGER.warning(
            f'{target_path} exists but no entry in database, either delete '
            f'{target_path} and restart, or figure out what\'s going on')


def fetch_file(dataset_id, args):
    """Fetch a file from remote data store to local target path.

    Args:
        dataset_id (str): dataset defined by config
        args (dict): string/value pairs that are used to match and fit the
            variables required to identify a datasource for download

    Returns:
        (str) path to local downloaded file.
    """
    SqlalchamyTable = DATA_CLASS_MAP[dataset_id]
    with Session(DB_ENGINE) as session:
        query_list = [
            getattr(SqlalchamyTable, key) == value
            for key, value in args.items()]

        stmt = sqlalchemy.select(SqlalchamyTable).where(and_(*query_list))
        result = session.execute(stmt).first()
    if result is not None and os.path.exists(result[0].file_path):
        local_path = result[0].file_path
        LOGGER.info(f'{local_path} is locally cached!')
        return local_path

    target_path, bucket_path = _construct_filepath(
        dataset_id, args)

    _download_file_from_s3(dataset_id, bucket_path, target_path)

    with Session(DB_ENGINE) as session:
        file_entry = SqlalchamyTable(file_path=target_path)
        for key, value in args.items():
            setattr(file_entry, key, value)
        # file_entry = File(
        #     dataset_id=dataset_id,
        #     variable_id=variable_id,
        #     date_str=formatted_date,
        #     file_path=target_path)
        session.add(file_entry)
        session.commit()
    LOGGER.info(f'result at {target_path}')
    return target_path


_initalize()

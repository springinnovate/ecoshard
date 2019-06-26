"""Main ecoshard module."""


def hash_file(
        base_path, target_token_path=None, target_dir=None, rename=False,
        hash_algorithm='md5', force=False):
    """EcoShard file by hashing it and appending hash to filename.

    An EcoShard is the hashing of a file and the rename to the following
    format: [base name]_[hashalg]_[hash][base extension]. If the base path
    already is in this format a ValueError is raised unless `force` is True.

    Parameters:
        base_path (str): path to base file.
        target_token_path (str): if present, this file is created and written
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
    pass


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
    pass

"""geoprocessing __init__ module."""
import logging
import types

from . import geoprocessing
from .geoprocessing import *
from .geoprocessing import ReclassificationMissingValuesError
from .geoprocessing import _create_latitude_m2_area_column
from .geoprocessing_core import calculate_slope
from .geoprocessing_core import raster_band_percentile
from .geoprocessing_core import greedy_pixel_pick_by_area
from .geoprocessing_core import greedy_pixel_pick_by_area_v2

__all__ = (
    'calculate_slope', 'raster_band_percentile',
    'ReclassificationMissingValuesError',
    'greedy_pixel_pick_by_area',
    'greedy_pixel_pick_by_area_v2',)
for attrname in dir(geoprocessing):
    if attrname.startswith('_'):
        continue
    attribute = getattr(geoprocessing, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())  # silence logging by default

# these are bit masks for the known geoprocessing types
UNKNOWN_TYPE = 0
RASTER_TYPE = 1
VECTOR_TYPE = 2

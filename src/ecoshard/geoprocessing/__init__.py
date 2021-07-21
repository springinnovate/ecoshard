"""geoprocessing __init__ module."""
import logging
import types

from . import geoprocessing
from .geoprocessing import *
from .geoprocessing import ReclassificationMissingValuesError
from .geoprocessing_core import calculate_slope
from .geoprocessing_core import raster_band_percentile

__all__ = (
    'calculate_slope', 'raster_band_percentile',
    'ReclassificationMissingValuesError')
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

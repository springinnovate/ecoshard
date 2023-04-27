"""
__init__ module imports all the fetch_data functions into this namespace.
"""
import sys
import types

from pkg_resources import get_distribution
from . import fetch_data

__all__ = ()
for attrname in dir(fetch_data):
    attribute = getattr(fetch_data, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)
    setattr(sys.modules['ecoshard.fetch_data'], attrname, attribute)

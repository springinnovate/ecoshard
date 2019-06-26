"""EcoShard: gis file hashing and database routines.

__init__ module imports all the ecoshard functions into this namespace.
"""
from __future__ import absolute_import

import types
import sys

from . import ecoshard

__all__ = tuple()
for attrname in dir(ecoshard):
    attribute = getattr(ecoshard, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)
        setattr(sys.modules['ecoshard'], attrname, attribute)

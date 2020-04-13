"""
__init__ module imports all the ecoshard functions into this namespace.
"""
import sys
import types

from pkg_resources import get_distribution
from . import ecoshard

__all__ = ()
for attrname in dir(ecoshard):
    attribute = getattr(ecoshard, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)
        setattr(sys.modules['ecoshard'], attrname, attribute)

__version__ = get_distribution(__name__).version

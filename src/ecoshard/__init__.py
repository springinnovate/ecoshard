"""
__init__ module imports all the ecoshard functions into this namespace.
"""
import sys
import types
from importlib.metadata import version

from . import ecoshard

__all__ = ()
for attrname in dir(ecoshard):
    attribute = getattr(ecoshard, attrname)
    if isinstance(attribute, types.FunctionType):
        __all__ += (attrname,)

        setattr(sys.modules[__name__], attrname, attribute)

__version__ = version(__name__)

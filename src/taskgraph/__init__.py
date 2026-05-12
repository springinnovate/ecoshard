"""TaskGraph init module."""
from importlib.metadata import version
from .Task import TaskGraph
from .Task import Task
from .Task import TaskRef
from .Task import _TASKGRAPH_DATABASE_FILENAME

__all__ = [
    'TaskGraph',
    'Task',
    'TaskRef',
    '_TASKGRAPH_DATABASE_FILENAME'
]

__version__ = version('ecoshard')

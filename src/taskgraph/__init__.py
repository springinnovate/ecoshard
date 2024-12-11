"""TaskGraph init module."""
from importlib.metadata import version
from .Task import TaskGraph
from .Task import Task
from .Task import _TASKGRAPH_DATABASE_FILENAME
from .Task import NonDaemonicPool
from .Task import NonDaemonicProcessPoolExecutor

__all__ = [
    'TaskGraph', 'Task',
    '_TASKGRAPH_DATABASE_FILENAME', 'NonDaemonicPool',
    'NonDaemonicProcessPoolExecutor']

__version__ = version('ecoshard')

"""ecoshard setup.py."""
import os

import numpy
from setuptools.extension import Extension
from setuptools import setup

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None


def _source_path(pyx_path):
    """Return Cython source, or generated C++ when Cython is unavailable."""
    cpp_path = os.path.splitext(pyx_path)[0] + '.cpp'
    if cythonize is None and os.path.exists(cpp_path):
        return cpp_path
    return pyx_path


EXTENSIONS = [
    Extension(
        name="ecoshard.geoprocessing.routing.routing",
        sources=[
            _source_path("src/ecoshard/geoprocessing/routing/routing.pyx")],
        include_dirs=[
            numpy.get_include(),
            'src/ecoshard/geoprocessing/routing'],
        language="c++",
    ),
    Extension(
        "ecoshard.geoprocessing.geoprocessing_core",
        sources=[
            _source_path('src/ecoshard/geoprocessing/geoprocessing_core.pyx')],
        include_dirs=[numpy.get_include()],
        language="c++"
    ),
]
if cythonize is not None:
    EXTENSIONS = cythonize(EXTENSIONS)

setup(
    packages=[
        'ecoshard',
        'ecoshard.geoprocessing',
        'ecoshard.geoprocessing.routing',
        'ecoshard.taskgraph',
        'ecoshard.geosharding',
        ],
    package_dir={
        'ecoshard': 'src/ecoshard',
        'ecoshard.taskgraph': 'src/taskgraph',
    },
    zip_safe=False,
    include_package_data=True,
    ext_modules=EXTENSIONS,
)

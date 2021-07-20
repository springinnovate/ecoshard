"""ecoshard setup.py."""
from setuptools import setup

LONG_DESCRIPTION = '%s\n\n%s' % (
    open('README.rst').read(),
    open('HISTORY.rst').read())

REQUIREMENTS = [
    x for x in open('requirements.txt').read().split('\n')
    if not x.startswith('#') and len(x) > 0]

setup(
    name='ecoshard',
    setup_requires=['setuptools_scm'],
    use_scm_version={'version_scheme': 'post-release',
                     'local_scheme': 'node-and-date'},
    install_requires=REQUIREMENTS,
    description='EcoShard GIS data',
    long_description=LONG_DESCRIPTION,
    maintainer='Rich Sharp',
    maintainer_email='richpsharp@gmail.com',
    url='https://github.com/therealspring/ecoshard',
    packages=['ecoshard'],
    package_dir={
        'ecoshard': 'src/ecoshard',
        'ecoshard.geoprocessing': 'src/ecoshard/geoprocessing',
    },
    zip_safe=False,
    include_package_data=True,
    license='BSD',
    keywords='computing reproduction',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: BSD License'
    ])

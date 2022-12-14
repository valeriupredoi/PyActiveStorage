import json
import os
import re
import sys
from pathlib import Path

from setuptools import Command, setup

PACKAGES = [
    'activestorage',
]

REQUIREMENTS = {
    # Installation script (this file) dependencies
    'setup': [
        'setuptools_scm',
    ],
    # Installation dependencies
    # Use with pip install . to install from source
    'install': [
        'dask',
        'h5py',  # needed by Kerchunk
        'kerchunk',
        'netcdf4',
        'pytest',
        'xarray',
        'zarr',
        # for testing
        'pytest-cov>=2.10.1',
        'pytest-xdist',
        'pytest-html!=2.1.0',
        'pytest-metadata>=1.5.1',
        # for documentation
        'autodocsumm',
        'sphinx>2',
        'sphinx_rtd_theme',
    ],
}


def discover_python_files(paths, ignore):
    """Discover Python files."""

    def _ignore(path):
        """Return True if `path` should be ignored, False otherwise."""
        return any(re.match(pattern, path) for pattern in ignore)

    for path in sorted(set(paths)):
        for root, _, files in os.walk(path):
            if _ignore(path):
                continue
            for filename in files:
                filename = os.path.join(root, filename)
                if (filename.lower().endswith('.py')
                        and not _ignore(filename)):
                    yield filename


setup(
    name='ActiveStorage',
    author="",
    description="",
    long_description="",
    long_description_content_type='text/markdown',
    url='',
    download_url='',
    license='',
    classifiers=[
        'Development Status :: 0 - Prototype',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Hydrology',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    packages=PACKAGES,
    # Include all version controlled files
    include_package_data=True,
    setup_requires=REQUIREMENTS['setup'],
    install_requires=REQUIREMENTS['install'],
    zip_safe=False,
)

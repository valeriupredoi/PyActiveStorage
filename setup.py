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
        'fsspec',
        'h5netcdf',
        'h5py',  # needed by Kerchunk
        'kerchunk>=0.2.4',  # issues with numcodecs in 0.2.2/3
        'netcdf4',
        'numcodecs>=0.12',  # github/issues/162
        'numpy!=1.24.3',  # severe masking bug
        'requests',
        's3fs>=2024.2.0',  # see environment.yml for pin reason
        # pin Zarr to use new FSStore instead of KVStore
        'zarr>=2.13.3', # github.com/zarr-developers/zarr-python/issues/1362
        # for testing
        'pytest',
        'pytest-cov>=2.10.1',
        'pytest-html!=2.1.0',
        'pytest-metadata>=1.5.1',
        'pytest-xdist',
        # for documentation
        # re-add when we deploy the docs
        # 'autodocsumm',
        # 'sphinx>=5',
        # 'sphinx_rtd_theme',
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
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
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

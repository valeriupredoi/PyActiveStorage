"""Create test data for heftier data tests."""
import os
import pytest

import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import s3fs

from activestorage.active import Active
from activestorage.config import *
from pyfive.core import InvalidHDF5File as InvalidHDF5Err

import utils


@pytest.fixture
def test_data_path():
    """Path to test data for CMOR fixes."""
    return Path(__file__).resolve().parent / 'test_data'


def create_hyb_pres_file_without_ap(dataset, short_name):
    """Create dataset without vertical auxiliary coordinate ``ap``."""
    dataset.createDimension('time', size=11)
    dataset.createDimension('lev', size=2)
    dataset.createDimension('lat', size=3)
    dataset.createDimension('lon', size=4)
    dataset.createDimension('bnds', size=2)

    # Dimensional variables
    dataset.createVariable('time', np.float64, dimensions=('time',))
    dataset.createVariable('lev', np.float64, dimensions=('lev',))
    dataset.createVariable('lev_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('lat', np.float64, dimensions=('lat',))
    dataset.createVariable('lon', np.float64, dimensions=('lon',))
    dataset.variables['time'][:] = range(11)
    dataset.variables['time'].standard_name = 'time'
    dataset.variables['time'].units = 'days since 1850-1-1'
    dataset.variables['lev'][:] = [1.0, 2.0]
    dataset.variables['lev'].bounds = 'lev_bnds'
    dataset.variables['lev'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev'].units = '1'
    dataset.variables['lev_bnds'][:] = [[0.5, 1.5], [1.5, 3.0]]
    dataset.variables['lev_bnds'].standard_name = (
        'atmosphere_hybrid_sigma_pressure_coordinate')
    dataset.variables['lev_bnds'].units = '1'
    dataset.variables['lat'][:] = [-30.0, 0.0, 30.0]
    dataset.variables['lat'].standard_name = 'latitude'
    dataset.variables['lat'].units = 'degrees_north'
    dataset.variables['lon'][:] = [30.0, 60.0, 90.0, 120.0]
    dataset.variables['lon'].standard_name = 'longitude'
    dataset.variables['lon'].units = 'degrees_east'

    # Coordinates for derivation of pressure coordinate
    dataset.createVariable('b', np.float64, dimensions=('lev',))
    dataset.createVariable('b_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('ps', np.float64,
                           dimensions=('time', 'lat', 'lon'))
    dataset.variables['b'][:] = [0.0, 1.0]
    dataset.variables['b_bnds'][:] = [[-1.0, 0.5], [0.5, 2.0]]
    dataset.variables['ps'][:] = np.arange(1 * 3 * 4).reshape(1, 3, 4)
    dataset.variables['ps'].standard_name = 'surface_air_pressure'
    dataset.variables['ps'].units = 'Pa'
    dataset.variables['ps'].additional_attribute = 'xyz'

    # Variable
    dataset.createVariable(short_name, np.float32,
                           dimensions=('time', 'lev', 'lat', 'lon'))
    dataset.variables[short_name][:] = np.full((1, 2, 3, 4), 22.0,
                                               dtype=np.float32)
    dataset.variables[short_name].standard_name = (
        'cloud_area_fraction_in_atmosphere_layer')
    dataset.variables[short_name].units = '%'


def create_hyb_pres_file_with_a(dataset, short_name):
    """Create netcdf file with issues in hybrid pressure coordinate."""
    create_hyb_pres_file_without_ap(dataset, short_name)
    dataset.createVariable('a', np.float64, dimensions=('lev',))
    dataset.createVariable('a_bnds', np.float64, dimensions=('lev', 'bnds'))
    dataset.createVariable('p0', np.float64, dimensions=())
    dataset.variables['a'][:] = [1.0, 2.0]
    dataset.variables['a_bnds'][:] = [[0.0, 1.5], [1.5, 3.0]]
    dataset.variables['p0'][:] = 1.0
    dataset.variables['p0'].units = 'Pa'
    dataset.variables['lev'].formula_terms = 'p0: p0 a: a b: b ps: ps'
    dataset.variables['lev_bnds'].formula_terms = (
        'p0: p0 a: a_bnds b: b_bnds ps: ps')


def save_cl_file_with_a(tmp_path):
    """Create netcdf file for ``cl`` with ``a`` coordinate."""
    save_path = tmp_path / 'common_cl_a.nc'
    nc_path = os.path.join(save_path)
    dataset = Dataset(nc_path, mode='w')
    create_hyb_pres_file_with_a(dataset, 'cl')
    dataset.close()
    uri = utils.write_to_storage(nc_path)
    if USE_S3:
        os.remove(save_path)

    return uri


def test_cl(tmp_path):
    ncfile = save_cl_file_with_a(tmp_path)
    active = Active(ncfile, "cl", utils.get_storage_type())
    active._version = 0
    d = active[4:5, 1:2]
    mean_result = np.mean(d)

    active = Active(ncfile, "cl", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[4:5, 1:2]
    print(result2, ncfile)
    # expect {'sum': array([[[[264.]]]], dtype=float32), 'n': array([[[[12]]]])}
    # check for typing and structure
    np.testing.assert_array_equal(result2["sum"], np.array([[[[264.]]]], dtype="float32"))
    np.testing.assert_array_equal(result2["n"], np.array([[[[12]]]]))
    # check for active
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])


def test_ps(tmp_path):
    ncfile = save_cl_file_with_a(tmp_path)
    active = Active(ncfile, "ps", utils.get_storage_type())
    active._version = 0
    d = active[4:5, 1:2]
    mean_result = np.mean(d)

    active = Active(ncfile, "ps", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[4:5, 1:2]
    print(result2, ncfile)
    # expect {'sum': array([[[22.]]]), 'n': array([[[4]]])}
    # check for typing and structure
    np.testing.assert_array_equal(result2["sum"], np.array([[[22.]]]))
    np.testing.assert_array_equal(result2["n"], np.array([[[4]]]))
    # check for active
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])


def test_native_emac_model_fails(test_data_path):
    """
    An example of netCDF file that doesn't work

    The actual issue  is with h5py - it can't read it (netCDF3 classic)

    h5py/_objects.pyx:54: in h5py._objects.with_phil.wrapper
        ???
    h5py/_objects.pyx:55: in h5py._objects.with_phil.wrapper
        ???
    _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

    >   ???
    E   OSError: Unable to open file (file signature not found)
    """
    ncfile = str(test_data_path / "emac.nc")
    uri = utils.write_to_storage(ncfile)

    if USE_S3:
        active = Active(uri, "aps_ave", utils.get_storage_type())
        with pytest.raises(InvalidHDF5Err):
            active[...]
    else:
        active = Active(uri, "aps_ave")
        active._version = 2
        active.method = "mean"
        active.components = True
        with pytest.raises(InvalidHDF5Err):
            result2 = active[4:5, 1:2]


def test_cesm2_native(test_data_path):
    """
    Test again a native model, this time around netCDF4 loadable with h5py
    Also, this has _FillValue and missing_value
    """
    ncfile = str(test_data_path / "cesm2_native.nc")
    uri = utils.write_to_storage(ncfile)
    active = Active(uri, "TREFHT", utils.get_storage_type())
    active._version = 0
    d = active[4:5, 1:2]
    mean_result = np.mean(d)

    active = Active(uri, "TREFHT", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[4:5, 1:2]
    print(result2, ncfile)
    # expect {'sum': array([[[2368.3232]]], dtype=float32), 'n': array([[[8]]])}
    # check for typing and structure
    np.testing.assert_allclose(result2["sum"], np.array([[[2368.3232]]], dtype="float32"), rtol=1e-6)
    np.testing.assert_array_equal(result2["n"], np.array([[[8]]]))
    # check for active
    np.testing.assert_allclose(mean_result, result2["sum"]/result2["n"], rtol=1e-6)


def test_daily_data(test_data_path):
    """
    Test again with a daily data file,
    """
    ncfile = str(test_data_path / "daily_data.nc")
    uri = utils.write_to_storage(ncfile)
    active = Active(uri, "ta", utils.get_storage_type())
    active._version = 0
    d = active[4:5, 1:2]
    mean_result = np.mean(d)

    active = Active(uri, "ta", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[4:5, 1:2]
    print(result2, ncfile)
    # expect {'sum': array([[[[1515.9822]]]], dtype=float32), 'n': array([[[8]]])}
    # check for typing and structure
    np.testing.assert_array_equal(result2["sum"], np.array([[[[1515.9822]]]], dtype="float32"))
    np.testing.assert_array_equal(result2["n"], np.array([[[[6]]]]))
    # check for active
    np.testing.assert_array_equal(mean_result, result2["sum"]/result2["n"])


def test_daily_data_masked(test_data_path):
    """
    Test again with a daily data file, with masking on
    """
    ncfile = str(test_data_path / "daily_data_masked.nc")
    uri = utils.write_to_storage(ncfile)
    active = Active(uri, "ta", utils.get_storage_type())
    active._version = 0
    d = active[:]
    d = np.ma.masked_where(d==999., d)
    mean_result = np.ma.mean(d)

    active = Active(uri, "ta", utils.get_storage_type())
    active._version = 2
    active.method = "mean"
    active.components = True
    result2 = active[:]
    print(result2, ncfile)
    # expect {'sum': array([[[[169632.5]]]], dtype=float32), 'n': 680}
    # check for typing and structure
    np.testing.assert_allclose(result2["sum"], np.array([[[[169632.5]]]], dtype="float32"), rtol=1e-6)
    np.testing.assert_array_equal(result2["n"], 680)
    # check for active
    np.testing.assert_allclose(mean_result, result2["sum"]/result2["n"], rtol=1e-6)

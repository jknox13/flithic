from __future__ import division, print_function, absolute_import

import mock
import pytest
import importlib
import numpy as np
import scipy.spatial.distance as sci_dist

from numpy.testing import assert_almost_equal, assert_raises

from flithic.distance import distance

@pytest.fixture
def u():
    return np.random.rand(100,1)

@pytest.fixture
def v():
    return np.random.rand(100,1)

@pytest.fixture
def v_nan(v):
    v_nan = v.copy()
    v_nan[:20] = np.nan
    return v_nan

@pytest.fixture
def test_bivariate(funcname, u, v, v_nan, *args, **kwargs):
    # get functions
    f_scipy = getattr(sci_dist, funcname)
    f_flithic = getattr(distance, funcname)

    # test need for function
    assert( np.isnan(f_scipy(u, v_nan, *args)).any() )

    # test funcs are equivalent
    scipy_ = f_scipy(u, v, *args, **kwargs)
    flithic_ = f_flithic.__wrapped__(u, v, *args, **kwargs)

    assert_almost_equal(scipy_, flithic_)

    # test nan version works like we think
    scipy_nan = f_flithic(u[20:], v[20:], *args, **kwargs)
    flithic_nan = f_flithic(u, v_nan, *args, **kwargs)

    assert_almost_equal( scipy_nan, flithic_nan)

@pytest.fixture
def check_bivariate(funcname, u, v, v_nan, **kwargs):
    # get functions
    f_scipy = getattr(sci_dist, funcname)

    assert_almost_equal( f_scipy(u,v), f_scipy(u, v_nan))

def get_vnan_bool(v_nan):
    v_nan_bool = v_nan.copy()
    v_nan_bool[~np.isnan(v_nan)] = v_nan_bool[~np.isnan(v_nan)] > 0.5

    return v_nan_bool


# =============================================================

# TODO
#def test_directed_hausdorff(u, v, v_nan):
    #test_bivariate("directed_hausdorff", u, v, v_nan)

# =============================================================

# TODO
#def test_minkowski(u, v, v_nan):
#

def test_sqeuclidean(u, v, v_nan):
    test_bivariate("sqeuclidean", u, v, v_nan)

def test_correlation(u, v, v_nan):
    test_bivariate("correlation", u, v, v_nan)

def test_cosine(u, v, v_nan):
    test_bivariate("cosine", u, v, v_nan)

def test_hamming(u, v, v_nan):
    check_bivariate("hamming", u, v, v_nan)

def test_jaccard(u, v, v_nan):
    check_bivariate("jaccard", u, v, v_nan)

def test_kulsinski(u, v, v_nan):
    test_bivariate("kulsinski", u, v, v_nan)

def test_seuclidean(u, v, v_nan):
    V = np.random.rand(*v.shape)
    test_bivariate("seuclidean", u, v, v_nan, V)

def test_citybock(u, v, v_nan):
    test_bivariate("cityblock", u, v, v_nan)

def test_mahalanobis(u, v, v_nan):
    VI = np.random.rand(u.shape[0], v.shape[0])
    test_bivariate("mahalanobis", u, v, v_nan, VI)

def test_chebyshev(u, v, v_nan):
    test_bivariate("chebyshev", u, v, v_nan)

def test_braycurtis(u, v, v_nan):
    test_bivariate("braycurtis", u, v, v_nan)

def test_canberra(u, v, v_nan):
    check_bivariate("canberra", u, v, v_nan)
    #test_bivariate("canberra", u, v, v_nan)

def test_yule(u, v, v_nan):
    test_bivariate("yule", u, v, v_nan)

def test_rogerstanimoto(u, v, v_nan):
    test_bivariate("rogerstanimoto", u, v, v_nan)

def test_russellrao(u, v, v_nan):
    test_bivariate("russellrao", u, v, v_nan)

def test_sokalmichener(u, v, v_nan):
    test_bivariate("sokalmichener", u, v, v_nan)

def test_sokalsneath(u, v, v_nan):
    test_bivariate("sokalsneath", u, v, v_nan)
    test_bivariate("sokalsneath", u > 0.5, v > 0.5, get_vnan_bool(v_nan))

# =============================================================

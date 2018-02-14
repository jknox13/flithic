from __future__ import division, print_function, absolute_import

import mock
import pytest
import importlib
import numpy as np
import scipy.spatial.distance as sci_dist

from numpy.testing import assert_array_almost_equal, assert_raises

from flithic.distance import distance

@pytest.fixture
def u():
    return np.random.rand(100,1)

@pytest.fixture
def v():
    return np.random.rand(100,1)

# ===========================
# Reimport patching decorator to test validitiy inn comparison to scipy
#with mock.patch('flithic.distance.nan_test', lambda x: x):
#importlib.reload(distance)



def test_directed_hausdorff(u, v):
    scipy_ = sci_dist.directed_hausdorff(u,v)
    flithic_ = distance.directed_hausdorff(u,v)

    assert_array_almost_equal( scipy_, flithic_)

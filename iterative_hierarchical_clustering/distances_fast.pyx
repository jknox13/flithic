#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

# Authors : Joseph Knox josephk@alleninstitute.org
# License :

from libc.math cimport isnan
import numpy as np
cimport numpy as np

ctypedef float [:, :] float_array_2d_t
ctypedef double [:, :] double_array_2d_t

cdef fused floating_array_2d_t:
    float_array_2d_t
    double_array_2d_t


np.import_array()

def _sqeuclidean_fast(floating_array_2d_t X,
                    floating_array_2d_t Y,
                    floating_array_2d_t result):
    cdef np.npy_intp i, j, k
    cdef np.npy_intp n_samples_X = 

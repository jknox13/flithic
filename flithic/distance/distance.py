"""
Distance computations

see scipy.spatial.distance for documentation

Edited by Joseph Knox 2018
"""

# Copyright (C) Damian Eads, 2007-2008. New BSD License.

# Edited by Joseph Knox 2018.
# Copyright (C) Joseph Knox, 2018. New BSD License.

from __future__ import division, print_function, absolute_import

__all__ = [
    'braycurtis',
    'canberra',
    'cdist',
    'chebyshev',
    'cityblock',
    'correlation',
    'cosine',
    'dice',
    'directed_hausdorff',
    'euclidean',
    'hamming',
    'is_valid_dm',
    'is_valid_y',
    'jaccard',
    'kulsinski',
    'mahalanobis',
    'minkowski',
    'pdist',
    'rogerstanimoto',
    'russellrao',
    'seuclidean',
    'sokalmichener',
    'sokalsneath',
    'sqeuclidean',
    'yule'
]


from functools import partial, wraps
from collections import namedtuple

import warnings
import numpy as np
import numpy.ma as ma

from scipy._lib.six import callable, string_types
from scipy._lib.six import xrange
from scipy.linalg import norm

import scipy.spatial.distance as sci_dist
from scipy.spatial.distance \
    import (_args_to_kwargs_xdist, _copy_array_if_base_present,
            _filter_deprecated_kwargs, _nbool_correspond_all,
            _nbool_correspond_ft_tf, _validate_cdist_input,
            _validate_pdist_input, _validate_vector,
            _METRICS, _TEST_METRICS, _METRIC_ALIAS, _convert_to_double, squareform)

from . import _distance_wrap
from . import _hausdorff


def nan_test(func):
    @wraps(func)
    def test_nan_and_call(*args, **kwargs):
        if any((np.isnan(arg).any() for arg in args if not callable(arg))):
            # call my version
            return func(*args, **kwargs)
        # call scipy version
        return getattr(sci_dist, func.__name__)(*args, **kwargs)
    return test_nan_and_call

# NOTE: only because scipy 1.0 does not have
def _validate_weights(w, dtype=np.double):
    w = _validate_vector(w, dtype=dtype)
    if np.any(w < 0):
        raise ValueError("Input weights should be all non-negative")
    return w

@nan_test
def directed_hausdorff(u, v, seed=0):
    """
    See scipy.spatial.distance.directed_hausdorff
    """
    u = np.asarray(u, dtype=np.float64, order='c')
    v = np.asarray(v, dtype=np.float64, order='c')
    result = _hausdorff.directed_hausdorff(u, v, seed)
    return result

# TODO : np.linalg.norm for nan
#@nan_test
#def minkowski(u, v, p=2, w=None):
#    """
#    See scipy.spatial.distance.minkowski
#    """
#    u = _validate_vector(u)
#    v = _validate_vector(v)
#    if p < 1:
#        raise ValueError("p must be at least 1")
#    u_v = u - v
#    if w is not None:
#        w = _validate_weights(w)
#        if p == 1:
#            root_w = w
#        if p == 2:
#            # better precision and speed
#            root_w = np.sqrt(w)
#        else:
#            root_w = np.power(w, 1/p)
#        u_v = root_w * u_v
#    dist = norm(u_v, ord=p)
#    return dist
#
#
#@nan_test
#def euclidean(u, v, w=None):
#    """
#    """
#    return minkowski(u, v, p=2, w=w)

def _mask_vector(x):
    x = _validate_vector(x)
    if np.isnan(x).any():
        return ma.array(x, mask=np.isnan(x))
    return x

def _validate_and_mask(x, **kwargs):
    return _mask_vector(_validate_vector(x, **kwargs))

@nan_test
def sqeuclidean(u, v, w=None):
    """
    """
    # Preserve float dtypes, but convert everything else to np.float64
    # for stability.
    utype, vtype = None, None
    if not (hasattr(u, "dtype") and np.issubdtype(u.dtype, np.inexact)):
        utype = np.float64
    if not (hasattr(v, "dtype") and np.issubdtype(v.dtype, np.inexact)):
        vtype = np.float64

    u = _validate_and_mask(u, dtype=utype)
    v = _validate_and_mask(v, dtype=vtype)

    u_v = u - v
    u_v_w = u_v  # only want weights applied once
    if w is not None:
        w = _validate_weights(w)
        u_v_w = w * u_v
    return ma.dot(u_v, u_v_w).data

@nan_test
def correlation(u, v, w=None, centered=True):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)

    if w is not None:
        w = _validate_weights(w)
    if centered:
        umu = ma.average(u, weights=w)
        vmu = ma.average(v, weights=w)
        u = u - umu
        v = v - vmu
    uv = ma.average(u * v, weights=w)
    uu = ma.average(np.square(u), weights=w)
    vv = ma.average(np.square(v), weights=w)
    dist = 1.0 - uv / ma.sqrt(uu * vv)
    return dist


@nan_test
def cosine(u, v, w=None):
    """
    """
    return correlation(u, v, w=w, centered=False)


def hamming(u, v, w=None):
    return sci_dist.hamming(u, v, w=w)


def jaccard(u, v, w=None):
    return sci_dist.hamming(u, v, w=w)


@nan_test
def kulsinski(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if w is None:
        n = float(len(u))
    else:
        w = _validate_weights(w)
        n = w.sum()
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)

    return (ntf + nft - ntt + n) / (ntf + nft + n)


@nan_test
def seuclidean(u, v, V):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    V = _validate_vector(V, dtype=np.float64)
    if V.shape[0] != u.shape[0] or u.shape[0] != v.shape[0]:
        raise TypeError('V must be a 1-D array of the same dimension '
                        'as u and v.')
    return euclidean(u, v, w=1/V)


@nan_test
def cityblock(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    l1_diff = abs(u - v)
    if w is not None:
        w = _validate_weights(w)
        l1_diff = w * l1_diff
    return l1_diff.sum()


@nan_test
def mahalanobis(u, v, VI):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = np.dot(np.dot(delta, VI), delta)
    return np.sqrt(m)


@nan_test
def chebyshev(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if w is not None:
        w = _validate_weights(w)
        has_weight = w > 0
        if has_weight.sum() < w.size:
            u = u[has_weight]
            v = v[has_weight]
    return max(abs(u - v))


@nan_test
def braycurtis(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v, dtype=np.float64)
    l1_diff = abs(u - v)
    l1_sum = abs(u + v)
    if w is not None:
        w = _validate_weights(w)
        l1_diff = w * l1_diff
        l1_sum = w * l1_sum
    return l1_diff.sum() / l1_sum.sum()


@nan_test
def canberra(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v, dtype=np.float64)
    if w is not None:
        w = _validate_weights(w)
    olderr = np.seterr(invalid='ignore')
    try:
        abs_uv = abs(u - v)
        abs_u = abs(u)
        abs_v = abs(v)
        d = abs_uv / (abs_u + abs_v)
        if w is not None:
            d = w * d
        d = np.nansum(d)
    finally:
        np.seterr(**olderr)
    return d


@nan_test
def yule(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if w is not None:
        w = _validate_weights(w)
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)
    return float(2.0 * ntf * nft / np.array(ntt * nff + ntf * nft))


@nan_test
def dice(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if w is not None:
        w = _validate_weights(w)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    else:
        dtype = np.find_common_type([int], [u.dtype, v.dtype])
        u = u.astype(dtype)
        v = v.astype(dtype)
        if w is None:
            ntt = (u * v).sum()
        else:
            ntt = (u * v * w).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
    return float((ntf + nft) / np.array(2.0 * ntt + ntf + nft))


@nan_test
def rogerstanimoto(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if w is not None:
        w = _validate_weights(w)
    (nff, nft, ntf, ntt) = _nbool_correspond_all(u, v, w=w)
    return float(2.0 * (ntf + nft)) / float(ntt + nff + (2.0 * (ntf + nft)))


@nan_test
def russellrao(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
        n = float(len(u))
    elif w is None:
        ntt = (u * v).sum()
        n = float(len(u))
    else:
        w = _validate_weights(w)
        ntt = (u * v * w).sum()
        n = w.sum()
    return float(n - ntt) / n


@nan_test
def sokalmichener(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
        nff = (~u & ~v).sum()
    elif w is None:
        ntt = (u * v).sum()
        nff = ((1.0 - u) * (1.0 - v)).sum()
    else:
        w = _validate_weights(w)
        ntt = (u * v * w).sum()
        nff = ((1.0 - u) * (1.0 - v) * w).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v)
    return float(2.0 * (ntf + nft)) / float(ntt + nff + 2.0 * (ntf + nft))


@nan_test
def sokalsneath(u, v, w=None):
    """
    """
    u = _validate_and_mask(u)
    v = _validate_and_mask(v)
    if u.dtype == v.dtype == bool and w is None:
        ntt = (u & v).sum()
    elif w is None:
        ntt = (u * v).sum()
    else:
        w = _validate_weights(w)
        ntt = (u * v * w).sum()
    (nft, ntf) = _nbool_correspond_ft_tf(u, v, w=w)
    denom = np.array(ntt + 2.0 * (ntf + nft))
    if not denom.any():
        raise ValueError('Sokal-Sneath dissimilarity is not defined for '
                         'vectors that are entirely false.')
    return float(2.0 * (ntf + nft)) / denom

@nan_test
def _correlation_cdist_wrap(XA, XB, dm, **kwargs):
    XA = XA - XA.mean(axis=1, keepdims=True)
    XB = XB - XB.mean(axis=1, keepdims=True)
    _distance_wrap.cdist_cosine_double_wrap(XA, XB, dm, **kwargs)


@nan_test
def _correlation_pdist_wrap(X, dm, **kwargs):
    X2 = X - X.mean(axis=1, keepdims=True)
    _distance_wrap.pdist_cosine_double_wrap(X2, dm, **kwargs)


@nan_test
def pdist(X, metric='euclidean', *args, **kwargs):
    """
    """
    kwargs = _args_to_kwargs_xdist(args, kwargs, metric, "pdist")

    X = np.asarray(X, order='c')

    # The C code doesn't do striding.
    X = _copy_array_if_base_present(X)

    s = X.shape
    if len(s) != 2:
        raise ValueError('A 2-dimensional array must be passed.')

    m, n = s
    out = kwargs.pop("out", None)
    if out is None:
        dm = np.empty((m * (m - 1)) // 2, dtype=np.double)
    else:
        if out.shape != (m * (m - 1) // 2,):
            raise ValueError("output array has incorrect shape.")
        if not out.flags.c_contiguous:
            raise ValueError("Output array must be C-contiguous.")
        if out.dtype != np.double:
            raise ValueError("Output array must be double type.")
        dm = out

    # compute blacklist for deprecated kwargs
    if(metric in _METRICS['minkowski'].aka or
       metric in ['test_minkowski'] or
       metric in [minkowski]):
        kwargs_blacklist = ["V", "VI"]
    elif(metric in _METRICS['seuclidean'].aka or
         metric == 'test_seuclidean' or metric == seuclidean):
        kwargs_blacklist = ["p", "w", "VI"]
    elif(metric in _METRICS['mahalanobis'].aka or
         metric == 'test_mahalanobis' or metric == mahalanobis):
        kwargs_blacklist = ["p", "w", "V"]
    else:
        kwargs_blacklist = ["p", "V", "VI"]

    _filter_deprecated_kwargs(kwargs, kwargs_blacklist)

    if callable(metric):
        mstr = getattr(metric, '__name__', 'UnknownCustomMetric')
        metric_name = _METRIC_ALIAS.get(mstr, None)

        if metric_name is not None:
            X, typ, kwargs = _validate_pdist_input(X, m, n,
                                                            metric_name,
                                                            **kwargs)

        k = 0
        for i in xrange(0, m - 1):
            for j in xrange(i + 1, m):
                dm[k] = metric(X[i], X[j], **kwargs)
                k = k + 1

    elif isinstance(metric, string_types):
        mstr = metric.lower()

        # NOTE: C-version still does not support weights
        if "w" in kwargs and not mstr.startswith("test_"):
            if(mstr in _METRICS['seuclidean'].aka or
               mstr in _METRICS['mahalanobis'].aka):
                raise ValueError("metric %s incompatible with weights" % mstr)
            # need to use python version for weighting
            kwargs['out'] = out
            mstr = "test_%s" % mstr

        metric_name = _METRIC_ALIAS.get(mstr, None)

        if metric_name is not None:
            X, typ, kwargs = _validate_pdist_input(X, m, n,
                                                            metric_name,
                                                            **kwargs)

            # get pdist wrapper
            pdist_fn = getattr(_distance_wrap,
                               "pdist_%s_%s_wrap" % (metric_name, typ))
            pdist_fn(X, dm, **kwargs)
            return dm

        elif mstr in ['old_cosine', 'old_cos']:
            warnings.warn('"old_cosine" is deprecated and will be removed in '
                          'a future version. Use "cosine" instead.',
                          DeprecationWarning)
            X = _convert_to_double(X)
            norms = np.einsum('ij,ij->i', X, X, dtype=np.double)
            np.sqrt(norms, out=norms)
            nV = norms.reshape(m, 1)
            # The numerator u * v
            nm = np.dot(X, X.T)
            # The denom. ||u||*||v||
            de = np.dot(nV, nV.T)
            dm = 1.0 - (nm / de)
            dm[xrange(0, m), xrange(0, m)] = 0.0
            dm = squareform(dm)
        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                dm = pdist(X, _TEST_METRICS[mstr], **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm


# NOTE : may not need
@nan_test
def is_valid_dm(D, tol=0.0, throw=False, name="D", warning=False):
    """
    """
    D = np.asarray(D, order='c')
    valid = True
    try:
        s = D.shape
        if len(D.shape) != 2:
            if name:
                raise ValueError(('Distance matrix \'%s\' must have shape=2 '
                                  '(i.e. be two-dimensional).') % name)
            else:
                raise ValueError('Distance matrix must have shape=2 (i.e. '
                                 'be two-dimensional).')
        if tol == 0.0:
            if not (D == D.T).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' must be '
                                     'symmetric.') % name)
                else:
                    raise ValueError('Distance matrix must be symmetric.')
            if not (D[xrange(0, s[0]), xrange(0, s[0])] == 0).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' diagonal must '
                                      'be zero.') % name)
                else:
                    raise ValueError('Distance matrix diagonal must be zero.')
        else:
            if not (D - D.T <= tol).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' must be '
                                      'symmetric within tolerance %5.5f.')
                                     % (name, tol))
                else:
                    raise ValueError('Distance matrix must be symmetric within'
                                     ' tolerance %5.5f.' % tol)
            if not (D[xrange(0, s[0]), xrange(0, s[0])] <= tol).all():
                if name:
                    raise ValueError(('Distance matrix \'%s\' diagonal must be'
                                      ' close to zero within tolerance %5.5f.')
                                     % (name, tol))
                else:
                    raise ValueError(('Distance matrix \'%s\' diagonal must be'
                                      ' close to zero within tolerance %5.5f.')
                                     % tol)
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e))
        valid = False
    return valid


# NOTE : may not need
@nan_test
def is_valid_y(y, warning=False, throw=False, name=None):
    """
    """
    y = np.asarray(y, order='c')
    valid = True
    try:
        if len(y.shape) != 1:
            if name:
                raise ValueError(('Condensed distance matrix \'%s\' must '
                                  'have shape=1 (i.e. be one-dimensional).')
                                 % name)
            else:
                raise ValueError('Condensed distance matrix must have shape=1 '
                                 '(i.e. be one-dimensional).')
        n = y.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        if (d * (d - 1) / 2) != n:
            if name:
                raise ValueError(('Length n of condensed distance matrix '
                                  '\'%s\' must be a binomial coefficient, i.e.'
                                  'there must be a k such that '
                                  '(k \\choose 2)=n)!') % name)
            else:
                raise ValueError('Length n of condensed distance matrix must '
                                 'be a binomial coefficient, i.e. there must '
                                 'be a k such that (k \\choose 2)=n)!')
    except Exception as e:
        if throw:
            raise
        if warning:
            warnings.warn(str(e))
        valid = False
    return valid


@nan_test
def cdist(XA, XB, metric='euclidean', *args, **kwargs):
    """
    """
    kwargs = _args_to_kwargs_xdist(args, kwargs, metric, "cdist")

    XA = np.asarray(XA, order='c')
    XB = np.asarray(XB, order='c')

    # The C code doesn't do striding.
    XA = _copy_array_if_base_present(XA)
    XB = _copy_array_if_base_present(XB)

    s = XA.shape
    sB = XB.shape

    if len(s) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if s[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = s[0]
    mB = sB[0]
    n = s[1]
    out = kwargs.pop("out", None)
    if out is None:
        dm = np.empty((mA, mB), dtype=np.double)
    else:
        if out.shape != (mA, mB):
            raise ValueError("Output array has incorrect shape.")
        if not out.flags.c_contiguous:
            raise ValueError("Output array must be C-contiguous.")
        if out.dtype != np.double:
            raise ValueError("Output array must be double type.")
        dm = out

    # compute blacklist for deprecated kwargs
    if(metric in _METRICS['minkowski'].aka or
       metric in _METRICS['wminkowski'].aka or
       metric in ['test_minkowski'] or
       metric in [minkowski]):
        kwargs_blacklist = ["V", "VI"]
    elif(metric in _METRICS['seuclidean'].aka or
         metric == 'test_seuclidean' or metric == seuclidean):
        kwargs_blacklist = ["p", "w", "VI"]
    elif(metric in _METRICS['mahalanobis'].aka or
         metric == 'test_mahalanobis' or metric == mahalanobis):
        kwargs_blacklist = ["p", "w", "V"]
    else:
        kwargs_blacklist = ["p", "V", "VI"]

    _filter_deprecated_kwargs(kwargs, kwargs_blacklist)

    if callable(metric):

        mstr = getattr(metric, '__name__', 'Unknown')
        metric_name = _METRIC_ALIAS.get(mstr, None)

        XA, XB, typ, kwargs = _validate_cdist_input(XA, XB, mA, mB, n,
                                                    metric_name, **kwargs)

        for i in xrange(0, mA):
            for j in xrange(0, mB):
                dm[i, j] = metric(XA[i], XB[j], **kwargs)

    elif isinstance(metric, string_types):
        mstr = metric.lower()

        # NOTE: C-version still does not support weights
        if "w" in kwargs and not mstr.startswith("test_"):
            if(mstr in _METRICS['seuclidean'].aka or
               mstr in _METRICS['mahalanobis'].aka):
                raise ValueError("metric %s incompatible with weights" % mstr)
            # need to use python version for weighting
            kwargs['out'] = out
            mstr = "test_%s" % mstr

        metric_name = _METRIC_ALIAS.get(mstr, None)
        if metric_name is not None:
            XA, XB, typ, kwargs = _validate_cdist_input(XA, XB, mA, mB, n,
                                                        metric_name, **kwargs)
            # get cdist wrapper
            cdist_fn = getattr(_distance_wrap,
                               "cdist_%s_%s_wrap" % (metric_name, typ))
            cdist_fn(XA, XB, dm, **kwargs)
            return dm

        elif mstr.startswith("test_"):
            if mstr in _TEST_METRICS:
                dm = cdist(XA, XB, _TEST_METRICS[mstr], **kwargs)
            else:
                raise ValueError('Unknown "Test" Distance Metric: %s' % mstr[5:])
        else:
            raise ValueError('Unknown Distance Metric: %s' % mstr)
    else:
        raise TypeError('2nd argument metric must be a string identifier '
                        'or a function.')
    return dm

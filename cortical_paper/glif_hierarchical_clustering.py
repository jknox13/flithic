# Authors: Joseph Knox josephk@alleninstitute.org
# License:
import numpy as np

from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import skew, zscore

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.utils import check_array 
from sklearn.model_selection import cross_val_score, ShuffleSplit

def _parameter_skew_check(X, alpha=1e-8):
    """Returns X or Log(X) based on skew.

    Parameters
    ----------
    X
    alpha

    Returns
    -------
    """

    C = X.copy()

    for j, col in enumerate(X.T):
        # each parameter, check skew
        log_col = np.log10(col+alpha)

        if skew(col) > skew(log_col):
            # use log_transformed parameter
            C[:,j] = log_col

    return C

def _ward_cluster(X):
    """Clusters 1-corr using Ward distance

    Parameters
    ----------
    X
    Returns
    -------
    """
    # pairwise (1-corr) of zscores
    D = pdist( X, metric="correlation" )

    # return top branch split using ward linkage
    return fcluster( ward(D), 2, criterion="maxclust" )

def _test_clusters(X, clusters):
    """Clusters 1-corr using Ward distance

    Parameters
    ----------
    X
    Returns
    -------
    """
    # C-SVM with radial kernel (sklearn.svm.SVC defaults)
    clf = SVC(C=1.0, kernel="rbf", gamma="auto")

    # 100 random 50/50 splits
    cv = SuffleSplit(n_splits=100, train_size=0.5, test_size=0.5)

    # return score
    return cross_val_score(clf, X, clusters, scoring="accuracy", cv=cv)

class GLIFClustering(BaseEstimator):
    """Clustering described in ...

    Recursively splits the data into clusters that satisfy ...

    NOTE : Docs taken heavily from:
        sklearn.cluster.hierarchical.AgglomerativeClustering

    Parameters
    ----------
    tol : float, optional (default=0.90)
        Tolerance of the split...

    Attributes
    ----------
    X_ :
    labels_ :

    References
    ----------
    CITE GLIF PAPER

    Examples
    --------
    >>> from cortical_paper.clustering import GLIFClustering
    >>>
    """

    def __init__(self, tol=0.80):
        self.tol = tol

    def _fit(self, indices, X):
        """Recursive helper for fit()

        skkdj

        Parameters
        ----------

        Returns
        -------
        """
        
        if X.shape[0] < 2:
            # base case
            # must have > 2 obs to cluster!
            return [indices]

        # ---------------------------------------------------------------------
        # use ward w/ (1-corr) to hierarcically cluster to split 
        # ---------------------------------------------------------------------
        clusters = _ward_cluster(X)

        # ---------------------------------------------------------------------
        # train/test svm (radial kernal on 100 random 50/50 splits of clustering
        # ---------------------------------------------------------------------
        try:
            scores = _test_clusters(X, clusters)
        except ValueError:
            # base case
            # two few of second class to split (say 9:1 or something)
            # assign entire population to terminal cluster
            return [indices]

        # ---------------------------------------------------------------------
        # if min(score) < tol (0.80 in glif paper), recursively repeat 3-5 on
        # each subpopulation
        # ---------------------------------------------------------------------
        if scores.min() < self.tol:
            # base case
            # assign entire population to terminal cluster
            return [indices]

        # recursively split
        a = np.where(clusters == 1)
        a = np.where(clusters == 2)

        return _fit(indices[a], X[a]) + _fit(indices[b], X[b])


    def fit(self, X, y=None):
        """Fit the hierarchical clustering on the data

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The samples a.k.a. observations.
        y : Ignored

        Returns
        -------
        self
        """
        X_ = check_array(X, ensure_min_samples=2, estimator=self)
        
        # ---------------------------------------------------------------------
        # for each parameter, if skew(param) > skew(log(param)), use log(param)
        # ---------------------------------------------------------------------
        X_ = _parameter_skew_check(X_)

        # ---------------------------------------------------------------------
        # z-score all params
        # ---------------------------------------------------------------------
        X_ = zscore(X_)

        # ---------------------------------------------------------------------
        # recursively split in hierarchical fashion
        # ---------------------------------------------------------------------
        indices = np.arange(X_.shape[0])
        self.labels_ = _fit(indices, X_)

        return self

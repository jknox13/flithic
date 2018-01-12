# Authors: Joseph Knox josephk@alleninstitute.org
# License:
import numpy as np

from collections import namedtuple

from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import skew, zscore

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.utils import check_array 
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.utils.validation import check_is_fitted

from tree import Node, iter_bft, pre_order
from utils import unravel_iterable

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
    cv = ShuffleSplit(n_splits=100, train_size=0.5, test_size=0.5)

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

    Cluster = namedtuple("Cluster", "indices, score, name, size")
    Leaf = namedtuple("Leaf", "name, indices")
    Split = namedtuple("Split", "name, children, score, size")

    def __init__(self, tol=0.80):
        self.tol = tol

    def _fit_recursive(self, X, cluster, node=None):
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
            return Node(self.Leaf(cluster.name, cluster.indices))

        # ---------------------------------------------------------------------
        # use ward w/ (1-corr) to hierarcically cluster to split 
        # ---------------------------------------------------------------------
        split = _ward_cluster(X)

        # ---------------------------------------------------------------------
        # train/test svm (radial kernal on 100 random 50/50 splits of clustering
        # ---------------------------------------------------------------------
        try:
            scores = _test_clusters(X, split)
        except ValueError:
            # base case
            # two few of second class to split (say 9:1 or something)
            # assign entire population to terminal cluster
            return Node(self.Leaf(cluster.name, cluster.indices))

        # ---------------------------------------------------------------------
        # if min(score) < tol (0.80 in glif paper), recursively repeat 3-5 on
        # each subpopulation
        # ---------------------------------------------------------------------
        score = scores.min()
        if score < self.tol:
            # base case
            # assign entire population to terminal cluster
            return Node(self.Leaf(cluster.name, cluster.indices))


        # recursively split
        a = np.where(split == 1)
        b = np.where(split == 2)

        A = self.Cluster(cluster.indices[a], score, cluster.name + "1", len(a[0]))
        B = self.Cluster(cluster.indices[b], score, cluster.name + "2", len(b[0]))

        # add score to binary tree
        if node is None:
            node = Node(self.Split(cluster.name, (A.name, B.name), score, cluster.size))
        else:
            # tree is built pre order
            raise ValueError("Should be null!!!")

        node.left = self._fit_recursive(X[a], A, node.left)
        node.right = self._fit_recursive(X[b], B, node.right)

        #return [ self._fit_recursive(X[a], A) ] + [ self._fit_recursive(X[b], B) ]
        return node

    def _fit(self, X):
        """wrapper, inits cluster"""
        # for linkage
        cluster = self.Cluster( np.arange(X.shape[0]), score=0.0, 
                                name="", size=X.shape[0])
        return self._fit_recursive(X, cluster)


#     def _assign_clusters(self, clusters):
#         """...
#         Parameters
#         ----------
#         Returns
#         -------
#         """
#         # get n_obs, initalize label array
#         n = sum( [len(x) for x in clusters] )
#         labels = np.empty(n)
# 
#         for i, cluster in enumerate(clusters):
#             # labels start at 1!!!! (like scipy.hierarchy)
#             labels[cluster] = i+1
# 
#         return labels
# 
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
        self.n_obs_, self.n_params_ = X_.shape
        
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
        self._cluster_tree = self._fit(X_)

        #self.clusters_ = unravel_iterable(clusters)
        for x in iter_bft(self._cluster_tree, reverse=True):
            print(x)

        return self

    def _get_labels(self):
        """..."""
        check_is_fitted(self, "_cluster_tree")
        
        order = pre_order(self._cluster_tree)
        leaves = [x for x in order if isinstance(x, self.Leaf)]
        labels = [ np.repeat(i+1, leaf.indices.size) 
                   for i, leaf in enumerate(leaves) ]
        return np.concatenate(labels)

    @property
    def labels_(self):
        try:
            return self._labels
        except AttributeError:
            self._labels = self._get_labels()
            return self._labels

    def _get_linkage(self):
        """Produces linkage like scipy.hierarchy.linkage"""
        FILL = 0#1e-4

        check_is_fitted(self, "clusters_")

        # mimic early agglomerative clustering
        # fills (n_obs - n_splits) rows
        Z = []
        cluster_ids = []
        cluster_id = self.n_obs_
        for i, cluster in enumerate(self.clusters_):

            # fencepost, link two observations
            a, b = cluster.indices[:2]
            tmp = [[a, b, FILL, 2]]
            for j, index in enumerate(cluster.indices[2:]):
                # assign everyother observation to fencepost cluster
                row = [index, cluster_id, FILL, 3+j]

                tmp.append(row)
                cluster_id += 1

                
            # append to master list
            Z.extend(tmp)
            cluster_ids.append(cluster_id)

        # link actual clustering 
        # post order traversal of binary tree of splits
        # for row in self._score_tree:

        return np.asarray(Z)

#    @property
#    def linkage_(self):
#        try:
#            return self._linkage
#        except AttributeError:
#            self._linkage = self._get_linkage()
#            return self._linkage
#
#    @property
#    def cluster_scores_(self):
#        check_is_fitted(self, "clusters_")
#        return [ cluster.score for cluster in self.clusters_ ]
#
#    @property
#    def cluster_orders_(self):
#        return { i+1 : cluster.order for i, cluster in enumerate(self.clusters_) }
#
#    @property
#    def cluster_sizes_(self):
#        check_is_fitted(self, "clusters_")
#        return [ cluster.size for cluster in self.clusters_ ]
##     def _define_labels(clusters):
#         """
#         ...
#         """
#         in_order = _unravel_iterable(clusters)
# 
#         result = []
#         for i, cluster in enumerate(in_order):
#             arr = np.vstack( (cluster.indices, 
#                               np.repeat(i+1, cluster.size),
#                               np.repeat(cluster.score, cluster.size),
#                               np.repeat(cluster.size, cluster.size)) )
# 
#             result.append(arr)
# 
#         return np.hstack(result).T


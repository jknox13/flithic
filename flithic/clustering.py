# Authors: Joseph Knox josephk@alleninstitute.org
# License:

# TODO: FIX lowest level of dendrogram (linkage property)
# TODO: option to run clustering past tolerance will make this more like
#       scipy.cluster.hierarchy.
# TODO: incorporate heapq - priority trees
import numpy as np

from collections import namedtuple, deque

from scipy.cluster.hierarchy import ward, fcluster
from scipy.spatial.distance import pdist
from scipy.stats import skew, zscore

from sklearn.svm import SVC
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.model_selection import cross_val_score, ShuffleSplit, StratifiedShuffleSplit
from sklearn.utils.validation import check_is_fitted

from tree import Node, iter_bft, iter_tree

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

def _test_clusters(X, clusters, stratified=False):
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
    if stratified:
        splitter = StratifiedShuffleSplit
    else:
        splitter = ShuffleSplit

    cv = splitter(n_splits=100, train_size=0.5, test_size=0.5)

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

    def __init__(self, tol=0.80, stratified=False):
        self.tol = tol
        self.stratified = stratified

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
            scores = _test_clusters(X, split, stratified=self.stratified)
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

        return node

    def _fit(self, X):
        """wrapper, inits cluster"""
        # for linkage
        cluster = self.Cluster( np.arange(self.n_obs_), score=0.0,
                                name="", size=self.n_obs_)
        return self._fit_recursive(X, cluster)

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

        X_ = _parameter_skew_check(X_)

        # z-score all params
        X_ = zscore(X_)

        # recursively split in hierarchical fashion
        self._cluster_tree = self._fit(X_)
        return self

    def _get_labels(self):
        """..."""
        check_is_fitted(self, ["_cluster_tree", "n_obs_"])

        i = 1
        labels = np.empty(self.n_obs_)
        for cluster in iter_tree(self._cluster_tree, order="pre"):
            if isinstance(cluster, self.Leaf):
                labels[cluster.indices] = i
                i += 1
        return labels

    @property
    def labels_(self):
        try:
            return self._labels
        except AttributeError:
            self._labels = self._get_labels()
            return self._labels

    def _get_linkage(self):
        """...:"""
        # NOTE : may rewrite zrow to be row index, (would be zrow+56 or smth)
        check_is_fitted(self, "_cluster_tree")

        FILL=0. #1e-20 # try 0

        # returned linkage (n_obs-1)x4
        Z = []

        # iterates through Z, used for referencing previously formed clusters
        z_row = self.n_obs_ - 1
        name_row_map = {}

        # get leaves, splits in reverse depth-first traversal order
        leaves, splits = [], []
        for cluster in iter_bft(self._cluster_tree, reverse=True):
            if isinstance(cluster, self.Leaf):
                leaves.append(cluster)
            else:
                splits.append(cluster)

        # NOTE : currently generates linkages sequentially
        #        to get the dendrogram to point to center of clusters,
        #        will need to start in the middle
        for leaf in leaves:
            # fencepost
            a, b = leaf.indices[:2]
            tmp = [[a, b, FILL*z_row, 2]] #scipy needs monotonic distances
            z_row += 1

            for j, index in enumerate(leaf.indices[2:]):
                # assign everyother observation to fencepost cluster
                row = [index, z_row, FILL*z_row, 3+j]

                tmp.append(row)
                z_row += 1

            # update
            Z.extend(tmp)
            name_row_map[leaf.name] = z_row

        # DISTANCES ARE CUMMULATIVE!!!
        name_distance_map = dict()
        for split in splits:

            # get distance
            try:
                distance = name_distance_map[split.name]
            except KeyError:
                # no children splits
                distance = split.score

            # parent
            parent = split.name[:-1]
            try:
                name_distance_map[parent] += distance
            except KeyError:
                name_distance_map[parent] = distance

            # indices when children were formed
            a, b = map(name_row_map.get, split.children)
            row = [a, b, distance, split.size]

            # update
            Z.append(row)
            z_row += 1
            name_row_map[split.name] = z_row

        # must contain doubles for scipy
        return np.asarray(Z, dtype=np.float64)

    @property
    def linkage_(self):
        try:
            return self._linkage
        except AttributeError:
            self._linkage = self._get_linkage()
            return self._linkage

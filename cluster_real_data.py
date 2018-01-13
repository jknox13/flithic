# Authors: Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import print_function
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from cortical_paper.clustering import GLIFClustering

def read_excel_file(path, sheetname):
    """Reads data file.

    ...

    Parameters
    ----------
    """
    # read sheet
    df = pd.read_excel(path, sheetname=sheetname)

    # rename to be consistent
    df.rename(columns={"line":"cre_line", 
                       "Source":"source_region", 
                       "target":"target_region",
                       "Source Module" : "source_module",
                       "Target Module" : "target_module",
                       "hemi":"projection_type"},
              inplace=True)
    
    # add agranular column
    df["agranular"] = df.loc[:,"L4"].isnull()

    # return in this order
    return df[["cre_line", "source_region", "target_region", "source_module", 
               "target_module", "agranular","projection_type", "L1", "L2/3", 
               "L4", "L5", "L6a"]]

def plot_dendrogram(X, Z, **kwargs):
    """..."""

    print( Z[Z[:,2] > 0, 2] ) 
    dendrogram(Z, p=5, truncate_mode="level", show_contracted=True, **kwargs)
    plt.show()
    

def plot_clustermap(X, clf, **kwargs):
    """Plots clustermap
    ...
    """
    # color columns with label id
    lut = dict(zip(np.unique(clf.labels_), "rgb"*20)) # FIX!!!!!
    col_colors = map(lut.get, clf.labels_)

    # figure
    plot = sns.clustermap(X.T, row_cluster=False, col_linkage=clf.linkage_,
                          col_colors=col_colors,
                          alpha=0.7, figsize=(15,10), robust=True,
                          xticklabels=[], yticklabels=[], **kwargs)

    plot.savefig("./output/real_data_agranular_clustermap.png")

if __name__ == "__main__":
    # settings
    path = "./excel_files/all above -1.5 NPV by source-line-target.gct.xlsx"
    sheetname = "Sheet1"
    tol = 0.90 # 90% tolerance in GLIF clustering procedure

    # deep tree, need larger limit
    sys.setrecursionlimit(10000)

    # read dataframe
    df = read_excel_file(path, sheetname)

    # granular
    granular = df.loc[~df.agranular]
    agranular = df.loc[df.agranular]
    agranular = agranular.drop("L4", axis=1)
    X = granular.loc[:,"L1":"L6a"]

    # fake data
    #a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[50,])
    #b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[6,])
    #X = np.concatenate((a, b),) + 10

    # use glif clustering procedure
    # print("tol\tn_clusters")
    tols = 0.01*np.arange(80,100)
    for tol in tols:
        clf = GLIFClustering(tol=tol)
        clf.fit(X)
        print("{}%\t".format(tol*100),len(np.unique(clf.labels_)))

    # fit 
    #clf = GLIFClustering(tol=tol)
    #clf.fit(X)

    # plot dendrogram
    #plot_dendrogram(X, clf.linkage_, labels=clf.labels_)
    #print( "tolerance  :{}\%".format(100*tol) )
    #print( "N clusters :", len(np.unique(clf.labels_)) )
    #plot_clustermap(X, clf)

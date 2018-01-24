# Authors: Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import print_function
import os
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

    # make RSPd agranular
    df.loc[(df.target_region == "RSPd"),"L4"] = np.nan

    # add agranular column
    df["agranular"] = df.loc[:,"L4"].isnull()

    # add label col
    df["label"] = 0

    # return in this order
    return df[["cre_line", "source_region", "target_region", "source_module",
               "target_module", "agranular","projection_type", "L1", "L2/3",
               "L4", "L5", "L6a", "label"]]

def plot_dendrogram(X, Z, **kwargs):
    """..."""

    print( Z[Z[:,2] > 0, 2] )
    dendrogram(Z, p=5, truncate_mode="level", show_contracted=True, **kwargs)
    plt.show()


def plot_clustermap(X, clf, title="", filename="tmp", **kwargs):
    """Plots clustermap
    ...
    """
    # color columns with label id
    # TODO : fix!!!
    lut = dict(zip(np.unique(clf.labels_), "rgb"*20))
    col_colors = map(lut.get, clf.labels_)

    # figure
    plot = sns.clustermap(X.T, row_cluster=False, col_linkage=clf.linkage_,
                          col_colors=col_colors,
                          alpha=0.7, figsize=(15,10), robust=True,
                          xticklabels=[], yticklabels=[], **kwargs)

    plt.title(title)
    plot.savefig(filename)


if __name__ == "__main__":
    # settings
    path = "./excel_files/all above -1.5 NPV by source-line-target.gct.xlsx"
    sheetname = "Sheet1"
    tol = 0.85 # 90% tolerance in GLIF clustering procedure
    stratified = True # stratifiedshufflesplit or shufflesplit
    suffix = "_stratified" if stratified else ""

    # output
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # deep tree, need larger limit
    sys.setrecursionlimit(10000)

    # read dataframe
    df = read_excel_file(path, sheetname)

    # fit and plot
    for title, agranular in zip( ("granular", "agranular"), (False, True) ):
        # subset
        X = df.loc[(df.agranular==agranular),"L1":"L6a"]
        if agranular:
            X = X.drop("L4", axis=1)

        # fit clustering
        clf = GLIFClustering(tol=tol, stratified=stratified)
        clf.fit(X)

        # update df
        offset = len(df.label.unique()) - 1
        df.loc[(df.agranular==agranular), "label"] = clf.labels_ + offset

        # plot
        filename = "real_data_%s_clustermap%s.png" % (title, suffix)
        filename = os.path.join(output_dir, filename)
        plot_clustermap(X, clf, title=title+suffix, filename=filename)

        # print summary
        print( "", title, "=============", sep="\n" )
        print( "tolerance  :{}\%".format(100*tol) )
        print( "N clusters :", len(np.unique(clf.labels_)) )

    # save updated df with labels
    df.to_csv( os.path.join(output_dir, "clustered_real_data%s.csv" % suffix) )

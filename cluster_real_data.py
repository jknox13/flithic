# Authors: Joseph Knox josephk@alleninstitute.org
# License:
from __future__ import print_function
import numpy as np
import pandas as pd

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


if __name__ == "__main__":
    # settings
    path = "./excel_files/all above -1.5 NPV by source-line-target.gct.xlsx"
    sheetname = "Sheet1"
    tol = 0.90 # 90% tolerance in GLIF clustering procedure

    # read dataframe
    df = read_excel_file(path, sheetname)

    # granular
    granular = df.loc[~df.agranular]

    # use glif clustering procedure
    clf = GLIFClustering(tol=tol)

    # cluster
    X = granular.loc[:,"L1":"L6a"]
    clf.fit(X)

    # print n_clusters
    n_clusters = len( np.unique(clf.labels_) )
    print( "Total # Clusters", n_clusters )

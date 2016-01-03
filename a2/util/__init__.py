import pandas as pd
import numpy  as np
import scipy.stats as scs

__author__ = 'mahieke'

def align(data):
    """
    Return the data aligned with its mean values.

    Args:
        data (pandas.core.frame.DataFrame):
            Dataset which should be aligned

    Returns:
        pandas.core.frame.DataFrame: New DataFrame with aligned data.
    """
    mean = data.mean()
    return data.sub(mean)


def pca(data, l=None):
    """
    principal_component_analysis
    Get base vectors q_i, projection of x_i on q_i and the standard
    deviation of all q_i. User can Limit dimension with l.

    Args:
        data (pandas.core.frame.DataFrame):
             Dataset which should be aligned
        l (int): Maximum amount of variables of Output

    Returns:
        Qi, Ai, Sigma (3-tuple): Qi are the basis vectors of the 
            principal components. Ai are the new principal 
            components. Sigma is the standard deviation of the 
            principal components.
    """
    d, n = data.shape

    if not l:
        l = n

    aligned_data = align(data)

    # singular value decomposition
    U, d, V = np.linalg.svd(aligned_data, full_matrices=False)

    # build diagonal matrix
    D = np.diag(d)

    # base vector
    Qi = V[:,:]

    # projection
    Ai = U.dot(D)[:,:]

    # standard deviation
    Sigma = d[:]

    return Qi, Ai, Sigma


def pca_correlation(data, pca_data, l=None):
    """
    Creates a DataFrame with the correlation between the
    pca_data and the original data frame. Principal 
    components can be limited by l.
    """

    d,n = data.shape
    
    if l:
        l = min(l,n)
    else:
        l = n

    # corrolate each dataset of pca_data with dataset data
    corr = [[scs.pearsonr(data[lbl], a)[0] for lbl in data] for a in pca_data.transpose()[:l,:]] 
                    
    return pd.DataFrame(corr, columns=data.columns, index=["a{}".format(s) for s in range(0,l)])

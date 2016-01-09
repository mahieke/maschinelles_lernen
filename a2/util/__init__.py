import pandas as pd
import numpy  as np
import scipy.stats as scs
from skimage.io import imread
import os

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
            Dataset which should be aligned.
        l (int): Maximum amount of variables of Output

    Returns:
        Qi, Ai, Sigma (3-tuple): Qi are the basis vectors of the 
            principal components. Ai are the new principal 
            components. Sigma is the standard deviation of the 
            principal components.
    """
    d, n = data.shape

    if l:
        l = min(l,n)
    else:
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

    Args:
        data (pandas.core.frame.DataFrame):
            Original data which shlould be correlated with pca_data.
        pca_data (pandas.core.frame.DataFrame):
            Principal component data which will be correlated with data.
        l (int): Maximum amount of variables of Output

    Returns (pandas.core.frame.DataFrame):
        Correlation matrix of pca_data and data
    """

    d,n = data.shape
    
    if l:
        l = min(l,n)
    else:
        l = n

    # corrolate each dataset of pca_data with dataset data
    corr = [[scs.pearsonr(data[lbl], a)[0] for lbl in data] for a in pca_data.transpose()[:l,:]] 
                    
    return pd.DataFrame(corr, columns=data.columns, index=["a{}".format(s) for s in range(0,l)])


def get_person_images(path, ext, min):
    """
    Returns all directories which have a min amount of files of type ext.
    Args:
        path (string): path entrypoint wehre to start
        ext (string): extension of the files
        min (int): minimal amount of files in directory
    Returns (list):
        A list with tuples containing the root path an the containing files
        of the matching directories.

    """
    import re
    # for all leaves in directory tree
    for root, dirs, files in os.walk(path):
        if not dirs:
            filtered_files = [x for x in files if re.search('{}$'.format(ext), x)]
            if len(filtered_files) >= min:
                yield (root, files)
                

def imstack2vectors(image):
    """
    Args:
        image:

    Returns:

    """
    s = image.shape
    if len(s) == 3:
        return np.asarray([image[:,:,index].flatten() for index in range(s[2])]).T
    else:
        return image.flatten()


def get_dataset(root, files, scale_factor=1):
    """
    Args:
        root (string): path to images
        files (list): list of image files in directory root
        scale_factor (int): scale image by this factor

    Returns (dict):
        Returns _data_ in a numpy array and metadata (_name_ and _amount_ of data)
        keys: 'amount', 'name', 'data'
    """
    name = root.split('/')[-1]
    amount_img = len(files)
    frame = []

    for f in files:
        img = imread('{}/{}'.format(root,f), as_grey=True)

        # make it work if someone
        scale = int(scale_factor)
        if scale > 1:
            img = img[::scale,::scale]

        img = imstack2vectors(img)

        frame.append(img)

    nparray = np.array(frame)

    return name, nparray, amount_img

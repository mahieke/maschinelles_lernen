import numpy as np
import pandas as pd

def create_isotropic_gaussian_twindataset(pos, amount_data, variance, sep_vec):
    """
    Creates two isotropic gaussian datasets with a given variance.

    Args:
        pos: centroid of the two datasets
        amount_data: amount of data per dataset
        sep_vec: sepration vector between centroid and each centerpoint of the datasets
        variance: varaince of the gaussian distribution

    Returns:

    """
    sigma = np.sqrt(variance)
    mu1 = np.add(pos,sep_vec)
    mu2 = np.subtract(pos, sep_vec)

    d1 = np.random.normal(0.0, sigma, (amount_data, 2))
    d2 = np.random.normal(0.0, sigma, (amount_data, 2))
    return (np.add(d1, mu1), [1]*amount_data), (np.add(d2, mu2), [-1]*amount_data)



class perceptron(object):
    """
    Perceptron learning machine
    """
    def __init__(self, data):
        """
        Args:
            data: List of 2 Datasets (2 Classes)
                Each must datapoint must be in the same vector space R^d
        """
        self.group1 = data[0]
        self.group2 = data[1]
        self.label1 = 1
        self.label2 = -1

    def learn(self, steps):
        pass

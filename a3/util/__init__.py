import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    return np.asmatrix(np.add(d1, mu1)), np.asmatrix(np.add(d2, mu2)), ['DarkGreen', 'Yellow']



class Perceptron(object):
    """
    Perceptron learning machine
    """
    def __init__(self, data, colors):
        """
        Args:
            data: List of 2 Datasets (2 Classes)
                Each must datapoint must be in the same vector space R^d
        """
        self._data = np.vstack(data)
        self.colors = colors
        amount1 = data[0].shape[0]
        amount2 = data[1].shape[0]
        self.l, self.d = self._data.shape

        self.R = np.max(np.linalg.norm(self._data, axis=1)) ** 2

        self.data  = np.zeros((self.l, self.d + 1))
        self.data[:,0] = self.R
        self.data[:,1:] = self._data
        self.labels = np.vstack([1]*amount1 + [-1]*amount2)
        self.weights = np.zeros(self.d + 1)
        self.eta = 1 # learnrate

        print("data set length: {}".format(self.l))
        print("data point dimension: {}".format(self.d))
        print("R: {}".format(self.R))

        self.wsteps = []


    @staticmethod
    def func_margin(xi, yi, w):
        return yi * ( np.dot(xi,w) )

    def get_all_margins(self):
        return np.array([Perceptron.func_margin(xi,yi, self.weights) for xi, yi in zip(self.data, self.labels)])

    def learn(self, max_steps=0):
        """
        Args:
            max_steps: steps

        Returns:
            None

        """
        steps = 0
        i = 0
        mi = self.get_all_margins()
        while any(mi <= 0):
            if mi[i] <= 0: # update on error classification
                # update weight vector
                self.weights = self.weights + self.labels[i] * self.eta * np.add(self.weights, self.data[i])

                self.wsteps.append(self.weights.copy())
                # update functional margins
                mi = self.get_all_margins()

            i = (i + 1) % self.l
            if i == 0:
                steps = steps + 1

            if max_steps > 0 and steps > max_steps:
                break

        print("correction steps: {}".format(len(self.wsteps)))


    def plot_result2D(self, amount_correction_steps=2):
        """
        Does only work with 2 dimensional data
        Args:
            amount_correction_steps: limits plots of the parting planes
        Returns:
            plot object
        """
        if self.d != 2:
            raise TypeError("Datapoints must be 2 dimensional")

        x_min = np.min(self._data[:,0])
        x_max = np.max(self._data[:,0])
        y_min = np.min(self._data[:,1])
        y_max = np.max(self._data[:,1])

        min_value = min(x_min, y_min) - 1
        max_value = max(x_max, y_max) + 1

        axis_x = np.arange(min_value,max_value)
        axis_y = -(self.weights[0] + axis_x * self.weights[1])/self.weights[2]

        amount = len(self.wsteps)

        if amount_correction_steps >= 0 and amount_correction_steps < amount:
            amount = amount_correction_steps + 1

        rows = np.ceil(amount / 3) + 1

        fig = plt.figure(figsize=(rows*7,15))

        tmp_ax = plt.subplot(rows, 3, 2)
        tmp_ax.axis([min_value,max_value, min_value,max_value])

        ax = [tmp_ax]


        for j in range(1, amount):
            tmp_ax = plt.subplot(rows, 3, 3 + j)
            tmp_ax.axis([min_value,max_value, min_value,max_value])
            ax.append(tmp_ax)


        mask1 = self.labels == 1
        mask2 = self.labels == -1

        d1 = np.compress(mask1[:,0], self._data, axis=0)
        d2 = np.compress(mask2[:,0], self._data, axis=0)

        for axi in ax:
            axi.scatter(d1[:,0], d1[:,1], c=self.colors[0])
            axi.scatter(d2[:,0], d2[:,1], c=self.colors[1])

        self.plot_parting_plane('Blue', ax[0], "End result")

        lower = len(self.wsteps) - amount

        i = lower
        for axi, w in zip(ax[1:], self.wsteps[lower-1:-1]):
            self.plot_parting_plane('Blue', axi, 'Step {}'.format(i+1), w)
            i += 1

        return ax


    def plot_parting_plane(self, color, ax, name, weights=None):
        """
        Args:
            color: color for the plane
            ax: matplotlib ax object
            weights: weight vector, if none the current weight vector of the object is used
        """
        xLimits = np.asarray(ax.get_xlim())
        if weights is None:
            weights = self.weights

        if abs(weights[2]) > abs(weights[1]):
            # if w[1]>w[0] in absolute value, plane is likely to be leaving tops of plot
            x0 = xLimits
            x1 = -(weights[0] + x0 * weights[1])/weights[2]
        else:
            # otherwise plane is likely to be leaving sides of plot.
            x1 = xLimits
            x0 = -(weights[0] + x1 * weights[1])/weights[2]

        ax.plot(x0, x1,'--', color=color)
        ax.set_title(name)
import numpy as np
import os
import math
from skimage.io import imread
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



class Perceptron(object):
    """
    Perceptron learning machine
    """
    def __init__(self, data):
        """
        Args:
            data: List of 2 Datasets (2 Classes)
                Each must datapoint must be in the same vector space R^d
        """
        _data = np.vstack(data)
        amount1 = data[0].shape[0]
        amount2 = data[1].shape[0]
        self.l, self.d = _data.shape

        self.R = np.max(np.linalg.norm(_data, axis=1)) ** 2

        self.data = np.zeros((self.l, self.d))
        self.data[:,:] = _data
        self.y = np.vstack([1]*amount1 + [-1]*amount2)
        self.w = np.zeros(self.d)
        self.b = 0
        self.eta = 0.1 # learnrate

        print("data set length: {}".format(self.l))
        print("data point dimension: {}".format(self.d))
        print("R: {}".format(self.R))

        self.wsteps = []
        self.mi = np.zeros(self.l)


    def learn(self, max_iterations=0):
        """
        Args:
            max_iterations: limits how often the dataset will be iterrated

        Returns:
            None

        """
        steps = 0
        i = 0

        while min(self.mi) <= 0:
            i = (i + 1) % self.l

            # update functional margin
            self.mi[i] = self.y[i] * (np.dot(self.w, self.data[i]) + self.b)

            if self.mi[i] <= 0: # update on error classification
                # update weight vector
                self.w = self.w + self.y[i] * self.eta * self.data[i]
                self.b = self.b + self.y[i] * self.eta * self.R

                self.wsteps.append((self.w.copy(), self.b))


            if i == 0:
                steps += 1

            if max_iterations > 0 and steps > max_iterations:
                print("Did not find linear seperation after {} steps".format(self.l * steps))
                break

        print("correction steps: {}".format(len(self.wsteps)))

    def classify(self, data, labels):
        """

        Args:
            data: data matrix which should be classified
            labels: vector with labels to check classification

        Returns:

        """

        false_negative, true_negative, false_positive, true_positive = 0,0,0,0

        for xi, yi  in zip(data, labels):
            m = xi.dot(self.w) + self.b

            if m <= 0: # negative
                if int(yi) == -1:
                    true_negative += 1
                else:
                    false_negative += 1

            else: # positive
                if int(yi) == -1:
                    false_positive += 1
                else:
                    true_positive += 1

        print("False negative (Miss): {} --> {:.2f}%".format(false_negative, 100 * false_negative / len(data)))
        print("False positive (Fehlalarmrate): {} --> {:.2f}%".format(false_positive, 100 * false_positive / len(data)))
        print("True negative (korrekte Rückweisung): {} --> {:.2f}%".format(true_negative, 100 * true_negative / len(data)))
        print("True positive (Detektionswahrscheinlichkeit): {} --> {:.2f}%".format(true_positive, 100 * true_positive / len(data)))


    def plot_result2D(self, colors, amount_correction_steps=2, ):
        """
        Does only work with 2 dimensional data
        Args:
            amount_correction_steps: limits plots of the parting planes
        Returns:
            plot object
        """
        if self.d != 2:
            raise TypeError("Datapoints must be 2 dimensional")

        x_min = np.min(self.data[:,0])
        x_max = np.max(self.data[:,0])
        y_min = np.min(self.data[:,1])
        y_max = np.max(self.data[:,1])

        min_value = min(x_min, y_min) - 1
        max_value = max(x_max, y_max) + 1

        amount = len(self.wsteps)

        if amount_correction_steps >= 0 and amount_correction_steps < amount:
            amount = amount_correction_steps + 1

        rows = np.ceil(amount / 3) + 1

        plt.figure(figsize=(15,5*rows))

        tmp_ax = plt.subplot(rows, 3, 2)
        tmp_ax.axis([min_value,max_value, min_value,max_value])

        ax = [tmp_ax]

        for j in range(0, amount):
            tmp_ax = plt.subplot(rows, 3, 4 + j)
            tmp_ax.axis([min_value,max_value, min_value,max_value])
            ax.append(tmp_ax)


        mask1 = self.y == 1
        mask2 = self.y == -1

        d1 = np.compress(mask1[:,0], self.data, axis=0)
        d2 = np.compress(mask2[:,0], self.data, axis=0)

        for axi in ax:
            axi.scatter(d1[:,0], d1[:,1], c=colors[0])
            axi.scatter(d2[:,0], d2[:,1], c=colors[1])

        self.plot_parting_plane('Blue', ax[0], "End result")

        lower = len(self.wsteps) - amount

        i = lower
        for axi, plane_param in zip(ax[1:], self.wsteps[lower:]):
            self.plot_parting_plane('Blue', axi, 'Step {}'.format(i+1), plane_param)
            i += 1

        return ax


    def plot_parting_plane(self, color, ax, name, plane_param=None):
        """
        Args:
            color: color for the plane
            ax: matplotlib ax object
            weights: weight vector, if none the current weight vector of the object is used
        """
        xLimits = np.asarray(ax.get_xlim())
        if plane_param is None:
            w = self.w
            b = self.b
        else:
            w = plane_param[0]
            b = plane_param[1]

        if abs(w[1]) > abs(w[0]):
            # if w[1]>w[0] in absolute value, plane is likely to be leaving tops of plot
            x0 = xLimits
            x1 = -(b + x0 * w[0])/w[1]
        else:
            # otherwise plane is likely to be leaving sides of plot.
            x1 = xLimits
            x0 = -(b + x1 * w[0])/w[1]

        ax.plot(x0, x1,'--', color=color)
        ax.set_title(name)


    def plot_discriminant_function(self):
        window_size = np.asarray([-50, 50])

        x = np.arange(window_size[0]-1, window_size[1]+1)
        y = np.arange(window_size[0]-1, window_size[1]+1)

        X, Y = np.meshgrid(x, y)
        Z = np.zeros(X.shape)
        tmp_array = np.zeros(2)

        for i in range(0, len(Z)):
            for j in range (0, len(Z)):
                tmp_array[0] = X[i][j]
                tmp_array[1] = Y[i][j]
                scalar = np.dot(self.w,tmp_array)
                res = scalar + self.b
                if res < 0:
                    res = -res
                Z[i][j] = res

        plt.contourf(X,Y,Z, 100, cmap=plt.get_cmap('afmhot'))


class GaussianNaiveBayes(object):
    def __init__(self, data):
        self.data_class1 , self.data_class2 = data

        self.l_class1 = len(self.data_class1)
        self.l_class2 = len(self.data_class2)
        self.l_all = self.l_class1 + self.l_class2


    def learn(self):
        self.p_class1 = self.l_class1 / self.l_all
        self.p_class2 = self.l_class2 / self.l_all

        # A1 flattens the resulting (1,d) matrix
        self.mean_class1 = np.mean(self.data_class1, axis=0).A1
        self.mean_class2 = np.mean(self.data_class2, axis=0).A1

        self.std_class1 = np.std(self.data_class1, axis=0).A1
        self.std_class2 = np.std(self.data_class2, axis=0).A1


    def classify(self, data, label):
        """
        Args:
            data: datapoints which should be classified
            label: label_vector to check if classification is correct

        Returns:

        """
        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        classifaction_vector = []

        for datapoint, y in zip(data, label):
            dp = datapoint.A1
            likelihood_class1 = np.multiply.reduce([GaussianNaiveBayes.GNB(x,mu,sigma) for x, mu, sigma in zip(dp, self.mean_class1, self.std_class1)])
            likelihood_class2 = np.multiply.reduce([GaussianNaiveBayes.GNB(x,mu,sigma) for x, mu, sigma in zip(dp, self.mean_class2, self.std_class2)])

            if likelihood_class1 * self.p_class1 > likelihood_class2 * self.p_class2:
                # datapoint belongs to class1: true
                if int(y) == 1:
                    true_positive += 1
                else:
                    true_negative += 1

                classifaction_vector.append(1)
            else:
                #datapoint belongs to class2: false
                if int(y) == -1:
                    false_positive += 1
                else:
                    false_negative += 1

                classifaction_vector.append(-1)


        print("False negative (Miss): {} --> {:.2f}%".format(false_negative, 100 * false_negative / len(data)))
        print("False positive (Fehlalarmrate): {} --> {:.2f}%".format(false_positive, 100 * false_positive / len(data)))
        print("True negative (korrekte Rückweisung): {} --> {:.2f}%".format(true_negative, 100 * true_negative / len(data)))
        print("True positive (Detektionswahrscheinlichkeit): {} --> {:.2f}%".format(true_positive, 100 * true_positive / len(data)))

        return classifaction_vector

    def plot_discriminant_function(self, colors):
        ax = plt.subplot(111)

        min_val = -2
        max_val =  2

        ax.axis([min_val,max_val, min_val,max_val])
        window_size =  np.arange(min_val-1, max_val+1)

        X,Y = np.meshgrid(window_size,window_size)
        Z = np.zeros(X.shape)
        x = np.zeros(2)

        for i in range(0, Z.shape[0]):
            for j in range (0, Z.shape[1]):
                x[0] = X[i][j]
                x[1] = Y[i][j]

                likelihood_class1 = np.multiply.reduce([GaussianNaiveBayes.GNB(x,mu,sigma) for x, mu, sigma in zip(x, self.mean_class1, self.std_class1)])
                likelihood_class2 = np.multiply.reduce([GaussianNaiveBayes.GNB(x,mu,sigma) for x, mu, sigma in zip(x, self.mean_class2, self.std_class2)])

                Z[i][j] = ((self.p_class1 * likelihood_class1) / (self.p_class2 * likelihood_class2)) - 1

        ax.contourf(X,Y,Z, 2, cmap=plt.get_cmap('gray'))
        ax.scatter(self.data_class1[:,0], self.data_class1[:,1], c=colors[0])
        ax.scatter(self.data_class2[:,0], self.data_class2[:,1], c=colors[1])

    @staticmethod
    def GNB(x, sigma, mu):
        variance = sigma**2
        pi = math.pi
        a = 1 / np.sqrt(2*pi*variance)
        exp = np.exp((-0.5 / variance) * (x - mu)**2)

        return a*exp
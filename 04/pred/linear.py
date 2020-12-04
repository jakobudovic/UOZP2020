from collections import defaultdict
import numpy
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse

import scipy.sparse as sp
import numpy as np
import lpputils as lpp


def append_ones(X):
    if sp.issparse(X):
        return sp.hstack((np.ones((X.shape[0], 1)), X)).tocsr()
    else:
        return np.hstack((np.ones((X.shape[0], 1)), X))


def hl(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return x.dot(theta)


def cost_grad_linear(theta, X, y, lambda_):
    # do not regularize the first element
    sx = hl(X, theta)
    j = 0.5 * numpy.mean((sx - y) * (sx - y)) + 1 / 2. * lambda_ * theta[1:].dot(theta[1:]) / y.shape[0]
    grad = X.T.dot(sx - y) / y.shape[0] + numpy.hstack([[0.], lambda_ * theta[1:]]) / y.shape[0]
    return j, grad

def past_month(x):
    example = [0] * 32
    departure = lpp.parsedate(x[6])

    if (departure.month in (1, 2, 11, 12)):  # gledamo samo november/december
    # if (departure.month):  # gledamo samo november/december
        example[departure.hour] = 1  # first 24 slots are for deprature hours
        example[departure.isoweekday() + 23] = 1  # last 7 slots are for days

    """
    if (departure.month in (1, 2, 11, 12)):  # gledamo samo november/december
    # if (departure.month):  # gledamo samo november/december
        example[departure.hour] = 1  # first 24 slots are for deprature hours
        if departure.isoweekday() < 6: # med tednom
            example[23] = 1  # last 7 slots are for days
        else:
            example[1 + 23] = 1  # last 7 slots are for days
    """

    return example # all 0s if the month is not nov/dec

class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """

        # X, y = [], [] # X: data, y: duration of route
        th = fmin_l_bfgs_b(cost_grad_linear,
                           x0=numpy.zeros(X.shape[1]),
                           args=(X, y, self.lambda_))[0]

        return LinearRegClassifier(th)


class LinearRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        # print("in lin reg class")
        # print(x)
        # x = numpy.hstack(([1.], past_month(x)))
        # print("x after hstack: ", x)
        return hl(x, self.th)

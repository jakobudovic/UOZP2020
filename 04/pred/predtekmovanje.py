from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse
import csv

import scipy.sparse as sp
import numpy as np
import os
import sys

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
    j = 0.5 * np.mean((sx - y) * (sx - y)) + 1 / 2. * lambda_ * theta[1:].dot(theta[1:]) / y.shape[0]
    grad = X.T.dot(sx - y) / y.shape[0] + np.hstack([[0.], lambda_ * theta[1:]]) / y.shape[0]
    return j, grad


class LinearLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = append_ones(X)

        th = fmin_l_bfgs_b(cost_grad_linear,
                           x0=np.zeros(X.shape[1]),
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
        x = np.hstack(([1.], x))
        return hl(x, self.th)


if __name__ == "__main__":

    # select working dir manually
    os.chdir("/home/jakob/git/UOZP2020/04/pred")

    files = [f for f in os.listdir('.') if os.path.isfile(f)]
    for f in files:
        print(f)




from collections import defaultdict
from scipy.optimize import fmin_l_bfgs_b
import scipy
import scipy.sparse
import csv

import scipy.sparse as sp
import numpy as np
import os
import sys
import csv
import linear

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

def join_day_time(data):
    """
    join day and time of one departure to one 1D vector
    :param data: Dates in non-parsed format
    :return: logical matrix with 2 ones in each row;
            - first one for departure hour
            - second one for departure day of the week
    """
    X = []
    for d in data:
        example = [0] * 31  # 24 + 7
        departure = lpp.parsedate(d[6]) # departure time
        if (departure.month): # december only
            example[departure.hour] = 1 # first 24 slots are for deprature hours
            example[departure.isoweekday() + 23] = 1 # last 7 slots are for days
            X.append(example)
    return X

def read_file(file_path):
    os.chdir("/home/jakob/git/UOZP2020/04/pred")    # select working dir manually
    csvreader = csv.reader(open(file_path), delimiter='\t')
    list = [d for d in csvreader]
    # headers = next(my_data) # remove header
    return np.array(list)

if __name__ == "__main__":
    X_pred = read_file('./train_pred.csv')
    X_test = read_file('./test_pred.csv')

    # build our model
    lin = linear.LinearLearner(lambda_=1.)
    # prediction_model = lin(X,y) # feed/train our model
    # results = [prediction_model(example) for example in X_test_matrix]

    print(lin)




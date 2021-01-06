from typing import Union

import numpy as np
from matplotlib import pyplot
from numpy.core._multiarray_umath import ndarray
from scipy.optimize import fmin_l_bfgs_b

from math import e
from math import log
from math import sqrt
import math
import os
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt

def draw_decision(X, y, classifier, at1, at2, grid=50):
    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02 * difx
    minx -= 0.02 * difx
    maxy += 0.02 * dify
    miny -= 0.02 * dify

    pyplot.figure(figsize=(8, 8))

    for c,(x,y) in zip(y,points):
        pyplot.text(x, y, str(c), ha="center", va="center")
        pyplot.scatter([x], [y], c=["b", "r"][int(c) != 0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi, x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi, y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x, y])
            dists = (diff[:, 0]**2 + diff[:, 1]**2)**0.5  # euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pyplot.imshow(prob, extent=(minx, maxx, maxy, miny), cmap="seismic")

    pyplot.xlim(minx, maxx)
    pyplot.ylim(miny, maxy)
    pyplot.xlabel(at1)
    pyplot.ylabel(at2)

    pyplot.show()


def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # returns between [0,1], i could add some really small value here to avoid undefined log(0) later
    return 1 / (1 + e ** (-x.dot(theta)))  # + 1.e-15


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    # ... dopolnite (naloga 1, naloga 2)
    # print("lambda_ in cost:", lambda_)
    reg = lambda_ * sum([e ** 2 for e in theta])  # L2 reg. Ridge Regression
    S = [yi * log(max(h(x, theta), 1.e-15)) + (1 - yi) * log(max((1 - h(x, theta)), 1.e-15)) for x, yi in zip(X, y)]
    # print("sum(S): ", sum(S))
    # print("return: ", -1 / len(y) * sum(S) + reg)
    return -1 / len(y) * sum(S) + reg


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    # ... dopolnite (naloga 1, naloga 2)
    l = []
    for i, e in enumerate(theta):
        l.append(1 / len(y) * sum([(h(x, theta) - yi) * x[i] for x, yi in zip(X, y)]) + 2 * lambda_ * e)
    return np.array(l)


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    l = []
    alpha = 0.0001
    for i, e in enumerate(theta):
        # formula:  (f(h + delta) - f(h - delta)) / 2 * delta
        t1 = [x for x in theta] # copy theta to new variable
        t2 = [x for x in theta]  # copy theta to new variable
        t1[i] = theta[i] + alpha
        t2[i] = theta[i] - alpha
        derivative = (cost(t1, X, y, lambda_) - cost(t2, X, y, lambda_)) / (2 * alpha)
        l.append(derivative)
    return np.array(l)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    # prečno preverjanje (cross validation)
    predictions = []
    seed = 42 # could be any number
    rng = range(0,X.shape[0]) # range of our data, number of examples

    # shuffle data before performing kx CV
    np.random.seed(seed)
    shuffle_idx = list(rng)
    np.random.shuffle(shuffle_idx) # shuffle our array indexes

    X_shuff = np.array([X[k] for k in shuffle_idx])
    y_shuff = np.array([y[k] for k in shuffle_idx])

    predictions = []
    for i in range(1,k+1):
        X_train, X_test, y_train, y_test = k_fold(X_shuff, y_shuff, i, k)
        classifier = learner(X_train, y_train)
        rez = [classifier(example) for example in X_test]
        predictions = predictions + rez

    # get our shuffled indexes back in order, go over all range of data and find on what index is he and get that result
    predictions_ordered = [predictions[shuffle_idx.index(j)] for j in rng]
    return predictions_ordered

def k_fold(X, y, i, k):
    """
    :param X: train + test data
    :param y: test + test data
    :param i: number of fold iteration
    :param k: number of folds (determines, how we split the data)
    :return: X train and test (split in relation (k-1):1, where k is number of folds), as well as
            y train and test in same relation
    """
    n = len(X)
    # indexes of data for train and test in iteration i, converted to list later
    indexes_to_test = set(range(n * (i - 1) // k, (n * i // k))) # batches of k data
    indexes_to_train = list(set(range(X.shape[0])) - indexes_to_test)
    indexes_to_test = list(indexes_to_test)

    return X[indexes_to_train],X[indexes_to_test],y[indexes_to_train],y[indexes_to_test]



def CA(real, predictions):
    # ... dopolnite (naloga 3)
    # predictions are pairs of probabilities for being: [1, 0]
    # real[i] is just a number if it is 1 or 0
    # print(range(len(real)))
    # print("real:", real)
    # print("predictions:", predictions)

    errors = [1 if (predictions[i][0] > 0.5 and real[i] == 0) or (predictions[i][1] > 0.5 and real[i] == 1) else 0 for i in range(len(real))]
    CA = sum(errors) / len(real)
    # print("errors:", errors, "")
    # print("CA:", CA, "\n\n")
    return CA
    # return sqrt(sum([(e1-e2)**2 for e1, e2 in zip(real, predictions)]) / len(real))

def AUC(real, predictions):
    # ... dopolnite (naloga 4)

    print("real:", real)
    pred = [x[1] for x in predictions] # get only second element in the tuple
    print("pred:", pred)
    indeces = np.argsort(pred)
    # print("indeces: ", indeces)

    pred_sorted = [pred[i] for i in indeces]
    pred_sorted = pred_sorted[::-1] # reverse so the higher "probabilities" are higher
    print("pred_sorted: ", pred_sorted)

    real_sorted = [real[i] for i in indeces]
    real_sorted = real_sorted[::-1]
    print("real_sorted: ", real_sorted)

    # filter data out (same probabilities with different "correct" classes should be skipped)
    false_data_indeces = set()
    for i,(x,y) in enumerate(zip(pred_sorted, real_sorted)):
        indeces = [i for i, j in enumerate(pred_sorted) if j == x] # find other occurences of certain probability
        classes = [real_sorted[i] for i in indeces]
        if 1 in classes and 0 in classes:
            false_data_indeces.update(indeces) # add indeces of false data to the set

    print("false_data_indeces:", false_data_indeces)
    all_ind = set(list(range(0, len(real_sorted))))
    correct_data_indeces = all_ind - false_data_indeces

    real_s = [real_sorted[i] for i in correct_data_indeces]
    pred_s = [pred_sorted[i] for i in correct_data_indeces]
    # real_s = real_sorted
    # pred_s = pred_sorted

    num_ones = 0
    num_zeros = 0
    for i in real_s:
        if i == 1:
            num_ones += 1
        else:
            num_zeros += 1

    print("num_ones: {}, num_zeros {}".format(num_ones, num_zeros))

    zeros_temp = num_zeros # remaining number of zeros
    stevec = 0
    for x, i in enumerate(real_s):
        print(real_s[x], pred_s[x])
        if i == 1:
            stevec += zeros_temp
        else:
            zeros_temp -= 1

    rez = stevec / ((len(real_s)/2)*(len(real_s)/2))

    print("stevec", stevec)
    print("len(real_sorted)/2:", len(real_s)/2)

    return rez


def del2(lamb=0.001):
    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=lamb)
    print(test_cv(learner, X, y, 5))

    classifier = learner(X, y)  # dobimo model
    draw_decision(X, y, classifier, 0, 1)
    pass


def del3():
    X, y = load('reg.data')
    learner = LogRegLearner(lambda_=0.0)
    res = test_cv(learner, X, y, k=5)
    print(res)
    return res


def del4():
    X, real = load('reg.data')
    learner = LogRegLearner(lambda_=0)
    predictions = test_cv(learner, X, real, 5)

    print("real:", real)
    pred = [x[0] for x in predictions] # get only second element in the tuple
    print("pred:", pred)
    indeces = np.argsort(pred)
    # print("indeces: ", indeces)

    pred_sorted = [pred[i] for i in indeces]
    pred_sorted = pred_sorted[::-1]  # reverse so the higher "probabilities" are higher
    print("pred_sorted: ", pred_sorted)

    real_sorted = [real[i] for i in indeces]
    real_sorted = real_sorted[::-1]
    print("real_sorted: ", real_sorted)

    # pred_sorted, real_sorted
    ROC(real_sorted, pred_sorted)

    pass


def del5():
    # ... dopolnite
    pass

def lambde():
    X, y = load('reg.data')

    for i in range(-5,5):
        lm = float(math.pow(10.0,i))
        learner = LogRegLearner(lambda_=lm)

        res = test_learning(learner,X,y)
        accuracyCA = CA(y,res)

        res_cv = test_cv(learner,X,y)
        accuracyCV = CA(y, res_cv)

        print("lambda: {}, & accuracy_CA: {}, accuracy_CV: {}".format(lm, accuracyCA, accuracyCV))

def ROC(real, predictions):

    plt.axis([0, 1, 0, 1])
    num_ones = 0
    num_zeros = 0
    for i in real:
        if i == 1:
            num_ones += 1
        else:
            num_zeros += 1

    print("num_ones: {}, num_zeros {}".format(num_ones, num_zeros))

    korak_gor = 1/num_ones
    korak_desno = 1/num_zeros

    x = 0
    y = 0
    for p in real:
        if p == 1:
            y += korak_gor
        else:
            x += korak_desno
        plt.scatter(y,x)

    plt.show()
    pass

if __name__ == "__main__":
    # Primer uporabe, ki bo delal, ko implementirate cost in grad
    del4()

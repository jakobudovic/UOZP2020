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
    # print("np.array(l): ", np.array(l))
    return np.array(l)


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    # TODO
    l = []
    for i, e in enumerate(theta):
        l.append((1 / len(y)) * sum([(h(x, theta) - yi) * x[i] for x, yi in zip(X, y)]) + 2 * lambda_ * e)
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

        """
        pred = test_learning(learner, X_train, y_train) # we create classifier in here and get results on train datar
        # print("pred:", pred)
        predictions = predictions + [pred]
        """

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
    print("predictions:", predictions)
    pred = [x[1] for x in predictions] # get only second element in the tuple
    print("pred:", pred)
    indeces = np.argsort(pred)
    print("indeces: ", indeces)

    pred_sorted = [pred[i] for i in indeces]
    pred_sorted = pred_sorted[::-1] # reverse so the higher "probabilities" are higher
    print("arr_sorted: ", pred_sorted)

    real_sorted = [real[i] for i in indeces]
    real_sorted = real_sorted[::-1]
    print("real_sorted: ", real_sorted)

    rng = range(len(real))

    arr = np.array([100, 500, 300, 200, 400])
    ar = np.argsort(arr)

    num_ones = 0
    num_zeros = 0
    for i in real_sorted:
        if i == 1:
            num_ones += 1
        else:
            num_zeros += 1

    print("num_ones: {}, num_zeros {}".format(num_ones, num_zeros))

    ones_temp = num_ones
    zeros_temp = num_zeros # remaining number of zeros
    stevec = 0
    for i in real_sorted:
        if i == 1:
            stevec += zeros_temp
            # ones_temp -= 1
        else:
            zeros_temp -= 1

    rez = stevec / ((len(real_sorted)/2)*(len(real_sorted)/2))

    print("stevec", stevec)
    print("len(real_sorted)/2:", len(real_sorted)/2)
    # for i in rng:   # linearno gremo čez vse primere
    #     for j in range(i + 1, len(real)):   # pregledamo od nekega primera naprej, koliko enk pokriva
    #        print(i, j)


    return rez


def del2():
    # ... dopolnite
    pass


def del3():
    X, y = load('reg.data')
    print(X.shape, y.shape)
    learner = LogRegLearner(lambda_=0.0)
    res = test_cv(learner, X, y, k=5)
    print(res)
    return res


def del4():
    X, y = load('reg.data')


    pass


def del5():
    # ... dopolnite
    pass

def lambde():
    X, y = load('reg.data')

    for i in range(-10,10):
        lm = float(math.pow(10.0,i))
        learner = LogRegLearner(lambda_=lm)
        res_cv = test_cv(learner,X,y)
        res = test_learning(learner,X,y)
        accuracy_cv = CA(y,res_cv)
        accuracy = CA(y,res)

        print("lambda: %f, $10^%d$ & %f  & %f \\\\" % (lm, i,accuracy,accuracy_cv))

if __name__ == "__main__":
    # Primer uporabe, ki bo delal, ko implementirate cost in grad
    del4()

    arr = np.array([100, 500, 300, 200, 400])
    ar = np.argsort(arr)
    arr_sorted = [arr[i] for i in ar]

    print("arr:", arr)
    print("ar:", ar)
    print("arr_sorted", arr_sorted)
    """
    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    print(test_cv(learner, X, y, 5))

    classifier = learner(X, y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print("napoved:", napoved)

    # izris odlocitev
    # draw_decision(X, y, classifier, 0, 1)

    # odpiranje GDS350
    X, y = load('GDS360.data')
    print(X.shape, y.shape)
    print(X, y)

    print("-------------------------------")
    # lambde()
    k = 5
    for i in range(1, k + 1):
        X_train, X_test, y_train, y_test = k_fold(X, y, i, k)
        classifier = learner(X_train, y_train)
        # predictions = predictions + [classifier(row) for row in X_test]
    """

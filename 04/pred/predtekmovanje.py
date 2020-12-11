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
import lpputils as lpp

from sklearn.linear_model import LinearRegression

def duration(data):
    """
    duration of the bus route
    :param data: matrix of departure and arrival dates (including other useless data)
    :return:
     """
    y = []
    for d in data:
        start = lpp.parsedate(d[6])
        if (start.month in (1, 11, 12)):
        # if (start.month):
            arrival = d[8]
            departure = d[6]
            y.append(lpp.diff_dates(arrival, departure))
    return y

def join_day_time(data, prazniki):
    """
    join day and time of one departure to one 1D vector
    :param data: Dates in non-parsed format
    :return: logical matrix with 2 ones in each row;
            - first one for departure hour
            - second one for departure day of the week
    """
    X = []
    for d in data:
        example = [0] * 31 # 24 + 7 + 1 = 32

        departure = lpp.parsedate(d[6]) # departure time

        if (departure.month in (1, 11, 12)):  # november and december only (both winter)
            example[departure.hour] = 1  # first 24 slots are for deprature hours
            example[departure.isoweekday() + 23] = 1 # last 7 slots are for days
            # if departure.month in prazniki_matrix[0, :] and departure.day in prazniki_matrix[1, :]:
                # print("praznik: ", departure.month, departure.day)
                # example[31] = 1 # last slot is for work - free - days
            X.append(example)

    return X

def read_file(file_path):
    os.chdir("/home/jakob/git/UOZP2020/04/pred")    # select working dir manually
    csvreader = csv.reader(open(file_path), delimiter='\t')
    headers = next(csvreader) # remove header
    list = [d for d in csvreader]
    return np.array(list)

if __name__ == "__main__":
    X_train = read_file('./train_pred.csv')
    X_test = read_file('./test_pred.csv')

    reader = csv.reader(open('prazniki.csv'), delimiter='\t')
    prazniki = [d for d in reader]

    prazniki_matrix = np.array([[int(praznik[0].split("-")[0]), int(praznik[0].split("-")[1])] for praznik in prazniki])

    X = np.vstack(join_day_time(X_train, prazniki_matrix)) # convert from list to ndarray
    y = np.array(duration(X_train))

    X_test_matrix = np.vstack(join_day_time(X_test, prazniki_matrix)) # matrix for testing and making predictions

    # build our model
    lin = linear.LinearLearner(lambda_=1.)
    prediction_model = lin(X,y) # feed/train our model

    # build our model from library
    LR = LinearRegression()
    model = LR.fit(X, y)
    print("model accuracy: ", model.score(X,y))
    rez_sk = LR.predict(X)

    result = [prediction_model(ex) for ex in X_test_matrix]

    fo = open("rez.txt", "wt")
    for l, e in zip(rez_sk, X_test):
        fo.write(lpp.add_seconds(e[6], l) + "\n")
    fo.close()


    """
    date = "2020-12-7 16:32:01.000"
    date_ = "2012-01-13 12:27:04.000"
    date1 = lpp.parsedate(date_)
    print(date1.isoweekday())
    print(date1.hour)
    print(date1)
    """


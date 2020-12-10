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
import time

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

def join_day_time(data, prazniki, test=False):
    """
    join day and time of one departure to one 1D vector
    :param data: 1D arrays of examples data, prazniki, boolean test: if data is train/test (test does not have arrival time)
    :return: dict with routes as keys and 2D array as value
            - arr1 : 2D one-hot-encoded data, rows are examples
            - arr2 : 1D array of durations
    """
    X_dict = {}
    for d in data:
        route_name = d[3]
        example = [0] * 31 # 24 + 7 + 1 = 32
        arrival = d[8]
        departure = d[6]
        duration = lpp.diff_dates(arrival, departure)

        if duration < 150 or duration > 9000:   # routes less than 3 min or more than 3h
            print('Bus {0} on route {1} from {2} to {3} lasted {4} seconds'.format(d[2], d[3], d[6], d[8], duration))
            continue

        departure = lpp.parsedate(d[6]) # departure time without 0's at the end

        if (departure.month in (1, 11, 12)):  # november and december only (both winter)

            if route_name not in X_dict.keys():  # create new 2 arrays for route, if it they not exist yet (for data and duration)
                X_dict[route_name] = [[], []]

            example[departure.hour] = 1  # first 24 slots are for deprature hours
            example[departure.isoweekday() + 23] = 1 # last 7 slots are for daysž

            # calculate duration of example:
            X_dict[route_name][0].append(example) # append our one-hot-encoded example to correct hash
            X_dict[route_name][1].append(duration)  # append our one-hot-encoded example to correct hash
    return X_dict

def read_file(file_path):
    os.chdir("/home/jakob/git/UOZP2020/04/tekm")    # select working dir manually
    csvreader = csv.reader(open(file_path), delimiter='\t')
    headers = next(csvreader) # remove header
    list = [d for d in csvreader]
    return np.array(list)

def print_res(rez_sk, X_test):
    fo = open("rez.txt", "wt")
    for l, e in zip(rez_sk, X_test):
        fo.write(lpp.add_seconds(e[6], l) + "\n")
    fo.close()

def read_data(train_path, test_path):
    """
    :param train_path: train csv file
    :param test_path: test csv file
    :return:
            - X: dict with routes as keys and 2D vector of encoded data and durations as value
            - X_test_matrix: data from test.csv prepared to be predicted on
            - X_test: raw data from test.csv (temporary)
    """
    X_train = read_file(train_path)
    X_test = read_file(test_path)
    reader = csv.reader(open('prazniki.csv'), delimiter='\t')
    prazniki = [d for d in reader]

    prazniki_matrix = np.array([[int(praznik[0].split("-")[0]), int(praznik[0].split("-")[1])] for praznik in prazniki])

    # create a dict
    # X = np.vstack() # convert from list to ndarray
    X = join_day_time(X_train, prazniki_matrix)
    # y = np.array(duration(X_train))

    # old: # X_test_matrix = np.vstack(join_day_time(X_test, prazniki_matrix)) # matrix for testing and making predictions
    # X_test_matrix = join_day_time_test(X_test, prazniki_matrix) # matrix for testing and making predictions
    return X, X_test

def return_models(X, y):
    """
    :param X: test data, one-hot encoded vectors 
    :param y: output value, 1D decimal array
    :return: models trained with linear.py and sklearn, prediction (decimal) on train data
    """
    # build our model with linear.py, for each route

    lin = linear.LinearLearner(lambda_=1.)
    model_lin = lin(X, y)  # feed/train our model

    # build our model from library sklearn
    LR = LinearRegression()
    model_sk = LR.fit(X, y)
    print("model accuracy: ", model_sk.score(X,y)) # test model accuracy
    ## rez_sk = LR.predict(X)

    return model_lin, model_sk # , rez_sk

def build_models(X, models_lin, models_sk):
    """
    :param X: data one-hot-encoded
    :param models_lin: empty dict that we fill up with linear models
    :param models_sk: empty dict that we fill up with linear models from SKlearn lib
    :return: 2 dicts of models assigned to correct keys
    """
    for key, value in X.items():
        if key not in models_lin.keys():
            models_lin[key] = {}
            models_sk[key] = {}

        X = np.vstack(value[0])
        y = np.asarray(value[1])

        model_lin, model_sk = return_models(X, y)
        models_lin[key] = model_lin
        models_sk[key] = model_sk

    return models_lin, models_sk

def encode_example(example):
    """
    :param example: not-encoded example
    :return: one-hot-encoded example
    """
    example_encoded = [0] * 31
    departure = lpp.parsedate(example[6])  # departure time without 0's at the end
    example_encoded[departure.hour] = 1  # first 24 slots are for deprature hours
    example_encoded[departure.isoweekday() + 23] = 1  # last 7 slots are for daysž
    return example_encoded

if __name__ == "__main__":
    start_time = time.time()
    # return dict with routes as keys and values as dict of one-hot encoded data matrix and last vector being times
    X, X_test = read_data('./train_short.csv', './test_short.csv')

    # build dict of models for each route
    models_lin, models_sk = build_models(X, {}, {})

    # make predictions by transforming each row in test.csv to correct array first
    rez_lin = []
    rez_sk = []
    rez_lin, rez_sk = get_results()
    for example in X_test:
        route = example[3]
        if route not in models_lin.keys():
            # print("model for route {0} is not built!".format(route))
            continue
        # print("model built")
        model_lin = models_lin[route]
        model_sk = models_sk[route]

        example_encoded = encode_example(example)
        rez_lin.append(model_lin(example_encoded))
        rez_sk.append(model_sk.predict([example_encoded]))
    rez_sk = [rez[0] for rez in np.asarray(rez_sk)] # convert sklearn results to list




    # , X_test_matrix, X_test
    # build our model
    # model_lin, model_sk, rez_sk = build_models(X, y)
    #
    # rez_lin = [model_lin(ex) for ex in X_test_matrix]
    # # rez_sk = [model_sk(ex) for ex in X_test_matrix]
    #
    # # print results, obtained from Sklearn lib or linear.py:
    # # rez_sk / rez_lin
    # print("--- %s seconds ---" % (time.time() - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))
    #
    # print_res(rez_sk, X_test)

    """
    date = "2020-12-7 16:32:01.000"
    date_ = "2012-01-13 12:27:04.000"
    date1 = lpp.parsedate(date_)
    print(date1.isoweekday())
    print(date1.hour)
    print(date1)
    """


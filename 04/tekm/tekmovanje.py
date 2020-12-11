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

def join_day_time(data, prazniki, sola, test=False):
    """
    join day and time of one departure to one 1D vector
    :param data: 1D arrays of examples data, prazniki, boolean test: if data is train/test (test does not have arrival time)
    :return: dict with routes as keys and 2D array as value
            - arr1 : 2D one-hot-encoded data, rows are examples
            - arr2 : 1D array of durations
    """
    X_dict = {}
    enka_debug = []
    for d in data:
        route_direction = d[3]
        route_name_description = d[4]
        first_station = d[5]
        route_name = route_direction + "#" + route_name_description + "#" + first_station
        if route_name not in enka_debug and (route_direction == "MESTNI LOG - VIŽMARJE" or route_direction == "VIŽMARJE - MESTNI LOG"):
            enka_debug.append(route_name)
        example = [0] * 33 # 23 + 3 + 6 + 1 = 33
        arrival = d[8]
        departure = d[6]
        # calculate duration of example:
        duration = lpp.diff_dates(arrival, departure)

        if duration < 150 or duration > 10000:   # routes less than 3 min or more than 3h
            print('Bus {0} on route {1} from {2} to {3} traveled {4} seconds'.format(d[2], d[3], d[6], d[8], duration))
            continue

        departure = lpp.parsedate(d[6]) # departure time without 0's at the end

        if (departure.month in (1, 11, 12)):  # november and december only (both winter)

            if route_name not in X_dict.keys():  # create new 2 arrays for route, if it they not exist yet (for data and duration)
                X_dict[route_name] = [[], []]

            departure_encoded = departure.minute // 15 # encode quarters of hour into 4 possible values
            # print(departure, departure.minute, departure.minute // 15)

            if departure.hour != 0:
                example[departure.hour - 1] = 1  # first 0-22 slots are for deprature hours
            if (departure.minute // 15) !=  0:
                example[((departure.minute // 15) - 1)+ 23] = 1  # 23, 24 and 25th slot for time quartils
            if departure.isoweekday() != 1:
                example[(departure.isoweekday() - 2) + 26] = 1 # 26-31 slots are for days.

            datum = "" + str(departure.day) + "," + str(departure.month)
            if [datum] in sola:
                example[32] = 1

            X_dict[route_name][0].append(example) # append our one-hot-encoded example to correct hash
            X_dict[route_name][1].append(duration)  # append our duration for this example to correct hash
    print("vse poti enke ki jih natreniram: -----------------------------")
    print(enka_debug)
    for r in enka_debug:
        print(r)
    print("=============")
    return X_dict

def encode_example(example, sola):
    """
    :param example: not-encoded example
    :return: one-hot-encoded example
    """
    example_encoded = [0] * 33
    departure = lpp.parsedate(example[6])  # departure time without 0's at the end

    if departure.hour != 0:
        example_encoded[departure.hour - 1] = 1  # first 0-22 slots are for deprature hours
    if (departure.minute // 15) != 0:
        example_encoded[((departure.minute // 15) - 1) + 23] = 1  # 23, 24 and 25th slot for time quartils
    if departure.isoweekday() != 1:
        example_encoded[(departure.isoweekday() - 2) + 26] = 1  # 26-31 slots are for days.

    datum = "" + str(departure.day) + "," + str(departure.month)
    if [datum] in sola:
        example_encoded[32] = 1

    # example_encoded[departure.hour] = 1  # first 24 slots are for deprature hours
    # example_encoded[departure.isoweekday() + 23] = 1  # last 7 slots are for days
    return example_encoded

def read_file(file_path):
    os.chdir("/home/jakob/git/UOZP2020/04/tekm")    # select working dir manually
    csvreader = csv.reader(open(file_path), delimiter='\t')
    headers = next(csvreader) # remove header
    print(headers)
    list = [d for d in csvreader]
    return np.array(list)

def read_data(train_path, test_path):
    """
    :param train_path: train csv file
    :param test_path: test csv file
    :return:
            - X: dict with routes as keys and 2D vector of encoded data and durations as value
            - X_test: raw data from test.csv in rows
    """
    X_train = read_file(train_path)
    X_test = read_file(test_path)
    reader = csv.reader(open('prazniki.csv'), delimiter='\t')
    prazniki = [d for d in reader]

    reader = csv.reader(open('sola.csv'), delimiter='\t')
    sola = [d for d in reader]

    prazniki_matrix = np.array([[int(praznik[0].split("-")[0]), int(praznik[0].split("-")[1])] for praznik in prazniki])
    X = join_day_time(X_train, prazniki_matrix, sola)
    return X, X_test, sola

def return_models(X, y):
    """
    :param X: test data, one-hot encoded vectors 
    :param y: output value, 1D decimal array
    :return: models trained with linear.py and sklearn, prediction (decimal) on train data, to estimate accuracy
    """
    # build our model with linear.py, for each route
    lin = linear.LinearLearner(lambda_=1.)
    model_lin = lin(X, y)  # feed/train our model

    # build our model from library sklearn
    LR = LinearRegression()
    model_sk = LR.fit(X, y)
    model_acc = model_sk.score(X,y)
    # print("model accuracy: ", model_acc) # test model accuracy
    ## rez_sk = LR.predict(X)

    return model_lin, model_sk, model_acc # , rez_sk

def build_models(X, models_lin, models_sk):
    """
    :param X: one-hot-encoded data
    :param models_lin: empty dict that we fill up with linear models
    :param models_sk: empty dict that we fill up with linear models from SKlearn lib
    :return: 2 dicts of built models for each route with linear.py and lib
    """
    model_accuracies = []
    for key, value in X.items():
        if key not in models_lin.keys():
            models_lin[key] = {}
            models_sk[key] = {}

        X = np.vstack(value[0])
        y = np.asarray(value[1])

        model_lin, model_sk, model_acc = return_models(X, y)
        models_lin[key] = model_lin
        models_sk[key] = model_sk
        model_accuracies.append(model_acc)
    print("All models accuracy: {0}".format(sum(model_accuracies)/len(model_accuracies)))
    return models_lin, models_sk

def get_results(X_test, rez_lin, rez_sk, sola):
    """
    :param X_test: not-encoded data, rows are examples, we make predictions on
    :param rez_lin: empty list
    :param rez_sk: empty list
    :return:
    """
    missing_models = []
    for d in X_test:
        route_direction = d[3]
        route_name_description = d[4]
        first_station = d[5]
        route_name = route_direction + "#" + route_name_description + "#" + first_station

        # hardcoded fixes for missing models
        if route_name == "MESTNI LOG - VIŽMARJE#  VIŽMARJE; sejem#Tbilisijska":
            route_name = "MESTNI LOG - VIŽMARJE#  VIŽMARJE#Koprska"
        elif route_name == "VIŽMARJE - MESTNI LOG#  MESTNI LOG; sejem#Šentvid":
            route_name = "VIŽMARJE - MESTNI LOG#  MESTNI LOG#Šentvid"

        if route_name not in models_lin.keys():
            if route_name not in missing_models:
                missing_models.append(route_name)
                print("model for route {0} is not built!".format(route_name))
            continue
        # print("model built")
        model_lin = models_lin[route_name]
        model_sk = models_sk[route_name]

        example_encoded = encode_example(d, sola)
        rez_lin.append(round(model_lin(np.asarray(example_encoded)), 3))
        rez_sk.append(model_sk.predict([example_encoded]))
    rez_sk = [rez[0] for rez in np.asarray(rez_sk)] # convert sklearn results to list
    print("missing_models: ", missing_models)
    return rez_lin, rez_sk

def print_rez(rez, X_test, file):
    fo = open(file, "wt")
    for l, e in zip(rez, X_test):
        fo.write(lpp.add_seconds(e[6], l) + "\n")
    fo.close()

if __name__ == "__main__":
    start_time = time.time()
    # return dict with routes as keys and values as dict of one-hot encoded data matrix and last vector being times
    s = './train_short.csv'
    l = './train.csv'
    s1 = './test_short.csv'
    l1 = './test.csv'
    X, X_test, sola = read_data(l, l1)
    models_lin, models_sk = build_models(X, {}, {})    # build dict of models for each route
    rez_lin, rez_sk = get_results(X_test, [], [], sola)
    print_rez(rez_lin, X_test, "rez1.txt")

    print("--- %s seconds ---" % (time.time() - start_time))

    """
    date = "2020-12-7 16:32:01.000"
    date_ = "2012-01-13 12:27:04.000"
    date1 = lpp.parsedate(date_)
    print(date1.isoweekday())
    print(date1.hour)
    print(date1)
    """


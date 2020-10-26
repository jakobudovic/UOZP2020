import csv
import numpy as np
import pandas as pd
import copy
import sys
import matplotlib.pyplot as plt
from random import randint


# function to read file and store data in dictionary
def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    f = open(file_name, "rt")
    values = csv.reader(f)
    header_line = next(values)  # skip header cell

    values = np.array(list(values))  # values used for unique countries extraction
    values_all = copy.deepcopy(values)  # deepcopy values to save them for later use
    distinct_countries = list(set(np.array(list(values))[:, 2]))  # extracting unique countries
    distinct_countries.sort()  # sorting countries so keys will be in alphabetical order

    # new dictionary, keys are unique countries ordered in alph. order, values of dict are vectors
    # with the length of all distinct countries

    dic = {}
    for country in distinct_countries:
        dic[country] = [0] * len(distinct_countries)
    # dic = dict.fromkeys(distinct_countries, [0]*len(distinct_countries)) # this didnt work and took me half a day :(

    for l in values_all:
        giver = l[2]
        receiver = l[3]
        idx = distinct_countries.index(receiver)  # index where we will add the points l[4]

        # we add points to the giver-country's array at an index of the country receiving, obdatined from
        # the sorted array of distinct countries.
        dic[giver][idx] += int(l[4])

    # write max numbers of voting on the diagonal
    # Country favorizes itself, so a max value would make more sense to be there instead of a min (0)
    for k,v in dic.items():
        idx = list(dic.keys()).index(k) # we store an index so we see how "far" the diagonal is in the values (matrix)
        dic[k][idx] = max(v) # assign value to the diagonal

    occurences = {}
    for l in values:
        country = l[2]
        year = l[0]
        occurences.setdefault(country, set()).add(year)

    for key, value in occurences.items():
        if len(value) < 6: # remove countries with 5 or less occurrences
            dic.pop(key)

    return dic

# used to unwrap arrays in arrays
def unwrap(arr):
    while (len(arr) == 1):
        arr = arr[0]
    return arr

def merge_clusters(clusters, c1, c2):
    # indexes where clusters c1 & c2 are stored in clusters arr
    idx1 = clusters.index(c1)
    clusters[idx1] = [clusters[idx1]]  # create double array to prepare to insert another array into it
    clusters[idx1].insert(len(clusters[idx1]) - 1, c2)  # insert c2 to the end of the c1
    clusters.remove(c2)  # remove cluster c2

# flatten an array that you don't know the dimensions of
def flatten_rec(arr, new_arr):
    if len(arr) == 1:
        new_arr.append(arr)
    else:
        for l in arr:
            flatten_rec(l, new_arr)

def izris(arr, i):
    if len(arr) == 1:
        print("    " * i, "---- ", arr[0], sep='')
    else:
        i += 1
        izris(arr[0], i)
        print("    " * (i - 1), "----|", sep='')
        izris(arr[1], i)


def get_xpos(arr, out):
    global index
    if len(arr) == 1:
        out[arr[0]] = index
        index = index+1
    else:
        get_xpos(arr[0], out)
        get_xpos(arr[1], out)

class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]
        self.countries = [[name] for name in self.data.keys()] # save for later, plotting graph

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        r1, r2 = string, string
        """
        v1 = self.data[r1]
        v2 = self.data[r2]

        e = sum((p - q) ** 2 for p, q in zip(v1, v2)) ** .5
        return e

    def cluster_distance(self, c1, c2):
        """
        Compute distance between two clusters.
        Implement either single, complete, or average linkage.
        Example call: self.cluster_distance(
            [[["Albert"], ["Branka"]], ["Cene"]],
            [["Nika"], ["Polona"]])
        """
        # We will use average linkage

        dist = 0
        c1_flat = []
        c2_flat = []
        # we flatten 2 arrays
        flatten_rec(c1, c1_flat)
        flatten_rec(c2, c2_flat)

        for c in c1_flat:
            for d in c2_flat:
                # distance between c & d
                dist += self.row_distance(c[0], d[0])
        dist /= (len(c1_flat) * len(c2_flat))
        return dist

    def closest_clusters(self):
        """
        Find a pair of closest clusters and returns the pair of clusters and
        their distance.

        Example call: self.closest_clusters(self.clusters)
        """
        min_dist = sys.maxsize
        # arrays of clusters
        min_c1 = []
        min_c2 = []
        for c1 in self.clusters:
            for c2 in self.clusters:
                if c1 is c2:
                    pass  # so we don't compare the same cluster to itself
                else:
                    dist = self.cluster_distance(c1, c2)
                    if dist < min_dist:
                        min_dist = dist
                        min_c1 = c1
                        min_c2 = c2
        return min_c1, min_c2, min_dist

    def run(self):
        """
        Given the data in self.data, performs hierarchical clustering.
        Can use a while loop, iteratively modify self.clusters and store
        information on which clusters were merged and what was the distance.
        Store this later information into a suitable structure to be used
        for plotting of the hierarchical clustering.
        """
        num_clusters = len(self.clusters)
        clusters = self.clusters
        distances = []
        locations = []
        # until we have more than 1 cluster
        while (num_clusters > 2):
            c1, c2, min_dist = self.closest_clusters()  # we get closest clusters and save them in c1, c2 & dist.
            idx1 = clusters.index(c1)
            idx2 = clusters.index(c2)
            locations.append((idx1, idx2)) # save locations of elements for later plotting
            merge_clusters(clusters, c1, c2)
            self.clusters = unwrap(clusters)
            num_clusters = len(self.clusters)
            distances.append(min_dist)

        self.distances = distances
        self.locations = locations

        print("distances:", distances)
        print("locations:", locations)

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        print("self.clusters:", self.clusters)
        izris(self.clusters, 0)
        pass

    def plot_graph(self):

        label_map = {
            0: {'label': 'BA', 'xpos': 0, 'ypos': 0},
            1: {'label': 'FI', 'xpos': 3, 'ypos': 0},
            2: {'label': 'MI', 'xpos': 4, 'ypos': 0},
            3: {'label': 'NA', 'xpos': 1, 'ypos': 0},
            4: {'label': 'RM', 'xpos': 2, 'ypos': 0},
            5: {'label': 'TO', 'xpos': 5, 'ypos': 0},
        }
        print("countries:", self.countries)
        print("countries len:", len(self.countries))
        label_map = {}

        # init the dict
        for i,c in enumerate(self.countries):
            label_map[i] = {}
        # print("label_map:", label_map)
        """
        get indexes where countries should be on the map
        dict = {'Greece': 0, 'Cyprus': 1, 'Malta': 2, ... }
        get_xpos(arr, out, index)
        """
        dict = {}
        get_xpos(self.clusters, dict)
        print("dict:", dict)
        self.dict = dict

        for i, val in label_map.items():
            val["label"] = self.countries[i][0]
            val["xpos"] = dict[self.countries[i][0]] # get index where country in our dict of x coordinates is
            val["ypos"] = 0 # always 0

        self.label_map = label_map
        print("label_map:", label_map)


        self.plot()

        pass

    def plot(self):
        """
        Source: https://stackoverflow.com/questions/56123380/how-to-draw-dendrogram-in-matplotlib-without-using-scipy
        levels = [138, 219, 255, 268, 295]
        locations = [(2, 5), (3, 4), (0, 3), (0, 1), (0, 2)]

        label_map = {
            0: {'label': 'BA', 'xpos': 0, 'ypos': 0},
            1: {'label': 'FI', 'xpos': 3, 'ypos': 0},
            2: {'label': 'MI', 'xpos': 4, 'ypos': 0},
            3: {'label': 'NA', 'xpos': 1, 'ypos': 0},
            4: {'label': 'RM', 'xpos': 2, 'ypos': 0},
            5: {'label': 'TO', 'xpos': 5, 'ypos': 0},
        }
        """
        max_level = max(self.distances)
        levels = [l / max_level for l in self.distances] # normalize
        locations = self.locations
        label_map = self.label_map

        fig, ax = plt.subplots()
        # plt.figure(figsize=(4, 12))

        for i, (new_level, (loc0, loc1)) in enumerate(zip(levels, locations)):
            print('step {0}:\t connecting ({1},{2}) at level {3}'.format(i, loc0, loc1, new_level))

            x0, y0 = label_map[loc0]['xpos'], label_map[loc0]['ypos']
            x1, y1 = label_map[loc1]['xpos'], label_map[loc1]['ypos']

            print('\t points are: {0}:({2},{3}) and {1}:({4},{5})'.format(loc0, loc1, x0, y0, x1, y1))

            p, c = mk_fork(x0, x1, y0, y1, new_level)

            ax.plot(*p)
            ax.scatter(*c)

            aa = randint(7, 12)
            for h in range(randint(0, 10)): # test, delete it
                aa = aa**h


            print('\t connector is at:{0}'.format(c))

            label_map[loc0]['xpos'] = c[0]
            label_map[loc0]['ypos'] = c[1]
            label_map[loc0]['label'] = '{0}/{1}'.format(label_map[loc0]['label'], label_map[loc1]['label'])
            print('\t updating label_map[{0}]:{1}'.format(loc0, label_map[loc0]))

            # ax.text(*c, label_map[loc0]['label'])

        _xticks = np.arange(0, len(self.countries), 1)
        _xticklabels = self.dict.keys()

        ax.set_xticks(_xticks)
        ax.set_xticklabels(_xticklabels, rotation=90, fontsize=7)

        ax.set_ylim(0, 1.05 * np.max(levels))

        plt.show()

def mk_fork(x0,x1,y0,y1,new_level):
    points=[[x0,x0,x1,x1],[y0,new_level,new_level,y1]]
    connector=[(x0+x1)/2.,new_level]
    return (points),connector

if __name__ == "__main__":
    index = 0
    DATA_FILE = "eurovision-finals-1975-2019.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    hc.plot_tree()
    hc.plot_graph()

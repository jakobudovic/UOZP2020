import csv
import numpy as np
import pandas as pd
import copy
import sys

# function to read file and store data in dictionary
def read_file(file_name):
    """
    Read and process data to be used for clustering.
    :param file_name: name of the file containing the data
    :return: dictionary with element names as keys and feature vectors as values
    """
    f = open(file_name, "rt", encoding="utf8")
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


class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        print(data)
        # self.clusters stores current clustering. It starts as a list of lists
        # of single elements, but then evolves into clusterings of the type
        # [[["Albert"], [["Branka"], ["Cene"]]], [["Nika"], ["Polona"]]]
        self.clusters = [[name] for name in self.data.keys()]
        print(self.clusters)

    def row_distance(self, r1, r2):
        """
        Distance between two rows.
        Implement either Euclidean or Manhattan distance.
        Example call: self.row_distance("Polona", "Rajko")
        """

        v1 = self.data[r1[0]]
        v2 = self.data[r2[0]]

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
                dist += self.row_distance(c, d)
        dist /= (len(c1) * len(c2))
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
        print(num_clusters)
        distances = []
        # until we have more than 1 cluster
        while (num_clusters > 2):
            c1, c2, min_dist = self.closest_clusters()  # we get closest clusters and save them in c1, c2 & dist.
            merge_clusters(clusters, c1, c2)
            self.clusters = unwrap(clusters)
            num_clusters = len(self.clusters)
            distances.append(min_dist)
        print("clusters final:")
        print(self.clusters)
        print("distances:")
        print(distances)

    def plot_tree(self):
        """
        Use cluster information to plot an ASCII representation of the cluster
        tree.
        """
        pass

if __name__ == "__main__":
    DATA_FILE = "eurovision-finals-1975-2019.csv"
    hc = HierarchicalClustering(read_file(DATA_FILE))
    hc.run()
    # hc.plot_tree()

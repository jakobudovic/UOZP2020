import sys
import time
from math import sqrt
from os import listdir
from os.path import join
from os import walk
import os
import re
from transliterate import translit, get_available_language_codes
from itertools import combinations
import copy

def kmers(s, k=3):
    """Generates k-mers for an input string."""
    arr = []
    for i in range(len(s)-k+1):
        str = s[i:i + k]
        # if re.match('^[a-z]*[\ .]*[\ .\ ]*[a-z]*$', str):
        if re.match('^[a-z]*[\ .]*[a-z]*$', str):
            arr.append(s[i:i+k])
    return arr

def terke(text, n=4):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    arr = kmers(text, n)
    # print(arr)
    dic = {}
    for key in arr:
        if key not in dic:
            dic[key] = 1
        else:
            dic[key] += 1
    return dic                  # return sorted dic with strings as keys

def read_clustering_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    print("Texts: ", listdir("clustering"))
    for fn in listdir("clustering"):
    # for fn in ['ww_bs.txt', 'ww_ru.txt']:
        if fn.lower().endswith(".txt"):
            with open(join("clustering", fn), encoding="utf8") as f:
                text = f.read()
                # nter = terke(translit(f.read().lower(), reversed=True), n=n_terke)
                # ['mn', 'ru', 'mk', 'sr', 'bg', 'hy', 'el', 'ka', 'l1', 'uk']
                if fn == "mac.txt":
                    nter = terke(translit(text.lower(), 'mk', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "rus.txt":
                    nter = terke(translit(text.lower(), 'ru', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "ser.txt":
                    nter = terke(translit(text.lower(), 'sr', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "bg.txt":
                    nter = terke(translit(text.lower(), 'bg', reversed=True), n=n_terke)
                    lds[fn] = nter
                else:
                    nter = terke(text.lower(), n=n_terke)
                    lds[fn] = nter
                    # nter = terke(text, n=n_terke)
    return lds


def read_prediction_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("prediction"):
        if fn.lower().endswith(".txt"):
            with open(join("prediction", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
    return lds


def cosine_dist(d1, d2):
    """
    Vrne kosinusno razdaljo med slovarjema terk d1 in d2.
    """

    # d1 = {'uses': 1, 'useu': 2, 'ush ': 3, 'ushe': 3, 'ushi': 3, 'usic': 1, 'usil': 2}
    # vectors A B
    # A * B = Ax * Bx + Ay * By + Az * Bz + ...
    # len: |A| = sqrt((A_x)² + (A_y)² + (A_z)²)

    # A * B
    ab = 0
    for ak, av in d1.items():  # akey and avalue
        if ak in d2.keys():     # we check if we have the same key in other dict also
            ab = ab + av * d2[ak]

    # |d1|
    dist1 = 0
    for v in d1.values():
        dist1 = dist1 + v**2
    dist1 = sqrt(dist1)

    # |d2|
    dist2 = 0
    for v in d1.values():
        dist2 = dist2 + v**2
    dist2 = sqrt(dist2)

    # print("ab={0}, dist1={1}, dist2={2}".format(ab, dist1, dist2))

    dist = ab / (dist1 * dist2)
    return dist


def k_medoids(data, medoids):
    """
    Za podane podatke (slovar slovarjev terk) in medoide vrne končne skupine
    kot seznam seznamov nizov (ključev v slovarju data).
    """
    seznam = []



    pass


def silhouette(data, clusters):
    """
    Za podane podatke (slovar slovarjev terk) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrne silhueto.
    """
    pass


def predict(data, text, n_terke):
    """
    Za podano bazo jezikov data za vsak jezik vrne verjetnost, da je besedilo text
    napisano v tem jeziku (izhod je v obliki slovarja).
    """
    pass

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# flatten an array that you don't know the dimensions of
def flatten_rec(arr, new_arr):
    if len(arr) == 1:
        new_arr.append(arr)
    else:
        for l in arr:
            flatten_rec(l, new_arr)

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

def izris(arr, i):
    if len(arr) == 1:
        print("    " * i, "---- ", arr[0], sep='')
    else:
        i += 1
        izris(arr[0], i)
        print("    " * (i - 1), "----|", sep='')
        izris(arr[1], i)

class HierarchicalClustering:
    def __init__(self, data):
        """Initialize the clustering"""
        self.data = data
        # print("data: ", sorted(list(data.keys())))
        # self.clusters = [[name] for name in self.data.keys()]
        self.clusters = [[name] for name in sorted(list(data.keys()))]
        self.clusters_indexed = [[name] for name in sorted(list(data.keys()))]

    def cluster_distance(self, c1, c2):
        dist = 0
        c1_flat = []
        c2_flat = []
        # we flatten 2 arrays
        flatten_rec(c1, c1_flat)
        flatten_rec(c2, c2_flat)

        for c in c1_flat:
            for d in c2_flat:
                # distance between c & d
                # dist += cosine_dist(self.data[c[0]], self.data[d[0]])
                try:
                    idx1 = self.clusters_indexed.index(c)
                    idx2 = self.clusters_indexed.index(d)
                except:
                    print(self.clusters)
                    print("c:", c)
                    print("d:", d)
                dist += self.dists_mtx[idx1][idx2]
        dist = dist / (len(c1_flat) * len(c2_flat))
        return dist

    def closest_clusters(self):
        max_similarity = 0
        # arrays of clusters
        min_c1 = []
        min_c2 = []
        for c1 in self.clusters:
            for c2 in self.clusters:
                if c1 is c2:
                    pass  # so we don't compare the same cluster to itself
                else:
                    dist = self.cluster_distance(c1, c2)
                    if dist > max_similarity:
                        max_similarity = dist
                        min_c1 = c1
                        min_c2 = c2
        return min_c1, min_c2, max_similarity

    def run(self):
        num_clusters = len(self.clusters)
        clusters = self.clusters
        print(clusters)

        dists_mtx = [[0] * num_clusters for i in range(num_clusters)]

        for i, c1 in enumerate(clusters):
            for j, c2 in enumerate(clusters):
                if j > i: # zgornje trikotna matrika
                    dist = cosine_dist(self.data[c1[0]], self.data[c2[0]])
                    # print("i;", i, ", j:", j, "dist: ", dist, "       countries: ", c1[0], c2[0])
                    dists_mtx[i][j] = dist
                    dists_mtx[j][i] = dist
        # print(dists)
        self.dists_mtx = dists_mtx


        ###
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
        print(self.distances)
        self.locations = locations


    def plot_tree(self):
        print("self.clusters:", self.clusters)
        izris(self.clusters, 0)
        pass

def del2():
    nterk = 6
    data = read_clustering_data(nterk)  # dolžino terk prilagodite
    hc = HierarchicalClustering(data)
    hc.run()
    hc.plot_tree()
    # hc.plot_graph()


def del4():
    data = read_clustering_data(3)  # dolžino terk prilagodite
    # ... nadaljujte


def del5():
    data = read_prediction_data(3)  # dolžino terk prilagodite
    # ... nadaljujte
    # primer klica predict: print(predict(data, "Danes je lep dan", 3))


if __name__ == "__main__":
    start_time = time.time()

    # file_name = "ww_en.txt"
    # file_name = "./clustering/ww_ser.txt"
    # f = open(file_name, "r")
    # print(listdir("clustering"))
    # dic = set(terke(translit(f.read().lower(), 'sr', reversed=True), 4))
    # print(dic)
    # print(len(dic))
    """
    lds = read_clustering_data(4)
    for a, b in combinations(lds.keys(), 2):
        dist = cosine_dist(lds[a], lds[b])
        print()
        print("dist between", a[3:6], "and", b[3:6], ":", dist)
    """
    # odkomenirajte del naloge, ki ga želite pognati
    del2()
    # del4()
    # del5()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("-- END --")
    pass


"""
rom - norw = 0.9



"""

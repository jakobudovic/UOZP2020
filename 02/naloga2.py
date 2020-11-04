import sys
import time
from math import sqrt
from os import listdir
from os.path import join
import numpy as np
from os import walk
import os
import re
from random import seed
from numpy import random as rand
import matplotlib.pyplot as plt


from transliterate import translit, get_available_language_codes
from itertools import combinations
import copy

def kmers(s, k=3): # default is 3
    """Generates k-mers for an input string."""
    s = re.sub('[1-9]', '', s).lower() # eliminate all numbers
    arr = []
    for i in range(len(s)-k+1):
        str = s[i:i + k]
        # if re.match('^[a-z]*[\ .]*[\ .\ ]*[a-z]*$', str):
        if re.match('^[a-z]*[\ . ]*[a-z]*', str):
            arr.append(s[i:i+k])
    return arr

def terke(text, n=4):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    arr = kmers(text, n)
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
    # print("Texts: ", listdir("clustering"))
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
                if fn[0:2] == "cz":
                    if "cz" not in lds.keys():
                        lds["cz"] = {} # Create a new dict of dicts
                    lds["cz"][fn] = nter
                elif fn[0:2] == "en":
                    if "en" not in lds.keys():
                        lds["en"] = {} # Create a new dict of dicts
                    lds["en"][fn] = nter
                elif fn[0:2] == "es":
                    if "es" not in lds.keys():
                        lds["es"] = {} # Create a new dict of dicts
                    lds["es"][fn] = nter
                elif fn[0:2] == "ma":
                    if "ma" not in lds.keys():
                        lds["ma"] = {} # Create a new dict of dicts
                    lds["ma"][fn] = nter
                elif fn[0:2] == "si":
                    if "si" not in lds.keys():
                        lds["si"] = {} # Create a new dict of dicts
                    lds["si"][fn] = nter
                else:
                    print("Error in reading prediction data...", fn)
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
    for v in d2.values():
        dist2 = dist2 + v**2
    dist2 = sqrt(dist2)

    # print("ab={0}, dist1={1}, dist2={2}".format(ab, dist1, dist2))

    dist = ab / (dist1 * dist2)
    return 1 - dist

def compute_distances(data):
    clusters = [[name] for name in sorted(list(data.keys()))]
    dists_mtx = [[0] * len(clusters) for i in range(len(clusters))]

    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if j > i:  # zgornje trikotna matrika
                dist = cosine_dist(data[c1[0]], data[c2[0]])
                # print("i;", i, ", j:", j, "dist: ", dist, "       countries: ", c1[0], c2[0])
                dists_mtx[i][j] = dist
                dists_mtx[j][i] = dist
            elif j == i:
                dists_mtx[i][i] = 0

    return dists_mtx, clusters


def k_medoids(data, medoids):
    """
    Za podane podatke (slovar slovarjev terk) in medoide vrne končne skupine
    kot seznam seznamov nizov (ključev v slovarju data).
    Klic:
    data = {"X": {"a": 1, "b": 1},
                "Y": {"a": 0.9, "b": 1},
                "Z": {"a": 1, "b": 0}}
    clusters = k_medoids(data, ["X", "Z"])  # dva medoida
    clusters = k_medoids(data, ["X", "Y", "Z"])  # trije medoidi

    za 2 medoida out:
    clusters = [["X", "Y"], ["Z"]]
    """
    clusters = [[name] for name in sorted(list(data.keys()))]
    num_clusters = len(clusters)

    dists_mtx = [[0] * num_clusters for i in range(num_clusters)]

    # if clusters[0] not in global_keys:
    for i, c1 in enumerate(clusters):
        for j, c2 in enumerate(clusters):
            if j > i:  # zgornje trikotna matrika
                dist = cosine_dist(data[c1[0]], data[c2[0]])
                # print("i;", i, ", j:", j, "dist: ", dist, "       countries: ", c1[0], c2[0])
                dists_mtx[i][j] = dist
                dists_mtx[j][i] = dist
            elif j == i:
                dists_mtx[i][i] = 0
    # else:
    #     dists_mtx = global_mtx

    converged = False
    # Assign all points to the closest medoid's cluster
    labels = [0 for i in range(len(clusters))]
    distances = [0 for i in range(len(medoids))] # maximum values (so minimal distances) to other points in the same group
    labels_old = labels.copy()
    while not converged:
        for idx1, point in enumerate(clusters): # go through all points
            min = np.inf # max value so anything will be closer
            for i, medoid in enumerate(medoids): # compare each point with each medoids
                idx2 = clusters.index([medoid])
                dist = dists_mtx[idx1][idx2]
                if dist < min:  # we found closer mediod
                    min = dist
                    labels[idx1] = i
        if labels == labels_old:
            converged = True
        else: # save labels for next iter
            labels_old = labels.copy()

            ## update medoids so that each medoids minimizes the distance WITHIN the cluster

            for candidate, label1 in zip(clusters, labels):

                dist1 = 0
                num = 0
                ii = clusters.index(candidate)

                for point, label2 in zip(clusters, labels):

                    if label1 == label2: # if we have points of the same label
                        num = num + 1
                        jj = clusters.index(point)
                        dist1 = dist1 + dists_mtx[ii][jj]

                dist1 = dist1 / num
                if dist1 < distances[label1]: # we find a better distance for label1
                    distances[label1] = dist1
                    medoids[label1] = candidate[0] # update medoid with label1 to a candidate with a better distance to others

    final_arr = [[] for i in range(len(medoids))]
    for i, point in enumerate(clusters):
        final_arr[labels[i]].append(point[0])
    return  final_arr


def silhouette(data, clusters):
    """
    Za podane podatke (slovar slovarjev terk) in skupine (seznam seznamov nizov:
    ključev v slovarju data) vrne silhueto.
    data = {"X": {"a": 1, "b": 1},
        "Y": {"a": 0.9, "b": 1},
        "Z": {"a": 1, "b": 0}}
    s1 = silhouette(data, [["X", "Y"], ["Z"], ["Q"]])
    """
    # S = (b - a) / max(b,a) = [-1, 1]
    # b: avg med tocko iz c1 in tockami iz c2. MAXIMIZE b
    # a: avg med tocko iz c1 in ostalimi tockami iz c1. MINIMIZE a

    keys = data.keys()

    all_clusters = [[name] for name in sorted(list(data.keys()))]
    num_clusters = len(all_clusters)
    dists_mtx = [[0] * num_clusters for i in range(num_clusters)]

    # for testing comment out this first if and else sentence
    # if [clusters[0][0]] not in global_keys:
    for i, c1 in enumerate(all_clusters):
        for j, c2 in enumerate(all_clusters):
            if j > i:  # zgornje trikotna matrika
                dist = cosine_dist(data[c1[0]], data[c2[0]])
                # print("i;", i, ", j:", j, "dist: ", dist, "       countries: ", c1[0], c2[0])
                dists_mtx[i][j] = dist
                dists_mtx[j][i] = dist
            elif j == i:
                dists_mtx[i][i] = 0 # one on the diagonal
    # else:  # we have global mtx of distances
    #    dists_mtx = global_mtx

    # pairs of closest clusters. We compute the silhouette between them
    pairs = []

    # go through all clusters and find the closest neighboring cluster.
    # Use that to compute the silhouette
    for i1, cluster1 in enumerate(clusters):
        min_dist = np.inf
        i2_memo = 0
        for i2, cluster2 in enumerate(clusters):
            if cluster1 is not cluster2: # so we don't calcualate the difference between the cluster with himself
                dist = 0
                for c1 in cluster1:         # c1 are elements of the cluster 1
                    for c2 in cluster2:     # c2 are elements of the cluster 2
                        index1 = all_clusters.index([c1])
                        index2 = all_clusters.index([c2])
                        dist = dist + dists_mtx[index1][index2]
                dist = dist / (len(cluster1) * len(cluster2)) # take average distance between the clusters
                if dist < min_dist:
                    min_dist = dist
                    i2_memo = i2 # remember this index
        pairs.append((i1, i2_memo))

    final_silhouettes = []
    # calculate silhouette for the following pairs of the closest clusters
    for p1, p2 in pairs:
        # print(p1, p2)
        cluster1 = clusters[p1]
        cluster2 = clusters[p2]

        silh = []
        for x1 in clusters[p1]:
            idx1 = all_clusters.index([x1])
            b = 0
            a = 0
            i = 0
            j = 0
            for x2 in clusters[p2]: # get the dist compared to points in other cluster: p1 with p2 cluster
                idx2 = all_clusters.index([x2])
                i = i + 1
                b = b + dists_mtx[idx1][idx2]
            """
            cosine theorem: 
            similarity is not equal to distance
            So, we need to flip the values with the formula:
                distance = sqrt[2(1-similarity)]
            in our case:
                b = sqrt(2*(1-b))
            theory behind the formula:
                https://stats.stackexchange.com/questions/36152/converting-similarity-matrix-to-euclidean-distance-matrix/36158#36158
            """
            b = b / i # average b
            # b = sqrt(2 * (1 - b))
            b = 1 - b
            for x3 in clusters[p1]: # p1 with p1 cluster
                idx3 = all_clusters.index([x3])
                if x1 is not x3:
                    j = j + 1
                    a = a + dists_mtx[idx1][idx3]
            if j != 0:
                a = a / j
                # a = sqrt(2 * (1 - a)) obratna vrednost
                a = 1 - a
            s = (b-a)/max(a,b)
            silh.append(s)
        avg_silh = sum(silh) / len(silh)
        final_silhouettes.append(avg_silh)

    # print(final_silhouettes)
    final_silhouettes_avg = sum(final_silhouettes) / len(final_silhouettes)
    return 1 - final_silhouettes_avg


def predict(data, text, n_terke):
    """
    Za podano bazo jezikov data za vsak jezik vrne verjetnost, da je besedilo text
    napisano v tem jeziku (izhod je v obliki slovarja).

    data = dict dictov
    text = string
    n_terke = int
    """

    text_vector = terke(text, n_terke)

    similarity_vector = {}
    for language_key, language in data.items():
        sim_language = [] # similarity list for 3 different texts
        for i, example in enumerate(language.values()): # go through all (3) vectors for one language
            sim_language.append(cosine_dist(text_vector, example))
        avg = sum(sim_language) / len(sim_language)
        similarity_vector[language_key] = avg

    normalized_vector = {}
    for k,v in similarity_vector.items():   # normalize similarities so they sum up in 1
        normalized_vector[k] = similarity_vector[k] / sum(similarity_vector.values())

    return normalized_vector
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
    data = read_clustering_data(6)  # dolžino terk prilagodite
    global_mtx, global_keys = compute_distances(data)

    silhouettes = []
    bad = []
    med = []
    high = []
    for i in range(50):
        seed(5)
        keys_cpy = list(lds.keys())[:]
        rand.shuffle(keys_cpy)
        medoids = []
        # medoids = [list(lds.keys())[index] for index in indeces]
        for i in range(5):
            medoids.append(keys_cpy.pop())

        skupine = k_medoids(lds, medoids)

        s = silhouette(data, skupine)
        silhouettes.append(s)
        print("medoids:", medoids, "\ts:", s)

        if s < 0.2:
            bad.append(medoids)
        elif 0.2 < s < 0.4:
            med.append(medoids)
        elif s > 0.4:
            high.append(medoids)
        else:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")

    # print(silhouettes)

    bins = np.arange(0, 1, 0.03)  # fixed bin size
    odmik = 0.1
    plt.xlim([min(silhouettes) - odmik, max(silhouettes) + odmik])
    plt.hist(silhouettes, bins=bins, alpha=0.8)
    plt.title('Graf porazdelitve silhuet za 50 meritev')
    plt.xlabel('Točnost silhuete (bin size = 0.05)')
    plt.ylabel('Število meritev')

    plt.show()

    print("------------ bad: ------------")
    for arr in bad:
        print(arr)
    # print("------------ med: ------------")
    # for arr in med:
    #    print(arr)
    print("------------ high: ------------")
    for arr in high:
        print(arr)

    # High: ['cz.txt', 'ser.txt', 'es.txt', 'mad.txt', 'norw.txt']


def del5():
    nterk = 6
    data = read_prediction_data(nterk)  # dolžino terk prilagodite
    # primer klica predict: print(predict(data, "Danes je lep dan", 3))
    string = "Danes je lep dan"
    print("Normalized similarity for string:'", string ,"':")
    rez = predict(data, string, nterk)
    print(rez)




if __name__ == "__main__":
    start_time = time.time()

    # file_name = "ww_en.txt"
    # file_name = "./clustering/ww_ser.txt"
    # f = open(file_name, "r")
    # print(listdir("clustering"))
    # dic = set(terke(translit(f.read().lower(), 'sr', reversed=True), 4))
    # print(dic)
    # print(len(dic))

    lds = read_clustering_data(6)
    global_mtx, global_keys = compute_distances(lds)

    """

    print(global_mtx)
    print(global_keys)
    """

    """
    for a, b in combinations(lds.keys(), 2):
        dist = cosine_dist(lds[a], lds[b])
        print()
        print("dist between", a[3:6], "and", b[3:6], ":", dist)
    """

    """
    seed(1)
    keys_cpy = list(lds.keys())[:]
    rand.shuffle(keys_cpy)
    medoids = []
    # medoids = [list(lds.keys())[index] for index in indeces]
    for i in range(5):
        medoids.append(keys_cpy.pop())
    print(medoids)

    arr = k_medoids(lds, medoids)
    """

    """
    print("arr:", arr)
    for i, a in enumerate(arr):
        print("Group {} consists of:".format(i))
        for a_ in a:
            print("\t", a_)
    """

    # odkomenirajte del naloge, ki ga želite pognati
    # del2()
    # del4()
    # del5()
    print("--- %s seconds ---" % (time.time() - start_time))
    print("-- END --")
    pass

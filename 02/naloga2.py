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
    for fn in listdir("clustering1"):
    # for fn in ['ww_bs.txt', 'ww_ru.txt']:
        if fn.lower().endswith(".txt"):
            with open(join("clustering1", fn), encoding="utf8") as f:
                text = f.read()
                # nter = terke(translit(f.read().lower(), reversed=True), n=n_terke)
                # ['mn', 'ru', 'mk', 'sr', 'bg', 'hy', 'el', 'ka', 'l1', 'uk']
                if fn == "ww_mac.txt":
                    nter = terke(translit(text.lower(), 'mk', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "ww_rus.txt":
                    nter = terke(translit(text.lower(), 'ru', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "ww_ser.txt":
                    nter = terke(translit(text.lower(), 'sr', reversed=True), n=n_terke)
                    lds[fn] = nter
                elif fn == "ww_bs.txt":
                    nter = terke(translit(text.lower(), 'ru', reversed=True), n=n_terke)
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


def del2():
    data = read_clustering_data(3)  # dolžino terk prilagodite



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

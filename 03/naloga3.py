import numpy as np
import re
from transliterate import translit, get_available_language_codes
from os import listdir
from os.path import join
import time

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

def num_occur(nter, langs): # count num occurences of nter string acros the documents
    total = 0

    for lang, dict in langs.items():
        if nter in dict.keys():
            total = total + 1

    return total

def prepare_data_matrix():
    """
    Return data in a matrix (2D numpy array), where each row contains triplets
    for a language. Columns should be the 100 most common triplets
    according to the idf measure.
    """
    start_time = time.time()
    # create matrix X and list of languages
    langs = read_clustering_data(3)
    all_nters = {}

    for language, dict in langs.items():
        for nter in dict:
            if nter not in all_nters.keys():
                all_nters[nter] = num_occur(nter, langs) # we have a new nter to store in our global all_nter dict

    all_nters_cp = {}
    # remove all nters not occuring everywhere
    for k,v in all_nters.items():
        if v == 20:
            all_nters_cp[k] = v

    # find tutal number of occurences of certain nter in all documents
    nters_total =  {}
    for k,v in all_nters_cp.items():
        nters_total[k] = 0
        for language, dict in langs.items(): # go through all languages
            nters_total[k] += dict[k]

    # total number of occurences sorted so we can later  on pick top 100
    nters_total_sorted_reverse = {k: v for k, v in sorted(nters_total.items(), key=lambda item: item[1])}

    nters_total_sorted = {}
    for x in list(reversed(list(nters_total_sorted_reverse)))[0:100]:
        nters_total_sorted[x] = nters_total_sorted_reverse[x]

    X = np.zeros((20,100)) # 20 rows for 20 languages with 100 triplets
    languages = list(key[:-4] for key in langs.keys()) # substring languages from data
    for i, k in enumerate(nters_total_sorted.keys()):
        j = 0
        for lang, vector in langs.items():  # fill column i for all languages, then move on to the next triplet aka next column
            X[j][i] = vector[k] # find the number of k occurences in document of language "lang" and save it in matrix
            j = j + 1

    print("--- %s seconds ---" % (time.time() - start_time))
    return X, languages


def power_iteration(X):
    """
    Compute the eigenvector with the greatest eigenvalue
    of the covariance matrix of X (a numpy array).

    Return two values:
    - the eigenvector (1D numpy array) and
    - the corresponding eigenvalue (a float)
    """
    pass


def power_iteration_two_components(X):
    """
    Compute first two eigenvectors and eigenvalues with the power iteration method.
    This function should use the power_iteration function internally.

    Return two values:
    - the two eigenvectors (2D numpy array, each eigenvector in a row) and
    - the corresponding eigenvalues (a 1D numpy array)
    """
    pass


def project_to_eigenvectors(X, vecs):
    """
    Project matrix X onto the space defined by eigenvectors.
    The output array should have as many rows as X and as many columns as there
    are vectors.
    """
    pass


def total_variance(X):
    """
    Total variance of the data matrix X. You will need to use for
    to compute the explained variance ratio.
    """
    return np.var(X, axis=0, ddof=1).sum()


def explained_variance_ratio(X, eigenvectors, eigenvalues):
    """
    Compute explained variance ratio.
    """
    pass


def plot_PCA():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of PCA on languages data.
    """
    X, languages = prepare_data_matrix()
    # ...


def plot_MDS():
    """
    Everything (opening files, computation, plotting) needed
    to produce a plot of MDS on languages data.

    Use sklearn.manifold.MDS and explicitly run it with a distance
    matrix obtained with cosine distance on full triplets, like
    in the previous homework.
    """
    pass
    # ...


if __name__ == "__main__":
   plot_MDS()
   plot_PCA()
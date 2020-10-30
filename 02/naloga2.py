from os import listdir
from os.path import join




def kmers(s, k=3):
    """Generates k-mers for an input string."""
    for i in range(len(s)-k+1):
        yield s[i:i+k]



def terke(text, K=4):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    for i in range(len(s)-k+1):
        yield s[i:i+k]

def read_clustering_data(n_terke):
    # Prosim, ne spreminjajte te funkcije. Vso potrebno obdelavo naredite
    # v funkciji terke.
    lds = {}
    for fn in listdir("clustering"):
        if fn.lower().endswith(".txt"):
            with open(join("clustering", fn), encoding="utf8") as f:
                text = f.read()
                nter = terke(text, n=n_terke)
                lds[fn] = nter
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
    pass


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
    # ... nadaljujte


def del4():
    data = read_clustering_data(3)  # dolžino terk prilagodite
    # ... nadaljujte


def del5():
    data = read_prediction_data(3)  # dolžino terk prilagodite
    # ... nadaljujte
    # primer klica predict: print(predict(data, "Danes je lep dan", 3))


if __name__ == "__main__":
    file_name = "./texts/germanski/ww_en.txt"
    f = open(file_name, "rt")
    print(f)
    # dic = terke(f, 4)
    # odkomenirajte del naloge, ki ga želite pognati
    # del2()
    # del4()
    # del5()
    pass
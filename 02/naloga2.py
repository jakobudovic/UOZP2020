from os import listdir
from os.path import join


def terke(text, n):
    """
    Vrne slovar s preštetimi terkami dolžine n.
    """
    pass


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
    pass
    # odkomenirajte del naloge, ki ga želite pognati
    # del2()
    # del4()
    # del5()

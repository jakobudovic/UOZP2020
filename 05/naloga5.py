import numpy as np
from matplotlib import pyplot
from scipy.optimize import fmin_l_bfgs_b


def draw_decision(X, y, classifier, at1, at2, grid=50):
    points = np.take(X, [at1, at2], axis=1)
    maxx, maxy = np.max(points, axis=0)
    minx, miny = np.min(points, axis=0)
    difx = maxx - minx
    dify = maxy - miny
    maxx += 0.02 * difx
    minx -= 0.02 * difx
    maxy += 0.02 * dify
    miny -= 0.02 * dify

    pyplot.figure(figsize=(8, 8))

    for c,(x,y) in zip(y,points):
        pyplot.text(x, y, str(c), ha="center", va="center")
        pyplot.scatter([x], [y], c=["b", "r"][int(c) != 0], s=200)

    num = grid
    prob = np.zeros([num, num])
    for xi, x in enumerate(np.linspace(minx, maxx, num=num)):
        for yi, y in enumerate(np.linspace(miny, maxy, num=num)):
            # probability of the closest example
            diff = points - np.array([x, y])
            dists = (diff[:, 0]**2 + diff[:, 1]**2)**0.5  # euclidean
            ind = np.argsort(dists)
            prob[yi,xi] = classifier(X[ind[0]])[1]

    pyplot.imshow(prob, extent=(minx, maxx, maxy, miny), cmap="seismic")

    pyplot.xlim(minx, maxx)
    pyplot.ylim(miny, maxy)
    pyplot.xlabel(at1)
    pyplot.ylabel(at2)

    pyplot.show()


def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y


def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    # ... dopolnite (naloga 1)
    return 0.


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    # ... dopolnite (naloga 1, naloga 2)
    return 0.


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne 1D numpy array v velikosti vektorja theta.
    """
    # ... dopolnite (naloga 1, naloga 2)
    return None


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    # ... dopolnite (naloga 1, naloga 2)
    return None


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    """
    Primer klica:
        res = test_cv(LogRegLearner(lambda_=0.0), X, y)
    ... dopolnite (naloga 3)
    """
    pass


def CA(real, predictions):
    # ... dopolnite (naloga 3)
    pass


def AUC(real, predictions):
    # ... dopolnite (naloga 4)
    pass


def del2():
    # ... dopolnite
    pass


def del3():
    # ... dopolnite
    pass


def del4():
    # ... dopolnite
    pass


def del5():
    # ... dopolnite
    pass


if __name__ == "__main__":
    # Primer uporabe, ki bo delal, ko implementirate cost in grad

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y) # dobimo model

    napoved = classifier(X[0])  # napoved za prvi primer
    print(napoved)

    # izris odlocitev
    draw_decision(X, y, classifier, 0, 1)

    # odpiranje GDS350
    X, y = load('GDS360.data')
    print(X.shape, y.shape)
    print(X, y)

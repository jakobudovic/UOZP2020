import unittest
import time

t = time.time()
import naloga2
if time.time() - t > 2:
    raise Exception("Uvoz vaše kode traja več kot dve sekundi. Pazite, da se ob importu koda ne požene.")

from naloga2 import terke, read_clustering_data, read_prediction_data, \
    cosine_dist, k_medoids, silhouette, predict, \
    del2, del4, del5


class Naloga2Test(unittest.TestCase):

    def test_terke(self):
        t = terke("Filozofija", 3)
        self.assertEqual(t, {'fil': 1, 'ilo': 1, 'loz': 1, 'ozo': 1, 'zof': 1, 'ofi': 1, 'fij': 1, 'ija': 1})
        t = terke("Filozofija", 9)
        self.assertEqual(t, {'filozofij': 1, 'ilozofija': 1})

        # ne odstranjuj presledkov
        t = terke("Filozofija riba", 4)
        self.assertNotIn("jari", t)

    def test_read_clustering_data(self):
        # Preveri samo format, ne pa vsebine
        data = read_clustering_data(3)
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 19)
        # podslovarji so tudi tipa dict
        for v in data.values():
            self.assertIsInstance(v, dict)

    def test_cosine_dist(self):
        d1 = {"a": 1, "b": 1}
        d2 = {"c": 1, "d": 1}
        dist = cosine_dist(d1, d2)
        self.assertAlmostEqual(dist, 1)
        d1 = {"a": 1, "b": 1}
        d2 = {"a": 1, "b": 1}
        self.assertAlmostEqual(cosine_dist(d1, d2), 0)

    def test_k_medoids(self):
        data = {"X": {"a": 1, "b": 1},
                "Y": {"a": 0.9, "b": 1},
                "Z": {"a": 1, "b": 0}}

        clusters = k_medoids(data, ["X", "Z"])  # dva medoida
        self.assertEqual(2, len(clusters))  # dva clustra na koncu
        self.assertIn(["Z"], clusters)  # poseben element bo ločen cluster
        self.assertTrue(["X", "Y"] in clusters or ["Y", "X"] in clusters)

        clusters = k_medoids(data, ["X", "Y", "Z"])  # trije medoidi
        self.assertEqual(3, len(clusters))  # trije clustri na koncu
        self.assertIn(["X"], clusters)
        self.assertIn(["Y"], clusters)
        self.assertIn(["Z"], clusters)

    def test_silhouette(self):
        data = {"X": {"a": 1, "b": 1},
                "Y": {"a": 0.9, "b": 1},
                "Z": {"a": 1, "b": 0}}

        s1 = silhouette(data, [["X", "Y"], ["Z"]])  # boljše skupine
        s2 = silhouette(data, [["X", "Z"], ["Y"]])  # slabše skupine
        s3 = silhouette(data, [["Y", "Z"], ["X"]])  # še slabše skupine
        self.assertLess(s2, s1)
        self.assertLess(s3, s2)

    def test_predict(self):
        data = read_prediction_data(3)
        res = predict(data, "Danes je lep dan in na cesti sem videl ribo.", 3)
        self.assertIsInstance(res, dict)
        for v in res.values():
            self.assertIsInstance(v, float)
        probs = res.values()
        # verjetnosti naj se seštejejo v 1
        self.assertAlmostEqual(1, sum(probs))


if __name__ == "__main__":
    unittest.main(verbosity=2)

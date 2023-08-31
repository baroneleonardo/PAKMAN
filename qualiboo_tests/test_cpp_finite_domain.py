import unittest
import numpy as np
from moe.build import GPP as cpp_moe


class CPPFiniteDomainTests(unittest.TestCase):

    def test_constructor(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.tolist(),
                                      data.shape[1])

    def test_dim(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.data.tolist(),
                                      data.shape[1])
        self.assertEqual(domain.dim(), 5)

    def test_print(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.data.tolist(),
                                      data.shape[1])
        domain.print()
        # stdout is discarded in success
        raise ValueError()

    def test_get_data(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.tolist(),
                                      data.shape[1])
        self.assertEqual(data.tolist(), domain.get_data())

    def test_find_distances_and_indexes_from_point(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.tolist(),
                                      data.shape[1])
        point = [1, 2, 3, 4, 5]
        distances, indexes = domain.find_distances_and_indexes_from_point(point)
        np_distances = np.linalg.norm(data - np.array(point), axis=1)
        np_indexes = np.argsort(np_distances)
        np_distances = np_distances[np_indexes]
        self.assertEqual(np_distances.tolist(), distances)
        self.assertEqual(np_indexes.tolist(), indexes)

    def test_set_seed(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        domain = cpp_moe.FiniteDomain(data.data.tolist(),
                                      data.shape[1])
        domain.set_seed(1979)

    def test_sample_points(self):
        np.random.seed(1984)
        data = np.random.randint(-10, 10, (7, 5)).astype(float)
        data_list = data.tolist()
        domain = cpp_moe.FiniteDomain(data_list,
                                      data.shape[1])
        points = domain.sample_points_in_domain(4, True)
        self.assertEqual(4, len(points), str(points))
        for point in points:
            self.assertIn(point, data_list)

        # Asking too many points
        self.assertFalse(domain.sample_points_in_domain(10, True))

        # Asking too many unique points
        domain.sample_points_in_domain(len(data_list), False)
        self.assertFalse(domain.sample_points_in_domain(1, False))


if __name__ == '__main__':
    unittest.main()

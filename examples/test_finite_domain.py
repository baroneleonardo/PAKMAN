import unittest

import numpy as np

from examples import finite_domain


class FiniteDomainTests(unittest.TestCase):

    def test_grid(self):
        domain = finite_domain.FiniteDomain.Grid([1, 2, 3], [10, 20])
        expected = np.array([[1, 10],
                             [1, 20],
                             [2, 10],
                             [2, 20],
                             [3, 10],
                             [3, 20]])
        self.assertTrue(np.all(domain._data == expected))

    def test_sample(self):
        i_index = [1, 2, 3]
        j_index = [10, 20]
        k_index = [100, 200]
        domain = finite_domain.FiniteDomain.Grid(i_index,
                                                 j_index,
                                                 k_index)
        points = np.array(np.meshgrid(i_index, j_index, k_index)).T.reshape(-1,3)
        sample_size = 4

        for _ in range(100):
            sample = domain.SamplePointsInDomain(sample_size,
                                                 allow_previously_sampled=True)
            for point in sample:
                self.assertTrue(any(np.all(point == row)
                                    for row in points))

        domain = finite_domain.FiniteDomain.Grid(i_index,
                                                 j_index,
                                                 k_index)
        # If resampling is not allowed, only 4 * 3 samples can be drawn
        for _ in range(3):
            sample = domain.SamplePointsInDomain(sample_size,
                                                 allow_previously_sampled=False)
            for point in sample:
                self.assertTrue(any(np.all(point == row)
                                    for row in points))
        # The next sample will be None
        self.assertIsNone(
            domain.SamplePointsInDomain(sample_size,
                                        allow_previously_sampled=False)
        )

    def test_find_closest_point(self):
        domain = finite_domain.FiniteDomain.Grid([1, 2, 3], [10, 20])
        test_point = np.array([2.1, 20.1])
        expected_point = np.array([2.0, 20.0])
        closest_point = domain.find_closest_point(test_point)
        self.assertTrue(np.all(np.equal(expected_point, closest_point)))


if __name__ == '__main__':
    unittest.main()

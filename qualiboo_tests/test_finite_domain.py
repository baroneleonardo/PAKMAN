"""Finite domain tests

Tests on the FiniteDomain API.
Both Python and CPP implementations are tested to confirm
that they can be swapped.
"""
import unittest

import numpy as np

from qualiboo import finite_domain


class PyFiniteDomainTests(unittest.TestCase):

    domain_class = finite_domain.FiniteDomain

    def test_grid(self):
        domain = self.domain_class.Grid([1, 2, 3], [10, 20])
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
        domain = self.domain_class.Grid(i_index,
                                                 j_index,
                                                 k_index)
        points = np.array(np.meshgrid(i_index, j_index, k_index)).T.reshape(-1, 3)
        sample_size = 4

        for _ in range(100):
            sample = domain.sample_points_in_domain(sample_size,
                                                    allow_previously_sampled=True)
            for point in sample:
                self.assertTrue(any(np.all(point == row)
                                    for row in points))

        domain = self.domain_class.Grid(i_index,
                                        j_index,
                                        k_index)
        # If resampling is not allowed, only 4 * 3 samples can be drawn
        for _ in range(3):
            sample = domain.sample_points_in_domain(sample_size,
                                                    allow_previously_sampled=False)
            for point in sample:
                self.assertTrue(any(np.all(point == row)
                                    for row in points), str(point))
        # The next sample will be None
        self.assertIsNone(
            domain.sample_points_in_domain(sample_size,
                                           allow_previously_sampled=False)
        )

    def test_find_closest_point(self):
        domain = self.domain_class.Grid([1, 2, 3], [10, 20])
        test_point = np.array([2.1, 20.1])
        expected_point = np.array([2.0, 20.0])
        _, _, closest_point = domain.find_distance_index_closest_point(test_point)
        self.assertTrue(np.all(np.equal(expected_point, closest_point)))


class CppFiniteDomainTests(PyFiniteDomainTests):
    domain_class = finite_domain.CPPFiniteDomain


if __name__ == '__main__':
    unittest.main()

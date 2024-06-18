"""Finite Domain

This module implements two classes that share the same interface.
One is implemented in pure python, the other one is a wrapper
araound a C++ implementation.
"""
import abc

import numpy as np
from scipy import spatial
from typing import Tuple

from moe.optimal_learning.python import geometry_utils

from moe.build import GPP as C_GP


class _AbstractFiniteDomain(abc.ABC):

    @classmethod
    def Grid(cls, *coords):
        """Build a new finite domain given the coordinates for each dimension

        Ex Grid([0, 0.5, 1], [-1, 0]) builds a 3 x 2 domain
        """
        grid = np.meshgrid(*coords, indexing='ij')
        data = np.vstack([g.ravel() for g in grid]).T
        return cls(data)

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        raise NotImplementedError


class FiniteDomain(_AbstractFiniteDomain):

    # TODO: Using this type of domain since C++ implementation
    #  of FiniteDomain is not complete
    _domain_type = C_GP.DomainTypes.tensor_product

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(**kwargs)  # Just used for multi-class inheritance
        self._data = data
        self._kdtree = spatial.KDTree(data)
        self._sampled = np.zeros(data.shape[0]).astype(bool)

        self._domain_bounds = [geometry_utils.ClosedInterval(np.min(data[:, i]).astype(float),
                                                             np.max(data[:, i]).astype(float))
                               for i in range(data.shape[1])]

    def sample_points_in_domain(self, sample_size: int, allow_previously_sampled: bool = False) -> np.ndarray:
        if allow_previously_sampled:
            indexes = np.arange(self._data.shape[0])
        else:
            indexes = np.flatnonzero(~self._sampled)
            if len(indexes) < sample_size:
                return None
        selected = np.random.choice(indexes, sample_size, replace=False)
        self._sampled[selected] = True
        return self._data[selected]

    @property
    def dim(self) -> int:
        return self._data.shape[1]

    @property
    def domain_bounds(self):
        return self._domain_bounds

    def generate_uniform_random_points_in_domain(self, num_points):
        r"""Generate ``num_points`` on a latin-hypercube (i.e., like a checkerboard).

        See python.geometry_utils.generate_latin_hypercube_points for more details.

        :param num_points: max number of points to generate
        :type num_points: int >= 0
        :return: uniform random sampling of points from the domain
        :rtype: array of float64 with shape (num_points, dim)

        """
        # TODO(GH-56): Allow users to pass in a random source.
        return geometry_utils.generate_latin_hypercube_points(
            num_points,
            self._domain_bounds
        )

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        Changes new_update_vector so that:
          ``point_new = point + new_update_vector``

        has coordinates such that ``CheckPointInside(point_new)`` returns true. We select ``point_new``
        by projecting ``point + update_vector`` to the nearest point on the domain.

        ``new_update_vector`` is a function of ``update_vector``.
        ``new_update_vector`` is just a copy of ``update_vector`` if ``current_point`` is already inside the domain.

        .. NOTE::
            We modify update_vector (instead of returning point_new) so that further update
            limiting/testing may be performed.

        :param max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
        :type max_relative_change: float64 in (0, 1]
        :param current_point: starting point
        :type current_point: array of float64 with shape (dim)
        :param update_vector: proposed update
        :type update_vector: array of float64 with shape (dim)
        :return: new update so that the final point remains inside the domain
        :rtype: array of float64 with shape (dim)

        """
        # TODO(GH-58): Vectorize the loop over j, step.
        output_update = np.empty(self.dim)
        # Note: since all boundary planes are axis-aligned, projecting becomes very simple.
        for j, step in enumerate(update_vector):
            # Distance to the nearest boundary in the j-th dimension
            distance_to_boundary = np.fmin(
                current_point[j] - self._domain_bounds[j].min,
                self._domain_bounds[j].max - current_point[j])

            desired_step = step
            # If we are close to a boundary, optionally (via max_relative_change) limit the step size
            # 0 < max_relative_change <= 1 so at worst we reach the boundary.
            if np.fabs(step) > max_relative_change * distance_to_boundary:
                # Move the max allowed distance, in the original direction of travel (obtained via copy-sign)
                desired_step = np.copysign(max_relative_change * distance_to_boundary, step)

            output_update[j] = desired_step

        return output_update

    def find_distances_indexes_closest_points(self, point: np.ndarray, k: int = 30) -> Tuple[float, int, np.ndarray]:
        distances, indexes = self._kdtree.query(point, k=k, workers=4)  # TODO: Decide the number of returned points k
        return distances, indexes, self._data[indexes]

    def find_distance_index_closest_point(self, point: np.ndarray) -> np.ndarray:
        return self.find_distances_indexes_closest_points(point, k=1)


class CPPFiniteDomain(_AbstractFiniteDomain):

    # TODO: Using this type of domain since C++ implementation
    #  of FiniteDomain is not complete
    _domain_type = C_GP.DomainTypes.tensor_product

    def __init__(self, data: np.ndarray, **kwargs):
        super().__init__(**kwargs)  # Just used for multi-class inheritance
        self._data = data
        self._cpp_finite_domain = C_GP.FiniteDomain(data.tolist(), data.shape[1])

        self._domain_bounds = [geometry_utils.ClosedInterval(np.min(data[:, i]).astype(float),
                                                             np.max(data[:, i]).astype(float))
                               for i in range(data.shape[1])]

    def sample_points_in_domain(self, sample_size: int, allow_previously_sampled: bool = False) -> np.ndarray:
        points_as_list = self._cpp_finite_domain.sample_points_in_domain(
            sample_size,
            allow_previously_sampled
        )
        if points_as_list:
            # Return something only if the list is not empty
            return np.array(points_as_list)

    @property
    def dim(self) -> int:
        return self._cpp_finite_domain.dim()

    @property
    def domain_bounds(self):
        return self._domain_bounds

    def generate_uniform_random_points_in_domain(self, num_points: int) -> np.ndarray:
        r"""Generate ``num_points`` on a latin-hypercube (i.e., like a checkerboard).

        See python.geometry_utils.generate_latin_hypercube_points for more details.

        :param num_points: max number of points to generate
        :type num_points: int >= 0
        :return: uniform random sampling of points from the domain
        :rtype: array of float64 with shape (num_points, dim)

        """
        # Set random seed variable
        np.random.seed(np.random.randint(0, 1000000))
        # TODO(GH-56): Allow users to pass in a random source.
        return geometry_utils.generate_latin_hypercube_points(
            num_points,
            self._domain_bounds
        )
        return self.sample_points_in_domain(num_points, True)
        # return self._cpp_finite_domain.GenerateLatinHypercubePoints(num_points=num_points)

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        Changes new_update_vector so that:
          ``point_new = point + new_update_vector``

        has coordinates such that ``CheckPointInside(point_new)`` returns true. We select ``point_new``
        by projecting ``point + update_vector`` to the nearest point on the domain.

        ``new_update_vector`` is a function of ``update_vector``.
        ``new_update_vector`` is just a copy of ``update_vector`` if ``current_point`` is already inside the domain.

        .. NOTE::
            We modify update_vector (instead of returning point_new) so that further update
            limiting/testing may be performed.

        :param max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
        :type max_relative_change: float64 in (0, 1]
        :param current_point: starting point
        :type current_point: array of float64 with shape (dim)
        :param update_vector: proposed update
        :type update_vector: array of float64 with shape (dim)
        :return: new update so that the final point remains inside the domain
        :rtype: array of float64 with shape (dim)

        """
        # TODO(GH-58): Vectorize the loop over j, step.
        output_update = np.empty(self.dim)
        # Note: since all boundary planes are axis-aligned, projecting becomes very simple.
        for j, step in enumerate(update_vector):
            # Distance to the nearest boundary in the j-th dimension
            distance_to_boundary = np.fmin(
                current_point[j] - self._domain_bounds[j].min,
                self._domain_bounds[j].max - current_point[j])

            desired_step = step
            # If we are close to a boundary, optionally (via max_relative_change) limit the step size
            # 0 < max_relative_change <= 1 so at worst we reach the boundary.
            if np.fabs(step) > max_relative_change * distance_to_boundary:
                # Move the max allowed distance, in the original direction of travel (obtained via copy-sign)
                desired_step = np.copysign(max_relative_change * distance_to_boundary, step)

            output_update[j] = desired_step

        return output_update

    def find_distances_indexes_closest_points(self, point: np.ndarray, k=30) -> Tuple[float, int, np.ndarray]:
        point_as_list = np.array(point).flatten().tolist()  # TODO: Multiple opposed cast, use something smarter
        distances, indexes = self._cpp_finite_domain.find_distances_and_indexes_from_point(point_as_list)
        distances, indexes = np.array(distances), np.array(indexes)
        return distances[:k], indexes[:k], self._data[indexes][:k]

    def find_distance_index_closest_point(self, point: np.ndarray) -> np.ndarray:
        distances, indexes, closest_point = self.find_distances_indexes_closest_points(point, 1)
        return distances, indexes, closest_point.flatten()

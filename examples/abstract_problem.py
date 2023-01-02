import numpy as np

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pyTPD
from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTPD


class _AbstractProblem:

    def get_inner_search_domain(self):
        closed_interval_list = [ClosedInterval(self._search_domain[i, 0], self._search_domain[i, 1])
                                for i in range(self._search_domain.shape[0] - self._num_fidelity)]
        return pyTPD(closed_interval_list)

    def get_cpp_search_domain(self):
        closed_interval_list = [ClosedInterval(bound[0], bound[1]) for bound in self._search_domain]
        return cppTPD(closed_interval_list)

    def get_cpp_inner_search_domain(self):
        closed_interval_list = [ClosedInterval(self._search_domain[i, 0], self._search_domain[i, 1])
                                for i in range(self._search_domain.shape[0]-self._num_fidelity)]
        return cppTPD(closed_interval_list)

    def get_initial_points(self):
        return np.zeros((self._num_init_pts, self._dim))

    @property
    def n_derivatives(self):
        if self._use_observations:
            return self._dim
        return 0

    @property
    def n_observations(self):
        return self.n_derivatives + 1

    @property
    def derivatives(self):  # _GenericProblem.derivatives
        return np.arange(self.n_derivatives)

    @property
    def observations(self):
        return np.arange(self.n_observations)
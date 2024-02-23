import numpy as np

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain as pyTPD
from moe.optimal_learning.python.cpp_wrappers.domain import TensorProductDomain as cppTPD


class AbstractProblem:

    def __init__(self, *,
                 search_domain,
                 min_value):
        self.search_domain = search_domain
        self.min_value = min_value

        self._evaluation_count = 0

    def get_search_domain(self):
        closed_interval_list = [ClosedInterval(bound[0], bound[1]) for bound in self.search_domain]
        return cppTPD(closed_interval_list)

    def get_initial_points(self):
        return np.zeros((self.num_init_pts, self.dim))

    @property
    def dim(self) -> int:
        return len(self.search_domain)

    @property
    def evaluation_count(self):
        return self._evaluation_count

    @property
    def n_observations(self):
        return 1

    @property
    def derivatives(self):  # _GenericProblem.derivatives
        return np.arange(0)

    @property
    def observations(self):
        return np.arange(self.n_observations)

    def evaluate(self, x, *, do_not_count=False):
        if not do_not_count:
            self._evaluation_count += 1
        return self.evaluate_true(x)

    def add_evaluation_count(self, n):
        self._evaluation_count += n
        return
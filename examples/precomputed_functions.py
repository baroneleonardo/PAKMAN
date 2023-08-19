import logging

import numpy as np

import datasets
from examples import finite_domain, abstract_problem

import moe.build.GPP as C_GP


_log = logging.getLogger(__name__)


class _PrecomputedFunction(finite_domain.FiniteDomain, abstract_problem.AbstractProblem):

    def __init__(self, dataset: datasets.Dataset):
        m = np.min(dataset.X, axis=0)
        M = np.max(dataset.X, axis=0)
        domain_bounds = np.vstack([m, M]).transpose()
        super().__init__(data=dataset.X.values,
                         dim=dataset.X.shape[1],
                         search_domain=domain_bounds,
                         min_value=np.min(dataset.y))
        self._dataset = dataset

    @property
    def minimum(self):
        ix = np.argmin(self._dataset.y)
        return self._dataset.X.loc[ix]

    def evaluate_true(self, x):
        distances, indexes, points = self.find_distances_indexes_closest_points(x)
        # if any(distances > 0.00001):
        #     _log.warning(f'POSSIBLE EVALUATION ERROR: The distance between the point in '
        #                  f'domain and the evaluated point is large')
        mask = distances == 0.0
        if np.sum(mask) == 0:  # No exact match, return closest point
            values = [self._dataset.y[indexes[0]]]
        elif np.sum(mask) == 1:  # Only one match, return it
            values = self._dataset.y[indexes[mask]]
        else:  # Multiple exact matches, random
            indexes = indexes[mask]
            values = [self._dataset.y[np.random.choice(indexes)]]
        return np.array(values)


LiGen = _PrecomputedFunction(
    dataset=datasets.LiGen
)

Query26 = _PrecomputedFunction(
    dataset=datasets.Query26
)

StereoMatch = _PrecomputedFunction(
    dataset=datasets.StereoMatch
)


class CPPPrecomputedFunction(finite_domain.CPPFiniteDomain):

    def __init__(self, dataset: datasets.Dataset):
        super().__init__(data=dataset.X.values)
        self._cpp_precomputed_function = C_GP.PrecomputedFunction(...)
        self._dataset = dataset

    @property
    def minimum(self) -> float:
        return self._cpp_precomputed_function.minimum

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        return self._cpp_precomputed_function.Evaluate(x)

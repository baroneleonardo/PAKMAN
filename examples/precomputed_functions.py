import logging

import numpy as np

import datasets
from examples import finite_domain, abstract_problem

import moe.build.GPP as C_GP


_log = logging.getLogger(__name__)


class PrecomputedFunction(finite_domain.FiniteDomain, abstract_problem.AbstractProblem):

    def __init__(self, dataset: datasets.Datasets):
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
        distance, ix, point = self.find_distance_index_closest_point(x)
        if distance > 0.00001:
            _log.warning(f'POSSIBLE EVALUATION ERROR: The distance between the point in '
                         f'domain and the evaluated point is {distance}')
        value = self._dataset.y[ix]
        return np.array([value])

    @classmethod
    def LiGen(cls):
        return cls(
            dataset=datasets.Datasets.LiGen()
        )

    @classmethod
    def Query26(cls):
        return cls(
            dataset=datasets.Datasets.Query26()
        )

    @classmethod
    def Stereomatch(cls):
        return cls(
            dataset=datasets.Datasets.Stereomatch()
        )


class CPPPrecomputedFunction(finite_domain.CPPFiniteDomain):

    def __init__(self, dataset: datasets.Datasets):
        super().__init__(data=dataset.X.values)
        self._cpp_precomputed_function = C_GP.PrecomputedFunction(...)
        self._dataset = dataset

    @property
    def minimum(self) -> float:
        return self._cpp_precomputed_function.minimum

    def evaluate_true(self, x: np.ndarray) -> np.ndarray:
        return self._cpp_precomputed_function.Evaluate(x)

import numpy as np

import datasets
from examples import finite_domain, abstract_problem


class PrecomputedFunction(finite_domain.FiniteDomain, abstract_problem._AbstractProblem):

    def __init__(self, dataset: datasets.Datasets, n_init_pts=5):
        super().__init__(data=dataset.X.values)
        self._dataset = dataset
        self.num_init_pts = n_init_pts
        self._use_observations = False

    @property
    def dim(self):
        return self._dataset.X.shape[1]

    @property
    def search_domain(self):
        m = np.min(self._data, axis=0)
        M = np.max(self._data, axis=0)
        return np.vstack([m, M]).transpose()

    @property
    def min_value(self):
        return np.min(self._dataset.y)

    def evaluate_true(self, x):
        distance, ix, point = self.find_distance_index_closest_point(x)
        if distance > 0.00001:
            print(f'POSSIBLE EVALUATION ERROR: The distance between the point in '
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

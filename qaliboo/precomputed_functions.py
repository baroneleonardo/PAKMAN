"""Precomputed functions

The instances exposed here are utilities to manage
 the sample datasets as domains and target functions
"""
import logging

import numpy as np

from examples import abstract_problem
from qaliboo import finite_domain, datasets


_log = logging.getLogger(__name__)
_log.setLevel(level=logging.DEBUG)


class _PrecomputedFunction(finite_domain.CPPFiniteDomain, abstract_problem.AbstractProblem):

    def __init__(self, dataset: datasets.Dataset):
        m = np.min(dataset.X, axis=0).astype(float)
        M = np.max(dataset.X, axis=0).astype(float)
        domain_bounds = np.vstack([m, M]).transpose()
        super().__init__(data=dataset.X.values,
                         search_domain=domain_bounds,
                         min_value=np.min(dataset.y))
        self._dataset = dataset
        self.lower_bounds = m
        self.upper_bounds = M

    @property
    def lower_bound(self):
        return self.lower_bounds
    @property
    def upper_bound(self):
        return self.upper_bounds
    @property
    def minimum(self):
        ix = np.argmin(self._dataset.y)
        return self._dataset.X.loc[ix].values
    @ property
    def dataset(self):
        return self._dataset

    def evaluate_true(self, x):
        distances, indexes, points = self.find_distances_indexes_closest_points(x)
        # if any(distances > 0.00001):
        #     _log.warning(f'POSSIBLE EVALUATION ERROR: The distance between the point in '
        #                  f'domain and the evaluated point is large')
        min_distance = np.min(distances)
        
        #if min_distance == 0.0:
        #    _log.debug('There is at least one exact match')
        
        mask = distances == min_distance
        if np.sum(mask) == 1:
            #_log.debug('Only one match, return it')
            my_index = indexes[mask][0]
            values = self._dataset.y[my_index]
            realtime = self._dataset.real_time[my_index]
        else:
            #_log.debug('Multiple matches, random pick...')
            indexes = indexes[mask]
            my_index = np.random.choice(indexes)
            values = self._dataset.y[my_index]
            realtime = self._dataset.real_time[my_index]
        
        return np.array(values), my_index, realtime
    
    def evaluate_time(self, x):
        distances, indexes, points = self.find_distances_indexes_closest_points(x)
        min_distance = np.min(distances)
        mask = distances == min_distance
        
        if np.sum(mask)==1:
            my_index = indexes[mask][0]
            values = self._dataset.time[my_index]
            #values = self._dataset.time[indexes[mask]]

        else:
            indexes = indexes[mask]
            my_index = np.random.choice(indexes)
            values = self._dataset.time[my_index]

        return values




Query26 = _PrecomputedFunction(
    dataset=datasets.Query26
)

ScaledQuery26 = _PrecomputedFunction(
    dataset=datasets.ScaledQuery26
)

StereoMatch = _PrecomputedFunction(
    dataset=datasets.StereoMatch
)

ScaledStereoMatch = _PrecomputedFunction(
    dataset=datasets.ScaledStereoMatch
)

LiGenTot = _PrecomputedFunction(
    dataset=datasets.LiGenTot
)

ScaledLiGenTot = _PrecomputedFunction(
    dataset=datasets.ScaledLiGenTot
)

ScaledStereoMatch10 = _PrecomputedFunction(
    dataset=datasets.ScaledStereoMatch10
)
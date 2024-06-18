"""Datasets

The module contains three data sets and some auxiliary classes
to load and read them.
"""
import logging

import pandas as pd
import numpy as np
import os
from typing import List


_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)


class Dataset:

    def __init__(self, csv_file: str, param_cols: List[str], target_col:str, 
                 time_col:str, Realtime_col:str=None, reduce_to_unique:bool=False):
        """
        Loads and processes data from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.
            param_cols (List[str]): List of parameter column names.
            target_col (str): Name of the target column.
            time_col (str): Name of the time column.
            Realtime_col (str): Name of the real-time column.
            reduce_to_unique (bool, optional): If True, drops duplicate rows based on parameter and target columns. Defaults to False.
        """
        self._param_cols = param_cols
        self._target_col = target_col
        self._time_col = time_col
        self._Realtime_col = Realtime_col
        self._reduce_to_unique = reduce_to_unique
        
        csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        self._data = pd.read_csv(csv_file)
        self._datatime = self._data[[time_col]]
        
        if self._Realtime_col is not None:
            self._dataRealtime = self._data[[Realtime_col]]
        
        if reduce_to_unique:
            n_init_rows = len(self._data)
            unique_data = self._data[param_cols + [target_col]].groupby(param_cols).agg(np.mean).reset_index()
            n_rows = len(unique_data)
            if n_init_rows > n_rows:
                logging.info(f'Duplicate data in {csv_file}. '
                                f'{n_init_rows - n_rows} rows could be dropped '
                                f'({(n_init_rows - n_rows)/n_init_rows*100:.02f}%). '
                                f'leaving {n_rows} rows')
                logging.info(f'Dropping {n_init_rows - n_rows} rows')
            self._data = unique_data
       

    @property
    def X(self):
        return self._data[self._param_cols]

    @property
    def y(self):
        return self._data[self._target_col]
    
    # Aggiustare qua perchè time è la variabile del machine learning
    @property
    def time(self):
        return self._datatime[self._time_col]
    
    @property
    def real_time(self):
        if self._Realtime_col == None: return None
        return self._dataRealtime[self._Realtime_col]
    
    @property
    def folder(self):
        return os.path.dirname(__file__)


LiGenTot = Dataset(
    csv_file='ligen_synth_table.csv',
    param_cols = ['ALIGN_SPLIT',
                  'OPTIMIZE_SPLIT',
                  'OPTIMIZE_REPS',
                  'CUDA_THREADS',
                  'N_RESTART',
                  'CLIPPING',
                  'SIM_THRESH',
                  'BUFFER_SIZE'],
    target_col='RMSD^3*TIME',
    time_col = 'RMSD_0.75',
    Realtime_col='TIME_TOTAL',
    reduce_to_unique=False
)
ScaledLiGenTot = Dataset(
    csv_file='scaledligentot.csv',
    param_cols = ['ALIGN_SPLIT',
                  'OPTIMIZE_SPLIT',
                  'OPTIMIZE_REPS',
                  'CUDA_THREADS',
                  'N_RESTART',
                  'CLIPPING',
                  'SIM_THRESH',
                  'BUFFER_SIZE'],
    target_col='RMSD^3*TIME',
    time_col = 'RMSD_0.75',
    Realtime_col='TIME_TOTAL',
    reduce_to_unique=False
)

Query26 = Dataset(
    csv_file='query26_vm_ram.csv',
    param_cols=['#vm', 'ram'],
    target_col='cost',
    time_col = 'time',
    Realtime_col='time',
    reduce_to_unique=False
)
ScaledQuery26 = Dataset(
    csv_file='scaledQuery26.csv',
    param_cols=['#vm', 'ram'],
    target_col='cost',
    time_col = 'time',
    Realtime_col = 'time',
    reduce_to_unique=False
)

StereoMatch = Dataset(
    csv_file='stereomatch.csv',
    param_cols=['confidence', 'hypo_step', 'max_arm_length', 'num_threads'],
    target_col='cost',
    time_col = 'exec_time_ms',
    Realtime_col = 'exec_time_ms',
    reduce_to_unique=False
)

ScaledStereoMatch = Dataset(
    csv_file='scaledstereomatch.csv',
    param_cols=['confidence', 'hypo_step', 'max_arm_length', 'num_threads'],
    target_col='cost',
    time_col = 'exec_time_ms',
    Realtime_col = 'exec_time_ms',
    reduce_to_unique=False
)

ScaledStereoMatch10 = Dataset(
    csv_file='scaledstereomatch10.csv',
    param_cols=['confidence', 'hypo_step', 'max_arm_length', 'num_threads'],
    target_col='cost',
    time_col = 'exec_time_s',
    Realtime_col = 'exec_time_s',
    reduce_to_unique=False
)
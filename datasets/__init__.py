import logging

import pandas as pd
import numpy as np
import os


_log = logging.getLogger(__name__)
_log.setLevel(logging.DEBUG)


class Dataset:

    def __init__(self, csv_file, param_cols, target_col, reduce_to_unique=False):
        cols = param_cols + [target_col]
        self._param_cols = param_cols
        self._target_col = target_col
        self._reduce_to_unique = reduce_to_unique
        csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        data = pd.read_csv(csv_file, usecols=cols)
        n_init_rows = len(data)
        unique_data = data[param_cols + [target_col]].groupby(param_cols).agg(np.mean).reset_index()
        n_rows = len(unique_data)
        if n_init_rows > n_rows:
            logging.warning(f'Duplicate data in {csv_file}. '
                            f'{n_init_rows - n_rows} rows could be dropped '
                            f'({(n_init_rows - n_rows)/n_init_rows*100:.02f}%). '
                            f'leaving {n_rows} rows')
        if reduce_to_unique:
            logging.warning(f'Dropping {n_init_rows - n_rows} rows')
            self._data = unique_data
        else:
            self._data = data

    @property
    def X(self):
        return self._data[self._param_cols]

    @property
    def y(self):
        return self._data[self._target_col]


LiGen = Dataset(
    csv_file='ligen.csv',
    param_cols=['ALIGN_SPLIT',
                'OPTIMIZE_SPLIT',
                'OPTIMIZE_REPS',
                'CUDA_THREADS',
                'N_RESTART',
                'CLIPPING',
                'SIM_THRESH',
                'BUFFER_SIZE'],
    target_col='AVG_RMSD^3_TIME'
)


Query26 = Dataset(
    csv_file='query26_vm_ram.csv',
    param_cols=['#vm', 'ram'],
    target_col='cost',
    reduce_to_unique=True
)


StereoMatch = Dataset(
    csv_file='stereomatch.csv',
    param_cols=['confidence', 'hypo_step', 'max_arm_length', 'num_threads'],
    target_col='cost'
)

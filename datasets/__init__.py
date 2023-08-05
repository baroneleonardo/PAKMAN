import logging

import pandas as pd
import numpy as np
import os


_log = logging.getLogger(__name__)


class Datasets:

    def __init__(self, csv_file, param_cols, target_col):
        cols = param_cols + [target_col]
        self._param_cols = param_cols
        self._target_col = target_col
        csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        data = pd.read_csv(csv_file, usecols=cols)
        n_init_rows = len(data)
        data = data[param_cols + [target_col]].groupby(param_cols).agg(np.mean).reset_index()
        n_rows = len(data)
        if n_init_rows > n_rows:
            logging.warning(f'Duplicate data in {csv_file}. '
                            f'{n_init_rows - n_rows} have been dropped '
                            f'({(n_init_rows - n_rows)/n_init_rows*100:.02f}%). '
                            f'{n_rows} remaining')
        self._data = data

    @property
    def X(self):
        return self._data[self._param_cols]

    @property
    def y(self):
        return self._data[self._target_col]

    @classmethod
    def LiGen(cls):
        return cls(
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

    @classmethod
    def Query26(cls):
        return cls(
            csv_file='query26_vm_ram.csv',
            param_cols=['#vm', 'ram'],
            target_col='cost'
        )

    @classmethod
    def Stereomatch(cls):
        return cls(
            csv_file='stereomatch.csv',
            param_cols=['confidence', 'hypo_step', 'max_arm_length', 'num_threads'],
            target_col='cost'
        )

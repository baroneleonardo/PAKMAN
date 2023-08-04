import pandas as pd
import os


class Datasets:

    def __init__(self, csv_file, param_cols, target_col):
        cols = param_cols + [target_col]
        self._param_cols = param_cols
        self._target_col = target_col
        csv_file = os.path.join(os.path.dirname(__file__), csv_file)
        self._data = pd.read_csv(csv_file)[cols]

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

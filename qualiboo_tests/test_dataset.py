"""Test Dataset

Test on the Dataset class
"""
import unittest
import pandas as pd
import numpy as np
from qaliboo import datasets


class DatasetTests(unittest.TestCase):

    def test_dataset(self):
        feature_cols = ['#vm', 'ram']
        target_col = 'cost'
        df = pd.read_csv('../qaliboo/datasets/query26_vm_ram.csv')
        df = df[feature_cols + [target_col]]
        n_rows = df.shape[0]
        n_cols = len(feature_cols)

        ds = datasets.Query26
        self.assertIsInstance(ds.X, pd.DataFrame)
        self.assertEqual((n_rows, n_cols), ds.X.shape)
        self.assertIsInstance(ds.y, pd.Series)
        self.assertEqual((n_rows,), ds.y.shape)

        self.assertTrue(np.all(ds.X == df[feature_cols]))
        self.assertTrue(np.all(ds.y == df[target_col]))


if __name__ == '__main__':
    unittest.main()

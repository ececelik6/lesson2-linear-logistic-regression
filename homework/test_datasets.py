import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from homework_datasets import CSVDataset


class DatasetTests(unittest.TestCase):
    def test_missing_values(self):
        """Пропуски не должны оставаться в тензорах."""
        data = pd.DataFrame({
            "age": [20, 30, None, 40, 50, 60, 70, 80, 90, 100],
            "city": ["A", "B"] * 5,
            "target": [0, 1] * 5
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            data.to_csv(path, index=False)
            train, test = CSVDataset.load(path, classification=True)

        self.assertEqual(len(train) + len(test), 10)
        self.assertTrue(torch.isfinite(train.X).all().item())
        self.assertTrue(torch.isfinite(test.X).all().item())

        X, y = train[0]
        self.assertEqual(X.ndim, 1)
        self.assertEqual(y.shape, (1,))

    def test_unknown_category(self):
        """Новая категория в test не должна вызывать ошибку."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder

        processor = ColumnTransformer([
            ("category", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False
            ), ["city"])
        ])

        train = CSVDataset(
            pd.DataFrame({"city": ["A", "B"]}),
            pd.Series([0, 1]),
            processor,
            fit=True
        )
        test = CSVDataset(
            pd.DataFrame({"city": ["C"]}),
            pd.Series([1]),
            processor
        )

        self.assertEqual(train.X.shape[1], test.X.shape[1])
        self.assertTrue(torch.isfinite(test.X).all().item())
        self.assertEqual(test.X.sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
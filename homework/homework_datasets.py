from pathlib import Path
import logging

import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score


logging.basicConfig(level=logging.INFO, format="%(message)s")
folder = Path(__file__).resolve().parent
(folder / "data").mkdir(exist_ok=True)


class CSVDataset(Dataset):
    """Данные CSV, преобразованные в тензоры."""

    def __init__(self, X, y, processor, fit=False):
        if fit:
            X = processor.fit_transform(X)
        else:
            X = processor.transform(X)

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(
            y.to_numpy(), dtype=torch.float32
        ).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    @staticmethod
    def load(path, classification=False):
        """Читаем CSV и готовим train/test."""
        data = pd.read_csv(path)
        X = data.drop(columns="target")
        y = data["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if classification else None
        )

        numeric = X_train.select_dtypes(include="number").columns.tolist()
        categorical = [
            column for column in X_train.columns if column not in numeric
        ]

        processor = ColumnTransformer([
            ("numbers", make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler()
            ), numeric),
            ("categories", make_pipeline(
                SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            ), categorical)
        ], sparse_threshold=0)

        train = CSVDataset(X_train, y_train, processor, fit=True)
        test = CSVDataset(X_test, y_test, processor)
        return train, test


def create_csv():
    """Сохраняем два реальных датасета в CSV."""
    for name, loader in [
        ("diabetes", load_diabetes),
        ("breast_cancer", load_breast_cancer)
    ]:
        data = loader()
        table = pd.DataFrame(data.data, columns=data.feature_names)
        table["target"] = data.target
        table.to_csv(folder / "data" / f"{name}.csv", index=False)


def train_model(name, classification=False):
    """Обучаем регрессию или бинарный классификатор."""
    torch.manual_seed(42)
    train, test = CSVDataset.load(
        folder / "data" / f"{name}.csv", classification
    )
    loader = DataLoader(train, batch_size=32, shuffle=True)
    model = nn.Linear(train.X.shape[1], 1)

    if classification:
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    logging.info(f"\n{name}: train={len(train)}, test={len(test)}")

    for epoch in range(200):
        total_loss = 0
        for X, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(X)

        if (epoch + 1) % 50 == 0:
            logging.info(
                f"Epoch {epoch + 1}: loss={total_loss / len(train):.4f}"
            )

    with torch.no_grad():
        output = model(test.X)

    if classification:
        predictions = (output.numpy().ravel() >= 0).astype(int)
        actual = test.y.numpy().ravel()
        logging.info(f"Accuracy: {accuracy_score(actual, predictions):.4f}")
        logging.info(f"F1: {f1_score(actual, predictions):.4f}")
    else:
        mse = mean_squared_error(test.y.numpy(), output.numpy())
        logging.info(f"Test RMSE: {mse ** 0.5:.2f}")


if __name__ == "__main__":
    create_csv()
    train_model("diabetes")
    train_model("breast_cancer", classification=True)
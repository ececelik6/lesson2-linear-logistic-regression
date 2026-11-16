import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures


folder = Path(__file__).resolve().parent
(folder / "plots").mkdir(exist_ok=True)

# Делим данные на обучающую, проверочную и тестовую части
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Масштабируем целевую переменную по обучающим данным
y_mean = y_train.mean()
y_std = y_train.std()
train_y = torch.tensor(
    (y_train - y_mean) / y_std, dtype=torch.float32
).reshape(-1, 1)


def train(A, B, C, optimizer_name, lr, batch_size):
    """Обучение модели и проверка на validation."""
    torch.manual_seed(42)

    scaler = StandardScaler()
    A = torch.tensor(scaler.fit_transform(A), dtype=torch.float32)
    B = torch.tensor(scaler.transform(B), dtype=torch.float32)
    C = torch.tensor(scaler.transform(C), dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(A, train_y),
        batch_size=batch_size,
        shuffle=True
    )
    model = nn.Linear(A.shape[1], 1)

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

    loss_function = nn.MSELoss()

    for epoch in range(100):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        val_predictions = model(B).numpy().ravel() * y_std + y_mean
        test_predictions = model(C).numpy().ravel() * y_std + y_mean

    val_mse = np.mean((val_predictions - y_val) ** 2)
    return float(val_mse), test_predictions


def save_results(filename, header, rows):
    """Сохранение таблицы результатов."""
    with open(folder / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    # 1. Сравниваем параметры
    results = []

    for name in ["SGD", "Adam", "RMSprop"]:
        for lr in [0.001, 0.01, 0.1]:
            for batch in [16, 64]:
                mse, _ = train(
                    X_train, X_val, X_test, name, lr, batch
                )
                results.append([name, lr, batch, mse])
                print(
                    f"{name}, lr={lr}, batch={batch}: "
                    f"validation MSE={mse:.2f}"
                )

    save_results(
        "hyperparameter_results.csv",
        ["optimizer", "lr", "batch_size", "validation_mse"],
        results
    )

    best = min(results, key=lambda row: row[3])
    print(f"\nЛучшие параметры: {best[:3]}")

    # 2. Создаём новые признаки
    feature_results = []
    predictions = {}

    for mode in ["baseline", "polynomial", "interactions", "statistics"]:
        if mode == "baseline":
            A, B, C = X_train, X_val, X_test

        elif mode in ["polynomial", "interactions"]:
            poly = PolynomialFeatures(
                degree=2,
                include_bias=False,
                interaction_only=(mode == "interactions")
            )
            A = poly.fit_transform(X_train)
            B = poly.transform(X_val)
            C = poly.transform(X_test)

        else:
            # Среднее и дисперсия признаков каждого объекта
            A, B, C = [
                np.column_stack([a, a.mean(axis=1), a.var(axis=1)])
                for a in [X_train, X_val, X_test]
            ]

        mse, test_predictions = train(A, B, C, *best[:3])
        feature_results.append([mode, A.shape[1], mse])
        predictions[mode] = test_predictions
        print(f"{mode}: validation MSE={mse:.2f}")

    save_results(
        "feature_results.csv",
        ["features", "feature_count", "validation_mse"],
        feature_results
    )

    # График сравнения признаков
    plt.bar(
        [row[0] for row in feature_results],
        [row[2] for row in feature_results]
    )
    plt.ylabel("Validation MSE")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(folder / "plots" / "feature_comparison.png")
    plt.close()

    # Выбираем признаки по validation, затем смотрим test
    best_mode = min(feature_results, key=lambda row: row[2])[0]
    print(f"\nЛучшие признаки: {best_mode}")

    for mode in dict.fromkeys(["baseline", best_mode]):
        test_mse = np.mean((predictions[mode] - y_test) ** 2)
        print(f"{mode}: test RMSE={test_mse ** 0.5:.2f}")
import copy
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.datasets import load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, ConfusionMatrixDisplay
)


logging.basicConfig(level=logging.INFO, format="%(message)s")
folder = Path(__file__).resolve().parent
(folder / "models").mkdir(exist_ok=True)
(folder / "plots").mkdir(exist_ok=True)


def tensor(data):
    return torch.tensor(data, dtype=torch.float32)


def linear_regression():
    """Линейная регрессия с L1, L2 и early stopping."""
    torch.manual_seed(42)
    X, y = load_diabetes(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = tensor(scaler.fit_transform(X_train))
    X_val = tensor(scaler.transform(X_val))
    X_test = tensor(scaler.transform(X_test))

    y_train = tensor(y_train).reshape(-1, 1)
    y_val = tensor(y_val).reshape(-1, 1)
    y_test = tensor(y_test).reshape(-1, 1)

    model = nn.Linear(X_train.shape[1], 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    wait = 0
    train_losses, val_losses = [], []

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()
        mse = criterion(model(X_train), y_train)

        # L1 и L2 применяем только к весам
        l1 = model.weight.abs().sum()
        l2 = model.weight.square().sum()
        loss = mse + 0.01 * l1 + 0.01 * l2

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val), y_val).item()

        train_losses.append(mse.item())
        val_losses.append(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if (epoch + 1) % 50 == 0:
            logging.info(
                f"Epoch {epoch + 1}: validation MSE={val_loss:.2f}"
            )

        if wait >= 30:
            logging.info(f"Early stopping: epoch {epoch + 1}")
            break

    model.load_state_dict(best_weights)

    with torch.no_grad():
        test_mse = criterion(model(X_test), y_test).item()

    logging.info(f"Linear test RMSE: {test_mse ** 0.5:.2f}")
    torch.save(model.state_dict(), folder / "models" / "linear.pth")

    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder / "plots" / "linear_training.png")
    plt.close()


def logistic_regression():
    """Многоклассовая классификация на Iris."""
    torch.manual_seed(42)
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=0.2, random_state=42, stratify=data.target
    )

    scaler = StandardScaler()
    X_train = tensor(scaler.fit_transform(X_train))
    X_test = tensor(scaler.transform(X_test))
    y_train = torch.tensor(y_train, dtype=torch.long)

    model = nn.Linear(X_train.shape[1], 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    with torch.no_grad():
        probabilities = torch.softmax(model(X_test), dim=1).numpy()
        predictions = probabilities.argmax(axis=1)

    logging.info(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    logging.info(
        f"Precision: {precision_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    logging.info(
        f"Recall: {recall_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    logging.info(
        f"F1: {f1_score(y_test, predictions, average='macro', zero_division=0):.4f}"
    )
    logging.info(
        f"ROC-AUC: {roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro'):.4f}"
    )

    torch.save(model.state_dict(), folder / "models" / "logistic.pth")

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=data.target_names, cmap="Blues"
    )
    plt.tight_layout()
    plt.savefig(folder / "plots" / "confusion_matrix.png")
    plt.close()

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(folder / "plots" / "classification_training.png")
    plt.close()


if __name__ == "__main__":
    linear_regression()
    logistic_regression()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, log_epoch, ClassificationDataset


class LogisticRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


if __name__ == '__main__':
    # Загружаем датасет breast_cancer
    X, y = make_classification_data(source='breast_cancer')

    # Масштабируем признаки
    feature_mean = X.mean(dim=0)
    feature_std = X.std(dim=0).clamp_min(1e-8)
    X = (X - feature_mean) / feature_std

    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            # Считаем ошибку и количество верных ответов
            total_loss += loss.item() * batch_X.size(0)
            predictions = (logits.detach() > 0).float()
            total_correct += (predictions == batch_y).sum().item()
            total_samples += batch_X.size(0)

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)

    # Сохраняем веса модели
    torch.save(model.state_dict(), 'logreg_torch.pth')

    # Сохраняем параметры масштабирования
    torch.save(
        {'mean': feature_mean, 'std': feature_std},
        'logreg_scaler.pth'
    )

    # Загружаем модель
    new_model = LogisticRegression(in_features=X.shape[1])
    new_model.load_state_dict(
        torch.load('logreg_torch.pth', weights_only=True)
    )
    new_model.eval()
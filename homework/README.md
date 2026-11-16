# Урок 2. Линейная и логистическая регрессия

## 1. Модификация моделей

В файле homework_model_modification.py реализованы:
- линейная регрессия с L1 и L2 регуляризацией;
- early stopping по ошибке на проверочной выборке;
- многоклассовая логистическая регрессия на датасете Iris;
- метрики precision, recall, F1, ROC-AUC и матрица ошибок.

Результаты:
- Diabetes: test RMSE = 53.73.
- Iris: accuracy = 0.9667, precision = 0.9697,
  recall = 0.9667, F1 = 0.9666, ROC-AUC = 1.0000.

Early stopping реализован, но в последнем запуске обучение
достигло 500 эпох, так как ошибка продолжала уменьшаться.

## 2. Работа с CSV

В файле homework_datasets.py создан класс CSVDataset.
Он поддерживает числовые и категориальные признаки.
Пропуски заполняются, числовые признаки масштабируются,
категориальные кодируются с помощью OneHotEncoder.
Параметры обработки определяются только по обучающим данным.

Датасеты Diabetes и Breast Cancer из sklearn сохраняются
в CSV, затем читаются из этих файлов.

Результаты:
- Diabetes: test RMSE = 55.87.
- Breast Cancer: accuracy = 0.9825, F1 = 0.9861.

## 3. Эксперименты

В файле homework_experiments.py сравниваются:
- оптимизаторы SGD, Adam и RMSprop;
- learning rate: 0.001, 0.01, 0.1;
- batch size: 16 и 64.

Всего проверено 18 комбинаций.
Лучшая комбинация по validation MSE:
Adam, learning rate = 0.1, batch size = 16.

Также проверены полиномиальные признаки, взаимодействия
и статистические признаки: среднее и дисперсия.

В этом эксперименте новые признаки не улучшили результат.
Лучшим оказался исходный набор признаков:
validation MSE = 2828.61, test RMSE = 52.17.

Параметры выбирались по проверочной выборке.
Тестовая выборка использовалась для итоговой оценки.

## 4. Проверки

В test_datasets.py проверяются обработка пропусков
и поддержка неизвестных категорий. Оба теста пройдены.

## Запуск

Команды выполняются из корневой папки проекта:

```powershell
.\.venv\Scripts\python.exe .\homework\homework_model_modification.py
.\.venv\Scripts\python.exe .\homework\homework_datasets.py
.\.venv\Scripts\python.exe .\homework\homework_experiments.py
.\.venv\Scripts\python.exe -m unittest discover -s homework -p "test_*.py" -v
```

Зависимости: torch, scikit-learn, numpy, matplotlib, pandas.
Таблицы результатов находятся в папке homework,
графики — в homework/plots, веса моделей — в homework/models.
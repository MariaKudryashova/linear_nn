#### Описание проекта
В проекте представлены разнообразные линейные модели собственной реализации в сравнении с базовыми реализациям библиотеки `sklearn`

1. Раздел Classifications
В разделе сравниваются линейные модели для задач классификации собственной реализации и линейные модели библиотеки `sklearn`

В качестве входных данных используется функция `make_blobs` библиотеки `sklearn`, которая создает набор данных, представляющий собой набор кластеров точек с некоторыми заданными характеристиками, где каждый кластер представляет собой группу точек, близких друг к другу. Часто используется для иллюстрации алгоритмов кластеризации (классификации).

Линейных моделей может быть бесконечно много, с различными комбинациями их основных свойств. Кратко опишем их:
1. Регрессия может быть линейной или логистической
2. Регрессия может быть простая или с регуляризацией:
    - L1 (Lasso), 
    - L2 (Ridge),
    - ElasticNet или эластичной регуляризацией (сочетающей L1 и L2)
3. Функция потерь может быть:
    - LogLoss логарифмической,
    - Mean Squared Error среднеквадратичной
4. Для минимизации функции потерь можно использовать разнообразные виды метода градиента:
    - простой
    - стохастический (SGD)
5. Метрики оценки регрессии могут быть:
    - средняя квадратичная MSE
    - средняя абсолютная MAE
    - корень из средней квадратичной EMSE
    - процентная средняя абсолютная ошибка MAPE
    - коэффициент детерминации

В нашем примере реализованы:
    - MyLogisticRegression логистическая регрессия
    - MyLogisticSGDRegression логистическая регрессия с использованием стохастического градиентного спуска
    - MyElasticLogisticRegression логистическая регрессия с использованием ElasticNet регуляризации
И дополнительно использованы готовые модели из `sklearn`:
    - KNeighborsClassifier модель на основе метода ближайших соседей
    - LogisticRegression логистическая регрессия

Компоненты:
    - `start.py` вызов и работа с данными моделями, сравнение их точностей
    - `find_better_params.py` использует `GridSearchCV` для поиска по сетке оптимальных параметров для моделей, в данном примере для модели KNeighborsClassifier

2. Раздел ConvNet
Реализация сверточной нейронной сети ConvNet. Обычно используется для работы с изображениями. Основная идея в использовании сверточных слоев для автоматического извлечения иерархии признаков из входных данных. В качестве обучающего используется популярный датасет прописных арабских цифр MNIST

Компоненты:
- `ConvNet.py` реализация самой модели сверточной нейронной сети
- `train.py` параметры обучения модели, загрузка обучающего дата сета `MNIST`, сохранение модели `ConvNet_CE` в папку `Models`
- `start.py` полный цикл работы с моделью: обучение, применение, оценка точности
- `convnet_model.ckpt` сохраненные параметры модели, обученной на заданном датасете

3. Раздел `FC` 
Включена работа с простой полносвязной нейронной сетью. Обычно используются для задач классификации. Такая сеть состоит из слоев нейронов, где каждый нейрон связан со всеми нейронами в предыдущем и следующем слоях.
Компоненты:
    - `start_train.py` задает слои полносвязной сети, проводит обучение на датасете MNIST и сохраняет гиперпараметры модели `FC_ReLU_CE`
    - `start_predict.py` демонстрирует работу с любой похожей сохраненной моделью, подается число и модель его распознает

4. Раздел `LeNet`
Реализация классической сверточной нейронной сети `LeNet` (изначально предложена Яном ЛеКуном в 1998 году) Эта модель заложила стандарты по использованию сверточных сетей в компьютерном зрении.
Компоненты:
    - `start_train.py` задает слои сверточной нейронной сети, проводит обучение на датасете MNIST и сохраняет гиперпараметры модели
    - `start_predict.py` демонстрирует работу с любой похожей сохраненной моделью, подается число и модель его распознает. Дополнительно выводятся весовые коэффициенты по классам.


5. Раздел `LinReg`
Реализация линейных моделей для задач регрессии
    - `MyGradientLinearRegression` линейная регрессия с градиентным методом поиска функции ошибок
    - `MyLinearRegression` простая линейная регрессия
    - `MyRidgeRegression` линейная регрессия с регуляризацией L2
    - `MySGDLasso` линейная регрессия с использованием стохастического градиентного спуска и регуляризации L1
    - `MySGDLinearRegression`  линейная регрессия с использованием стохастического градиентного спуска
    - `MySGDRidge` - линейная регрессия с использованием стохастического градиентного спуска и регуляризации L2

6. Раздел `Perseptron`
полная собственная реализация многослойно перцептрона (`Multilayer Perceptron`)

Общие разделы:
    - `Comparisons` раздел с сохранением разнообразных сравнений моделей
    - `Loaders` раздел с датасетами по всему проекту
    - `Models` раздел с сохраненными моделями проекта
    
7. Общие компоненты:
    - `Saver.py` сохранение параметров модели, оценок качества моделей 
    - `train_loop.py` цикл обучения, все модели для обучения используют его
    - `test_lossfunctions.py` модуль сравнения различных функций потерь на одной какой-нибудь модели, в данном примере на полносвязной нейронной сети
    - `test_activation_functions.py` модуль сравнения различных функций активации на какой-нибудь отдельной модели, в данном примере тоже на простой полносвязной сети
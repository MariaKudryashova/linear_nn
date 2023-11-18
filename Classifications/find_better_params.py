from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_blobs(n_samples=1000, centers=[[-2,0.5],[2,-0.5]], cluster_std=1, random_state=42)

# выделим половину объектов на тест
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

print("KNeighborsClassifier")
reg = KNeighborsClassifier(n_neighbors=3, p=2)
reg.fit(X_train, y_train)
preds = reg.predict(X_test)
print('Test MSE: ', mean_squared_error(y_test, preds))
print("")


params = {
    'n_neighbors': np.arange(1, 10), 
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

neigh_grid = GridSearchCV(neigh, params, cv=5, scoring='accuracy', n_jobs=-1)
# p = 1 Manhattan 
# p = 2 Euclidean 

# запустим поиск
neigh_grid.fit(feature_matrix, labels)
print(neigh_grid.best_params_)

# создание модели с указанием гиперпараметра C
optimal_neigh = KNeighborsClassifier(n_neighbors=4, p=1,weights='distance')

# обучение модели
optimal_neigh.fit(train_feature_matrix, train_labels)

# предсказание на тестовой выборке
optimal_y_pred = optimal_neigh.predict(test_feature_matrix)

optimal_y_pred_proba = optimal_neigh.predict_proba(test_feature_matrix)
# Лабораторная работа 5. Алгоритмы кластеризации данных
## Задание
Перед выполнением лабораторной работы необходимо загрузить набор данных в соответствии с вариантом на диск.

1. Произвести масштабирование признаков (scaling).
2. С использованием библиотеки scikit-learn написать программу с использованием алгоритмов кластеризации данных, позволяющую разделить исходную выборку на классы, соответствующие предложенной вариантом задаче (http://scikit-learn.org/stable/modules/clustering.html).
3. Провести эксперименты и определить наилучший алгоритм кластеризации, параметры алгоритма. Необходимо использовать не менее 3-х алгоритмов. Данные экспериментов необходимо представить в отчете (графики, ход проведения эксперимента, выводы).
## Вариант 2
Adult

## Загрузка dataset и его подготовка
```
# Загрузка данных из файла
file_path = 'water-treatment.txt'
data = pd.read_csv(file_path)

# Пропущенные значения в вашем dataset обозначены как '?', заменим их на NaN
data = data.replace('?', np.nan)

# Для числовых столбцов преобразуем данные в float
# Первый столбец - дата, остальные - числовые параметры
numeric_columns = data.columns[1:]  # все столбцы кроме первого (дата)
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Выбираем только числовые признаки для кластеризации
X = data[numeric_columns]

# Заполняем пропущенные значения медианой каждого столбца
X_filled = X.fillna(X.median())

# Масштабируем признаки
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filled)

# Преобразуем в плотную матрицу (уже плотная, но для consistency)
X_scaled_dense = X_scaled
```

## Функция для оценки кластеризации
```
def evaluate_clustering(model, data):
    labels = model.fit_predict(data)
    # Проверка, что есть более 1 кластера
    if len(set(labels)) > 1 and -1 in labels:
        core_labels = labels[labels != -1]
        core_data = data[labels != -1]
        score = silhouette_score(core_data, core_labels)
    elif len(set(labels)) > 1:
        score = silhouette_score(data, labels)
    else:
        score = -1
    return labels, score
```
core_data — массив данных, для которых считается метрика.
core_labels — метки кластеров для этих данных, то есть, к каким кластерам принадлежат точки.

## Метод кластеризации - KMeans

```
# 1. KMeans
# 1. KMeans
kmeans_params = [2, 3, 4, 5, 6]
best_score_kmeans = -1
best_kmeans = None
best_labels_kmeans = None
best_k = None
labels_for_k = []  # список для хранения меток для каждого k

for k in kmeans_params:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels, score = evaluate_clustering(kmeans, X_scaled_dense)
    print(f'KMeans с k={k}, Силуэтный коэффициент: {score:.3f}')
    labels_for_k.append(labels)  # сохраняем метки для каждого k
    if score > best_score_kmeans:
        best_score_kmeans = score
        best_kmeans = kmeans
        best_labels_kmeans = labels  # метки для лучшего k
        best_k = k

print(f'Лучшее число кластеров для KMeans: {best_k} с коэффициентом: {best_score_kmeans:.3f}')

# Визуализация для каждого k
fig, axes = plt.subplots(1, len(kmeans_params), figsize=(13, 4))
for i, k in enumerate(kmeans_params):
    labels = labels_for_k[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'k={k}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

K-Means (К-средних) — это один из самых популярных и широко используемых алгоритмов кластеризации в машинном обучении. Он относится к категории неконтролируемого обучения, что означает, что алгоритм работает с данными без заранее известных меток или категорий.

K — количество кластеров, которые мы хотим найти
Means (средние) — алгоритм использует средние значения точек для определения центров кластеров

Основная задача K-Means — автоматически группировать похожие объекты в кластеры на основе их характеристик:

Объединяет похожие данные в группы (кластеры)
Разделяет различные данные на разные группы
Находит скрытые паттерны в данных без какого-либо предварительного знания о том, какие группы должны существовать


## Метод кластеризации - Agglomerative Clustering

```# 2. Agglomerative Clustering
agg_params = [2, 3, 4, 5, 6]
best_score_agg = -1
best_labels_agg = []
best_n_agg = None

# В список для хранения меток для каждого k
labels_list = []

for n in agg_params:
    agg = AgglomerativeClustering(n_clusters=n)
    labels, score = evaluate_clustering(agg, X_scaled_dense)
    print(f'Agglomerative с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list.append(labels)
    if score > best_score_agg:
        best_score_agg = score
        best_labels_agg = labels
        best_n_agg = n

print(f'Лучшее число кластеров для Agglomerative: {best_n_agg} с коэффициентом: {best_score_agg:.3f}')

# Визуализация для всех вариантов
fig, axes = plt.subplots(1, len(agg_params), figsize=(13, 4))
for i, n in enumerate(agg_params):
    labels = labels_list[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```
Agglomerative Clustering (Агломеративная кластеризация) — это иерархический алгоритм кластеризации, который работает по принципу "снизу вверх". Это означает, что кластеры могут содержать внутри себя другие кластеры, образуя многоуровневую структуру.

Принцип "снизу вверх" (также называемый агломеративным подходом) — это стратегия, при которой процесс начинается с самых мелких элементов и постепенно объединяет их во все более крупные группы.


## Метод кластеризации - SpectralClustering

```
# 3. SpectralClustering
spectral_params = [2, 3, 4, 5, 6]
best_score_spectral = -1
best_labels_spectral = []
best_n_spectral = None

# Список для хранения меток для каждого k
labels_list_spectral = []

for n in spectral_params:
    spectral = SpectralClustering(n_clusters=n, affinity='nearest_neighbors', random_state=42)
    labels, score = evaluate_clustering(spectral, X_scaled_dense)
    print(f'SpectralClustering с n_clusters={n}, Силуэтный коэффициент: {score:.3f}')
    labels_list_spectral.append(labels)
    if score > best_score_spectral:
        best_score_spectral = score
        best_labels_spectral = labels
        best_n_spectral = n

print(f'Лучшее число кластеров для SpectralClustering: {best_n_spectral} с коэффициентом: {best_score_spectral:.3f}')

# Визуализация результатов для всех вариантов
fig, axes = plt.subplots(1, len(spectral_params), figsize=(13, 4))
for i, n in enumerate(spectral_params):
    labels = labels_list_spectral[i]
    axes[i].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
    axes[i].set_title(f'n={n}')
    axes[i].set_xlabel('PC1')
    axes[i].set_ylabel('PC2')

plt.tight_layout()
plt.show()
```

Spectral Clustering (Спектральная кластеризация) — это алгоритм кластеризации, основанный на теориях из линейной алгебры и спектральной теории графов. Spectral Clustering использует собственные значения и собственные векторы матриц для преобразования данных в более удобное пространство.

Основная задача Spectral Clustering — обнаруживать сложные нелинейные структуры в данных, которые традиционные алгоритмы (как K-Means) не могут выявить.

## Определение лучшего метода

```
# Итог: определение лучшего метода
scores = {
    'KMeans': best_score_kmeans,
    'Agglomerative': best_score_agg,
    'SpectralClustering': best_score_spectral
}

best_method = max(scores, key=scores.get)
print(f'Лучший метод кластеризации: {best_method} с коэффициентом {scores[best_method]:.3f}')
```







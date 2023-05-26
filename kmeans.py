import pandas as pd
import numpy as np
from sklearn import tree
import time
from sklearn.preprocessing import LabelEncoder
import graphviz
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("diabetes_prediction_dataset.csv")

le_sex = LabelEncoder()
le_smoke = LabelEncoder()

df['gender'] = le_sex.fit_transform(df['gender'])
df['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])

X = df.drop('diabetes', axis=1)
y = df['diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados
scaler = StandardScaler()

scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

km = KMeans(n_clusters=2)
km.fit(scaled_X_train)

y_train_pred = km.predict(scaled_X_train)
y_test_pred = km.predict(scaled_X_test)

# Create an instance of the DecisionTreeClassifier
model = tree.DecisionTreeClassifier(max_depth=22, min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=22)

# Treinar o modelo de árvore de decisão usando os dados de treinamento
model.fit(X_train, y_train)

# Calcular a pontuação de precisão do modelo nos dados de teste
print(f'Training accuracy tree: {model.score(X_train, y_train)*100}%')
print(f'Testing accuracy tree: {model.score(X_test, y_test)*100}%')

# Calculate the accuracy
accuracy_train = np.sum(y_train_pred == y_train) / len(y_train)
accuracy_test = np.sum(y_test_pred == y_test) / len(y_test)

print(f'Training accuracy k-means: {accuracy_train * 100}%')
print(f'Testing accuracy k-means: {accuracy_test * 100}%')
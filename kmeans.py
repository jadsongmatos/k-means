import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("diabetes_prediction_dataset.csv")

# Drop rows with missing values and reset the index
df = df.dropna().reset_index(drop=True)

# Use LabelEncoder to encode categorical variables to numeric values
le_sex = LabelEncoder()

df['gender'] = le_sex.fit_transform(df['gender'])
df['smoking_history'] = le_sex.fit_transform(df['smoking_history'])

print(df.dtypes)

X_train, X_test = train_test_split(df, test_size=0.2, random_state=42)

inputs = pd.DataFrame()
inputs['gender'] = le_sex.fit_transform(X_train['gender'])
inputs['age'] = X_train['age']
inputs['smoking_history'] = le_sex.fit_transform(X_train['smoking_history'])
inputs['blood_glucose_level'] = X_train['blood_glucose_level']
inputs['bmi'] = X_train['bmi']
inputs['HbA1c_level'] = X_train['HbA1c_level']
inputs['hypertension'] = X_train['hypertension']
inputs['heart_disease'] = X_train['heart_disease']
inputs = inputs.dropna()

inputs_test = pd.DataFrame()
inputs_test['gender'] = X_test['gender']
inputs_test['age'] = X_test['age']
inputs_test['smoking_history'] = X_test['smoking_history']
inputs_test['blood_glucose_level'] = X_test['blood_glucose_level']
inputs_test['bmi'] = X_test['bmi']
inputs_test['HbA1c_level'] = X_test['HbA1c_level']
inputs_test['hypertension'] = X_test['hypertension']
inputs_test['heart_disease'] = X_test['heart_disease']
inputs_test = inputs_test.dropna()

plt.scatter(df['age'], df['bmi'], c=df['diabetes'], alpha=0.6)

plt.xlabel('age')
plt.ylabel('bmi')
plt.show()

target = le_sex.fit_transform(X_test['diabetes'])

scaler = StandardScaler()

scaled_inputs = scaler.fit_transform(inputs)
scaled_inputs_test = scaler.fit_transform(inputs_test)

km = KMeans(n_clusters=2)
km.fit(scaled_inputs)
y_km = km.predict(scaled_inputs)

# Visualizando os clusters
plt.scatter(inputs['age'], inputs['bmi'], c=y_km, alpha=0.6)
plt.xlabel('age')
plt.ylabel('bmi')
plt.show()

target = le_sex.fit_transform(X_test['diabetes'])

result = km.predict(scaled_inputs_test)


# Calculando a diferença e a soma da diferença
print((np.sum(result - target) * 100)/target.size)



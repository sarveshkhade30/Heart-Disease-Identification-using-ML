import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
from tensorflow import keras
from keras import layers
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the dataset
data = pd.read_csv('heart.csv')

# Data Scaling
y = data['target']
x = data.drop('target', axis=1)

scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7)

# Balancing Datas
sm = SMOTE(random_state=0)
X_train, Y_train = sm.fit_resample(X_train, Y_train)

# Best Model
model_params = {
    'logistic': {
        'model': LogisticRegression(random_state=0),
        'param': {
            'C': [1, 5, 10],

        }
    },
    'svm': {
        'model': SVC(random_state=0, probability=True),
        'param': {
            'C': [1, 10, 20],
            'kernel': ['rbf', 'linear'],
            'degree': [1, 5, 10]
        }
    },
    'KNN': {
        'model': KNeighborsClassifier(),
        'param': {
            'n_neighbors': [1, 5, 20],
            'leaf_size': [1, 10, 50]
        }
    },
    'decision_tree': {
        'model': tree.DecisionTreeClassifier(),
        'param': {

        }
    }
}

scores = []

for model_name, mp in model_params.items():
    clf = GridSearchCV(mp['model'], mp['param'], cv=10, return_train_score=False)
    clf.fit(X_train, Y_train)
    scores.append({
        'model': model_name,
        'best_score': clf.best_score_,
        'best_params': clf.best_params_
    })

df = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])

# Model Training
model1 = LogisticRegression()
model1.fit(X_train, Y_train)

model2 = RandomForestRegressor()
model2.fit(X_train, Y_train)

model3 = keras.Sequential([
    layers.Dense(36, activation='relu', input_shape=[13]),
    layers.Dense(26, activation='relu'),
    layers.Dense(15, activation='relu'),
    layers.Dense(10, activation='relu'),
    layers.Dense(5, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
model3.compile(optimizer='adam',
               loss='binary_crossentropy',
               metrics=['accuracy'])
model3.fit(X_train, Y_train, epochs=400)

model4 = XGBRegressor(n_estimators=500)
model4.fit(X_train, Y_train)

model5 = KNeighborsClassifier(leaf_size=1, n_neighbors=1)
model5.fit(X_train, Y_train)

model6 = tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=1, min_samples_split=5)
model6.fit(X_train, Y_train)

# Save the models and scaler using pickle
pickle.dump(model1, open('heart_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Since the neural network creates several models, we need to find the best among them.
# Usually the least MSE error model, comes the top, when sorted by name.
# This validation is a confirmation that it is the best model obtained.

from keras.models import Sequential
from keras.layers import Dense, Dropout


def mlp(dropout_rate=0.0, activation1='relu', activation2='relu', activation3='relu', activation4='relu',
        activation5='relu', activation6='relu'):
    model = Sequential()
    model.add(Dense(1024, input_dim=len(X.columns), kernel_initializer='normal', activation=activation1))
    model.add(Dropout(dropout_rate))
    model.add(Dense(512, kernel_initializer='normal', activation=activation2))
    model.add(Dropout(dropout_rate))
    model.add(Dense(256, kernel_initializer='normal', activation=activation3))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, kernel_initializer='normal', activation=activation4))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, kernel_initializer='normal', activation=activation5))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, kernel_initializer='normal', activation=activation6))
    model.add(Dense(4, kernel_initializer='normal', activation='relu'))
    # Compile model
    model.compile(loss='mae', optimizer='adam', metrics=['mae'])
    return model


# Splitting the data once again into inputs and outputs.
import pandas as pd

df = pd.read_csv('train.csv')
X = df.loc[:, 'hour_of_day':'traffic_type_10']
Y = df.loc[:, 'hits']

# Here we only need the test dataset to test the models. Previously pickled scaler is reused on test data.
from sklearn.model_selection import train_test_split
_, X_test, __, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
import joblib
scaler = joblib.load('scaler.pkl')
X_test.loc[:, 'hour_of_day':'path_length'] = scaler.transform(X_test.loc[:, 'hour_of_day':'path_length'])

# Testing a small set of best possible models and storing it to a csv file.
import glob
# path = 'models'
path = '/media/raghu/6A3A-B7CD/PredictHits'
files = sorted(glob.glob(path + '/model-10*.hdf5'), reverse=False)

# It looks like the best model is model-1008.6310.hdf5. MSE of 31.7 on validation dataset
from sklearn.metrics import mean_squared_error
import csv
from keras.models import load_model
from math import sqrt

for file in files[0:50]:
    model = load_model(file)
    Y_test_pred = model.predict(X_test)
    mse = mean_squared_error(Y_test_pred, Y_test)
    key = file.split('/')[-1].split('.hdf5')[0]
    print(key, sqrt(mse))
    with open('validation_results.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([key, sqrt(mse)])

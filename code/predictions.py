# Predictions are based the best model.

from keras.models import Sequential
from keras.layers import Dense,Dropout

def mlp(dropout_rate=0.0,activation1='relu', activation2='relu', activation3='relu', activation4='relu',
                       activation5='relu',activation6='relu'):
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

# Transforming the given test set in the same way as the training set.
import pandas as pd
df = pd.read_csv('test.csv', index_col=0)
X = df.loc[:, 'hour_of_day':'traffic_type_10']

# Using the pickled scaler once again to transform the data.
import joblib
scaler = joblib.load('scaler.pkl')
X.loc[:, 'hour_of_day':'path_length'] = scaler.transform(X.loc[:, 'hour_of_day':'path_length'])

from keras.models import load_model
path = 'models'
model = load_model(path +'/model-1008.6310.hdf5')

# Saving the predictions in the specified format.
predictions_df = pd.DataFrame(model.predict(X),columns=['hits'])

predictions_df.index = df.index
predictions_df.to_csv('predictions.csv')


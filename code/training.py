
# Caution: 1. Before training please check the comments below about the Keras regressor.
#          2. This creates a lot of checkpoints in models folder.

# Loading the data and splitting it to inputs and ouputs
import pandas as pd
df = pd.read_csv('train.csv')
X = df.loc[:, 'hour_of_day':'traffic_type_10']
Y = df.loc[:, 'hits']

# Splitting the train and test data
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)

# The data is standardized. It is reconcatinated back, as there is a validation split for early stopping (check last few lines).
# Scaler is pickled, for later data transforms.
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
scaler = StandardScaler(with_mean=False)
X_train.loc[:, 'hour_of_day':'path_length'] = scaler.fit_transform(X_train.loc[:, 'hour_of_day':'path_length'])
X_test.loc[:, 'hour_of_day':'path_length'] = scaler.transform(X_test.loc[:, 'hour_of_day':'path_length'])
X_scaled = np.concatenate((X_train, X_test))
joblib.dump(scaler, "scaler.pkl")

# A simple Keras sequential multilayer neural network, this usually solves all kinds of prediction problems
# provided dropout is added to prevent overfitting.
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
    model.add(Dense(1, kernel_initializer='normal', activation='elu'))
    # Compile model
    model.compile(loss='mse', optimizer='adam', metrics=['mse'] )
    return model

# Want to see same results everytime I run this
import numpy as np
seed = 7
np.random.seed(seed)

# Keras callbacks for checkpointing and earlystopping
from keras.callbacks import ModelCheckpoint, EarlyStopping
outputFolder = 'models'
filepath = outputFolder + "/model-{val_loss:.4f}.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_mean_squared_error', verbose=1, \
                             save_best_only=True, save_weights_only=False, \
                             mode='auto', period=1)
earlystop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')

callbacks_list = [earlystop, checkpoint]

# Grid serach based on activations, batchsize and dropout
activation1 = ['elu', 'sigmoid']
activation2 = ['elu', 'sigmoid']
activation3 = ['elu', 'sigmoid']
activation4 = ['elu', 'sigmoid']
activation5 = ['elu', 'sigmoid']
activation6 = ['elu', 'sigmoid']

dropout_rate = [0.15, 0.2]

# Batch size based on the size of the sample
(row_size, _) = np.shape(X_train)
min_bsize = int(row_size / 100)
batch_size = [min_bsize, 2 * min_bsize]

param_grid = dict(activation1=activation1, activation2=activation2, activation3=activation3, activation4=activation4,
                  activation5=activation5, activation6=activation6, batch_size=batch_size, dropout_rate=dropout_rate)

# Here keras regressor doesn't have callbacks in the original code. So the source code should be tweaked at bit to include it.
# please refer to this link. https://github.com/keras-team/keras/issues/4278#issuecomment-258922449
from keras.wrappers.scikit_learn import KerasRegressor
estimator = KerasRegressor(build_fn=mlp, epochs=150, verbose=2, validation_split=0.2,
                           callbacks=callbacks_list)

# using scikit learns GridSearchCV to iterate over paramgrid and
# use estimator defined above. Shufflesplit is used here skip the
#  cross validation on several folds, rather only use the gridsearch feature.
from sklearn.model_selection import GridSearchCV, ShuffleSplit
grid = GridSearchCV(estimator, param_grid=param_grid, cv=ShuffleSplit(test_size=0.01, n_splits=1),
                    scoring='neg_mean_absolute_error', verbose=2)
# Careful before starting training. This creates more than 200 models due to checkpointing.
grid_result = grid.fit(X_scaled, Y.to_numpy())

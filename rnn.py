import tensorflow as tf
from tensorflow import keras
from math import sqrt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import matplotlib.pyplot as plt

import numpy as np
def build_model():
    model = keras.Sequential([
        keras.layers.SimpleRNN(3, input_shape=(train_x.shape[1], train_x.shape[2])),
        keras.layers.Dense(1)
    ])

    opt = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse', 
            optimizer=opt,
            metrics=['mae'])
    return model

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def getData():
    f = open("load.txt", "r")
    # Remove column headers
    train_data, train_labels, test_data, test_labels = [], [], [], []
    lines = f.readlines()[1:]
    for line in lines:
        line = line.split()
        # element = (time, t_db, RH, Rad, load)
        element = np.array([line[0], line[1], line[2], line[3], line[4]])
        # Convert into floats
        element = np.asarray(element, dtype = float)
        # Add element to data
        element = element[1:5]
        train_data.append(element)
        #train_data.append(np.concatenate(element[4],0, element.pop()))
    f.close()
    return np.array(train_data), np.array(train_labels)


train_data, train_labels = getData()

#May need tanh to normalise since t_db has negative values
scaler = MinMaxScaler(feature_range=(0, 1))
# Alternative scaler
#scaler = StandardScaler()
train_data = scaler.fit_transform(train_data)
lag_timesteps = 3
features = train_data.shape[0]

train_data = series_to_supervised(train_data, 1, 1)
train_data.drop(train_data.columns[[5,6,7]], axis=1, inplace=True)

train_data = train_data.values
train_timesteps = 7000 
train = train_data[:train_timesteps, :]
test = train_data[train_timesteps:, :]
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

print(train_data[:5])

model = build_model()
model.summary()
#
EPOCHS = 100
#
## The patience parameter is the amount of epochs to check for improvement

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(train_x, train_y, epochs=EPOCHS, 
        validation_data=(test_x, test_y), verbose=2, 
        shuffle=False, callbacks=[early_stop])
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
# make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# invert scaling for forecast
inv_yhat = np.concatenate((test_x[:,1:],yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_x[:,1:],test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)
print('Average test accuracy: %.3f' % (100-(mae*100/7000))+'%')

plt.clf()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.scatter(inv_yhat, inv_y, s=1)
_ = plt.plot([-4000,4000], [-4000,4000])
plt.show()


plt.clf()
plt.xlabel('Time')
plt.ylabel('Predictions/Actual')
time = [i for i in range(len(inv_y))]
plt.scatter(time, inv_y, s=1, label='Actual Load')
plt.scatter(time,inv_yhat, s=1, label='Predicted Load')
plt.legend()
plt.show()

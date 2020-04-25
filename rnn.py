#!/usr/bin/env python
# coding: utf-8

# In[8]:


from pandas import read_csv, DataFrame, concat
import tensorflow as tf
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from numpy import concatenate
from math import sqrt


# In[ ]:





# In[9]:


dataset = read_csv("load.txt", delim_whitespace=True)
values = dataset.values
print(dataset.head())
# specify columns to plot
groups = [0, 1, 2, 3, 4]
i = 1
# plot each column
pyplot.figure()
for group in groups:
	pyplot.subplot(len(groups), 1, i)
	pyplot.plot(values[:, group])
	pyplot.title(dataset.columns[group], y=0.5, loc='right')
	i += 1
pyplot.show()

dataset.drop(dataset.columns[[0]], axis=1, inplace=True)
print(dataset.head())
values = dataset.values

print(dataset.loc[dataset['load'].idxmax()])
print(dataset.loc[dataset['load'].idxmin()])


# In[10]:


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


# In[11]:


values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
print(values)
scaled = scaler.fit_transform(values)
#scaled = values


# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[4,5,6]], axis=1, inplace=True)
# Inputs var1 = t_db, var2 = RH, var3 = Rad, var4 = load
print(reframed.head(10))


# In[12]:


values = reframed.values
n_train_hours = 7000
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
print(test)
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# In[13]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.SimpleRNN(3, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(tf.keras.layers.Dense(1))
opt = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mae', optimizer=opt)
model.summary()


# In[14]:


EPOCHS = 100
VERBOSE = 1

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(train_X, train_y, epochs=EPOCHS, 
        validation_data=(test_X, test_y), verbose=VERBOSE, 
        shuffle=False, callbacks=[early_stop], batch_size=72)

pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()


# In[15]:


# make a prediction
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


# In[16]:


# invert scaling for forecast
inv_yhat = concatenate((test_X[:,[0,1,2]], yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,-1]


# In[17]:


# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_X[:,[0,1,2]], test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,-1]


# In[18]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
mae = mean_absolute_error(inv_y, inv_yhat)
print('Test MAE: %.3f' % mae)


# In[19]:


time = [i for i in range(len(inv_y))]
pyplot.clf()
pyplot.title("Test Set")
pyplot.xlabel('Timestep')
pyplot.ylabel('Predictions/Actual')
pyplot.rcParams['figure.figsize'] = [30, 10]
pyplot.plot(time, inv_y, label='Actual Load',linewidth=1)
pyplot.plot(time,inv_yhat, label='Predicted Load',linestyle='dashed',linewidth=1)
pyplot.legend()
pyplot.savefig('predicted.png')
pyplot.show()


# In[20]:


pyplot.clf()
pyplot.xlabel('Predicted')
pyplot.ylabel('Actual')
pyplot.scatter(inv_yhat, inv_y, s=1)
#_ = pyplot.plot([-4000,4000], [-4000,4000])
pyplot.show()


# In[ ]:





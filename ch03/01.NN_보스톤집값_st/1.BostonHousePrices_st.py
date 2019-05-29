#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

df = pd.read_csv("./housing.csv", delim_whitespace=True, header=None)
'''
print(df.info())    # 506개 dataframe, 14 col
print(df.head())
'''
dataset = df.values
X = dataset[:,0:13]
Y = dataset[:,13]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)

#%%
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1))    # dafault :  activation='linear'

model.compile(loss='mean_squared_error',
              optimizer='adam', metrics=['mae'])

num_epochs = 200
all_mae_histories = []
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=16)

mae_history = history.history['mean_absolute_error']
all_mae_histories.append(mae_history)

test_mse_score, test_mae_score = model.evaluate(X_test, Y_test)
print(test_mse_score, test_mae_score)

#%%
import numpy as np
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
print(average_mae_history)

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Test MAE')
plt.show()


# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:18:35 2019

@author: shkim
"""

from keras import backend as K
# 메모리 해제
K.clear_session()


from keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) =  boston_housing.load_data()

# (404, 13) (404,) (102, 13) (102,)
print(train_data.shape, train_targets.shape, test_data.shape, test_targets.shape)
print(train_targets[:5])

"""
Features
1. Per capita crime rate.
2. Proportion of residential land zoned for lots over 25,000 square feet.
3. Proportion of non-retail business acres per town.
4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
5. Nitric oxides concentration (parts per 10 million).
6. Average number of rooms per dwelling.
7. Proportion of owner-occupied units built prior to 1940.
8. Weighted distances to five Boston employment centres.
9. Index of accessibility to radial highways.
10. Full-value property-tax rate per $10,000.
11. Pupil-teacher ratio by town.
12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
13. % lower status of the population.

타깃은 주택의 중간 가격으로 천달러 단위입니다
"""

#%%
# 데이터 준비
#print(train_data[0])
mean = train_data.mean(axis=0)
train_data -= mean
#print(train_data[0])
std = train_data.std(axis=0)
train_data /= std
#print(train_data[0])

test_data -= mean
test_data /= std

#%%
# 모델 구성
from keras import models
from keras import layers

def build_model():
    # 동일한 모델을 여러 번 생성할 것이므로 함수를 만들어 사용합니다
    model = models.Sequential()
    model.add(layers.Dense(30, activation='relu',
                           input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # MAE(Mean Absolute Error)
#    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae', 'acc'])
    return model


# 새롭게 컴파인된 모델을 얻습니다
model = build_model()

# 전체 데이터로 훈련시킵니다
num_epochs = 80
all_mae_histories = []
history = model.fit(train_data, train_targets,
          epochs=num_epochs, batch_size=16, verbose=1)

mae_history = history.history['mean_absolute_error']
all_mae_histories.append(mae_history)

test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
#print(test_mse_score, test_mae_score)

#%%
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
#print(average_mae_history)

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Test MAE')
plt.show()
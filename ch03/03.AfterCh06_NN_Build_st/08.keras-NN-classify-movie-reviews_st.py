# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:13:09 2019

@author: shkim
"""

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# (25000,), (25000,), (25000,), (25000,)
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

#%%
print(len(train_data[0]))  # 218
print(len(train_data[1]))  # 189
print(len(train_data[15000]))  # 281
print('-'*50)
print(train_data[0])
print('-'*50)
print(train_labels[0])
print('-'*50)
print( max([max(seq) for seq in train_data]) )

#%%
word_index = imdb.get_word_index()
print(type(word_index))  # dict
tmp = [item for item in word_index.items()]
print(tmp[:10])

#%%
reverse_word_index = dict([(val, key) for (key, val) in word_index.items()])
# 0, 1, 2 : '패딩', '문서 시작', '사전에 없음' 을 위한 index
decoded_review = ''.join([reverse_word_index.get(i-3, '?') for i in train_data[0]]) 
print(decoded_review)

#%%
import numpy as np
seq = [1, 3, 7, 8]
rst = np.zeros(10)
print(rst)
rst[seq] = 1
print(rst)

#%%
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    # 크기가 (len(sequences), dimension))이고 모든 원소가 0인 행렬을 만듭니다
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]에서 특정 인덱스의 위치를 1로 만듭니다
    return results

# 훈련 데이터를 벡터로 변환합니다
x_train = vectorize_sequences(train_data)
print(x_train.shape)
print(x_train[0])
print('-'*50)
# 테스트 데이터를 벡터로 변환합니다
x_test = vectorize_sequences(test_data)
print(x_test.shape)
print(x_test[0])

#%%
# 레이블을 벡터로 바꿉니다
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
print(y_train[:5], y_test[:5])

#%%
# 신경망 모델 만들기
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%%
# 훈련 검증 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

#%%
history_dict = history.history
history_dict.keys()  # dict_keys(['val_loss', 'val_binary_accuracy', 'loss', 'binary_accuracy'])
print(len(history_dict['val_loss']), len(history_dict['val_binary_accuracy']))  # 20 20
print((history_dict['val_loss'])[-1], (history_dict['val_binary_accuracy'])[-1])
print((history_dict['loss'])[-1], (history_dict['binary_accuracy'])[-1])

#%%
import matplotlib.pyplot as plt

plt.clf()   # 그래프를 초기화합니다
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

# ‘bo’는 파란색 점을 의미합니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# ‘b’는 파란색 실선을 의미합니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#%%
plt.clf()   # 그래프를 초기화합니다
history_dict = history.history
acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

#%%
# Loss, Acc 위 그래프를 보면 과적합 현상이 나타남  --> epoch을 10로 줄여 학습시킴
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=512)

results = model.evaluate(x_test, y_test)
print(results)  # loss, acc

results = model.predict(x_test)
print(results)  # probabilities

#%%
# Training Loss와 Accuracy를 그래프로 그려보세요~

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:48:33 2019

@author: shkim
"""

import keras
print(keras.__version__)

#%%
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

#%%
print(train_images.shape, train_labels.shape)  # (60000, 28, 28) (60000,)
print(len(train_labels))
print(train_labels[:10])

#%%
print(test_images.shape, test_labels.shape)  # (10000, 28, 28) (10000,)
print(len(test_labels))
print(test_labels[:10])

#%%
import matplotlib.pyplot as plt

digit = train_images[4]

#plt.imshow(digit, cmap=plt.cm.binary)
plt.imshow(digit, cmap='Greys')
plt.show()

#%%
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

#%%
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#%%
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#%%
from keras.utils import to_categorical

print(train_labels[4])
train_labels = to_categorical(train_labels)
print(train_labels[4])
test_labels = to_categorical(test_labels)

#%%
history = network.fit(train_images, train_labels, epochs=5, batch_size=128)
#network.train_on_batch(x_batch, y_batch)

#%%
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)



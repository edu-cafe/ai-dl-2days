# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 17:42:58 2019

@author: shkim
"""
# pip install pillow  &  kernel restart
# mkdir tmp
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#%%
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')

#%%
img = load_img('./car_image.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
print(x.shape)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
print(x.shape)

#%%
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='tmp', save_prefix='car', save_format='jpeg'):
    i += 1
    if i > 20:
        break  # otherwise the generator would loop indefinitely

#%%
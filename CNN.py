# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 23:42:53 2018

@author: Atta
"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#Initialising the ANN

classifier = Sequential()

#step1-Adding Convolution Layer

classifier.add(Dense(32, 3 ,3, input_shape=(64, 64,3), activation='relu'))# 32 feature detectors of 3x3 dimesions

#step2-Pooling
classifier.add(Dense(pool_size=(2,2)))

#step3-Flattening
classifier.add(Flatten())

#Step4-Full Connection
classifier.add(Dense(output_dim= 128, activation='relu'))
classifier.add(Dense(output_dim= 1, activation='sigmoid'))

#Completng the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

#part 2- Fitting the ANN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                            'dataset/training_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                                        'dataset/test_set',
                                                        target_size=(64, 64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2000,
        epochs=25,
        validation_data=test_set,
        validation_steps=1000)




import numpy as np
from keras.preprocessing import image 
test_image = image.load_img('dataset/single_prediction/male_or_female1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    predicition = 'male'
else:
    predicition = 'female'






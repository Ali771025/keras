# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 11:09:05 2021

@author: Reza
"""
import numpy as np 
import matplotlib.pyplot as plt 
from keras import backend as K
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import EarlyStopping

(X_train, L_train), (X_test, L_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1], X_test.shape[2])
    input_shape = (1, X_train.shape[1], X_train.shape[2])
else:
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    
L_train = to_categorical(L_train, num_classes=10)
L_test = to_categorical(L_test, num_classes=10)
X_train = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test = (X_test - X_test.min())/(X_test.max() - X_test.min())

Model = Sequential()
Model.add(Conv2D(32, (3,3), input_shape =input_shape, activation = 'relu' ))
Model.add(MaxPooling2D((2,2)))
Model.add(Conv2D(64, (3,3), activation = 'relu' ))
Model.add(MaxPooling2D((2,2)))
Model.add(Flatten())
Model.add(Dropout(0.5))
Model.add(Dense(10, activation = 'softmax'))

Model.summary()

es = EarlyStopping(monitor= 'val_loss', mode = 'auto', verbose=1)

Model.compile(Adam(),
              loss = 'categorical_crossentropy',
              metrics= ['accuracy'])

h = Model.fit(X_train, L_train, epochs = 20, batch_size = 64,
              validation_split=0.1, callbacks= [es])

print(h.history.keys())

plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('The loss of training model')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])
plt.title('The accuracy of training model')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['train','val'])
plt.show()

Pre = Model.predict(X_test, batch_size=64)
Output_model = Model.predict_classes(X_test, batch_size=64)
Score = Model.evaluate(X_test, L_test, batch_size= 64)
print("Test Loss: ", Score[0])
print("Test accuracy: ", Score[1])







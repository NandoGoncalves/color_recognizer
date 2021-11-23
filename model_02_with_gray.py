import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers

import matplotlib.pyplot as plt
import os

width = 128 
height = 128
epochs = 15
classes_count = 17

df = pd.read_csv ("./images/color_range_with_gray.csv", sep=',') 
#print(df['color'].head())


# One-Hot-Encoding
df = pd.get_dummies(df, columns=['color'])
#print(df.head())

X = df[['red', 'green', 'blue']] 
y = df.drop(['red', 'green','blue'], axis=1)
#print(X)
print(y.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=189)

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(y_train)

model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.003), input_shape=[3]), #inputshape=[3] # 
    keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.003), activation='relu'), # 
    keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.003), activation='relu'), #
    keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.003), activation='relu'), # 
    keras.layers.Dense(64, kernel_regularizer=regularizers.l2(0.003), activation='relu'), #     
    keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.003), activation='relu'), #     
    keras.layers.Dense(24, kernel_regularizer=regularizers.l2(0.003), activation='relu'), # 
    keras.layers.Dense(32, kernel_regularizer=regularizers.l2(0.003), activation='relu'), #  


    keras.layers.Dense(classes_count)
  ])



optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

model.compile(loss=loss_function,
                optimizer=optimizer,
                metrics=['acc'])

model.summary()

history = model.fit(x=X, y=y, 
                    validation_split=0.15, 
                    epochs=epochs, 
                    batch_size=6, 
                    #verbose=0,
                    shuffle=True)


acc_train = history.history['acc']
acc_val = history.history['val_acc']
epochs = range(1,epochs+1)
plt.plot(epochs,acc_train, 'g', label='training accuracy')
plt.plot(epochs, acc_val, 'b', label= 'validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('./models/model_02.h5')


from tensorflow import keras
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

width = 128 
height = 128
epochs = 5

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'    
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    "./images/train", target_size=(height,height), batch_size=32, class_mode='binary',shuffle=True)


validation_generator = test_datagen.flow_from_directory(
    "./images/validation", target_size=(height,height), batch_size=32,class_mode='binary',shuffle=True)


model = keras.Sequential([
    keras.layers.Conv2D(128,(3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPool2D((2,2)),
    #keras.layers.Conv2D(512,(3,3),activation='relu'),
    #keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(256,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    keras.layers.Conv2D(128,(3,3), activation='relu'),
    keras.layers.MaxPool2D(2,2),
    #keras.layers.Conv2D(1024,(3,3), activation=keras.layers.LeakyReLU(alpha=0.01)),    
    #keras.layers.MaxPool2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(16, activation='softmax')])

model.summary()

model.compile(optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['acc'])

history = model.fit_generator(train_generator, 
    #steps_per_epoch = 20, 
    epochs=epochs, 
    validation_data=validation_generator #, validation_steps=40
    )

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

model.save('./models/model_01.h5')
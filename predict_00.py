from tensorflow import keras
from keras_preprocessing import image
import os

model = keras.models.load_model('./models/model_00.h5')

model.summary()

y_labels = []
folder = './images/train'
y_labels = [name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))]
print('labels :')
print( y_labels)
print('-----')

predict_folder = './images/predict'
for subdir, dirs, files in os.walk(predict_folder):
    for image_name in files:

        img = image.load_img(os.path.join(predict_folder,image_name), target_size=(128,128))
        x = image.img_to_array(img)
        x = x.reshape((1,)+x.shape)

        y_prob = model.predict(x) 
        y_classes = y_prob.argmax(axis=-1)

        print(image_name, ':', y_labels[int(y_classes)])
        #print(y_classes)
        #print(y_labels[int(y_classes)])



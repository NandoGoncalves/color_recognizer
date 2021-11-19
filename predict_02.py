import pandas as pd
from tensorflow import keras
from keras_preprocessing import image
import os



model = keras.models.load_model('./models/model_02.h5')

model.summary()



y_labels = ['Blue', 'Blue-Magenta', 'Cyan', 'Cyan-Blue',
       'Green', 'Green-Cyan', 'Magenta',
       'Magenta-Pink', 'Orange-Brown', 'Orange-Yellow',
       'Pink', 'Pink-Red', 'Red', 'Red-Orange',
       'Yellow', 'Yellow-Green']

ratios = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

df_color = pd.DataFrame(list(zip(y_labels, ratios)), columns=['color','ratio'])

predict_folder = './images/predict'
for subdir, dirs, files in os.walk(predict_folder):
    for image_name in files:

        df_color['ratio'] = 0

        img = image.load_img(os.path.join(predict_folder,image_name), target_size=(128,128))


        pixels_sequence = img.getdata()
        pixels = list(pixels_sequence)

        y_prob = model.predict(pixels)

        for i in range( 0, len(y_labels)):

            df_color.at[i,'ratio'] += y_prob[0][i]        

        
        df_color_sorted = df_color.sort_values(by='ratio', ascending=False)


        color_labels = []
        for index, row in df_color_sorted.iterrows(): # df_color_sorted.head(3).iterrows()
            color_labels.append(row['color'])

        #print(image_name[0:10], '... : ', color_labels[0], ' - ', color_labels[1], ' - ', color_labels[2])
        print(image_name[0:10], '... : ', color_labels)




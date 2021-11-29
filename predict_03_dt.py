import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from PIL import Image
from time import sleep
import pickle
import random

import os

features_model_count = 378
filename = './models/model_03.sav'
clf = pickle.load(open(filename, 'rb'))

predict_folder = './images/predict'
for subdir, dirs, files in os.walk(predict_folder):
    for image_name in files:

        img = Image.open(os.path.join(predict_folder,image_name))

        imgArray = np.array(img)

        df = pd.DataFrame(columns=['red', 'green', 'blue'])

        len_imgArray = len(imgArray)

        for index_imgArray in range(0, len_imgArray):

            imgArray_unique = np.unique(imgArray[index_imgArray], axis=0, return_counts = False)

            r = imgArray_unique[0][0]
            g = imgArray_unique[0][1]
            b = imgArray_unique[0][2]

            df.at[index_imgArray, 'red'] = r
            df.at[index_imgArray,'green'] = g
            df.at[index_imgArray,'blue'] = b
            
            #print('r', r, 'g', g, 'b', b, index_imgArray, '/', len_imgArray)
            #print('.', end="")


        pred = clf.predict(df)

        print(image_name[0:15], '... : ', int(len(pred[pred=='Blue']) * 100 / len(pred)), '%')


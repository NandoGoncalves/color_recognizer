import numpy as np
import pandas as pd
from PIL import Image
import skimage
from skimage import io
import pathlib
import os
import shutil
import csv

import sys

import random

def transform(img_path):

    img = skimage.io.imread(img_path)/255 #.0
    img = skimage.util.random_noise(img, mode='gaussian', mean=random.random(), var=random.random())

    io.imsave(img_path, img) 
    

width = 128 
height = 128
images_per_classes = 1000

color_range_csv = 'color_range.csv' 

if len(sys.argv) > 1:
    color_range_csv = 'color_range_with_gray.csv'

f_csv = open(os.path.join('images', 'images_color_range.csv'), 'w', newline='')
writer_csv = csv.writer(f_csv)
writer_csv.writerow(['red','green','blue','color'])



df = pd.read_csv (os.path.join('dataset', color_range_csv), sep=';', usecols= ['color', 'color_name','red', 'green', 'blue', 'lig']) 
print(df.head())

i = 0

shutil.rmtree(os.path.join('images', 'train'), ignore_errors=True)
shutil.rmtree(os.path.join('images', 'validation'), ignore_errors=True)
shutil.rmtree(os.path.join('images', 'test'), ignore_errors=True)

while(True):

    for f in ('train','validation','test'):


        for index, row in df.iterrows():

            color = row['color']
            color_name = row['color_name']
            red = row['red']
            green = row['green']
            blue = row['blue']
            lig = row['lig']

            #if lig > 50 or lig < 40:
            #    continue

            path = os.path.join('images', f, color)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            if sum([len(files) for r, d, files in os.walk(path)]) < images_per_classes:
                img = Image.new('RGB', (width, height), color = (red, green, blue))

                
                
                image_name = color_name + '_' + str(i) +'.png'
                image_full_path = os.path.join( path, image_name)

                
                
                img.save(image_full_path)

                if i%3 == 0:
                    transform(image_full_path)

                writer_csv.writerow([red,green,blue,color])
                print(image_full_path)
                i = i+1
    
    if sum([len(files) for r, d, files in os.walk('./images')]) > (images_per_classes-100) * 16 * 3:
        break

 

f_csv.close()

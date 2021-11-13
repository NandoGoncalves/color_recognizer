import pandas as pd
from PIL import Image
import pathlib
import os

width = 64 
height = 64

df = pd.read_csv ("./dataset/color_range.csv", sep=';', usecols= ['color', 'color_name','red', 'green', 'blue']) 
print(df.head())

i = 0

for f in ('train','validation','test'):

    for index, row in df.iterrows():

        color = row['color']
        color_name = row['color_name']
        red = row['red']
        green = row['green']
        blue = row['blue']

        path = os.path.join('images', f, color)
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

        img = Image.new('RGB', (width, height), color = (red, green, blue))
        image_name = color_name + '_' + str(i) +'.png'
        image_full_path = os.path.join( path, image_name) 
        img.save(image_full_path)
        print(image_full_path)
        i = i+1

 



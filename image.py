import pandas as pd
from PIL import Image
import pathlib
import os

width = 128 
height = 128
images_per_classes = 600

df = pd.read_csv ("./dataset/color_range.csv", sep=';', usecols= ['color', 'color_name','red', 'green', 'blue', 'lig']) 
print(df.head())

i = 0

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
                print(image_full_path)
                i = i+1
    
    if sum([len(files) for r, d, files in os.walk('./images')]) > (images_per_classes-100) * 16 * 3:
        break

 



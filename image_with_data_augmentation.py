import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import skimage
from skimage import io
import pathlib
import os
import shutil
import csv

import sys

import random
from time import sleep

def random_color():
    return random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)

def transform(img_path, toggle=0, color=(255,0,0)):

    img = skimage.io.imread(img_path)/255 #.0

    if toggle%2 == 0:
        img = skimage.util.random_noise(img, mode='gaussian') # , mean=random.random(), var=random.random()
    else:
        img = skimage.util.random_noise(img, mode='salt')


    io.imsave(img_path, img)

    border(img_path, color) # 
    ellipse(img_path, color) # 
    rectangle(img_path, color) # 

def border(img_path, color=random_color()): # (255,0,0)
    img = Image.open(img_path)
    border_img = Image.new('RGB', (img.width, img.height), random_color())
    border_img.paste(img, (int(img.width*random.random()/int((random.random()+1)*10)), int(img.height*random.random()/int((random.random()+1)*10))))
    img = border_img.rotate(random.randint(0, 360), expand=1, fillcolor = color)
    border_img.save(img_path)    

def ellipse(img_path, color=random_color()): #   (255,0,0)
    img = Image.open(img_path)
    
    draw = ImageDraw.Draw(img)
    draw.ellipse((int(10*random.random()), int(30*random.random()), int(75*random.random()), int(100*random.random())), outline=random_color(), width=int(5*random.random()),fill=color)
    draw.ellipse((int(25*random.random()), int(75*random.random()), int(90*random.random()), int(110*random.random())), outline=random_color(), width=int(5*random.random()),fill=color)
    img = img.rotate(random.randint(0, 360), expand=1, fillcolor = color)
    img.save(img_path)

def rectangle(img_path, color=random_color()): # (255,0,0) 
    img = Image.open(img_path)
    
    draw = ImageDraw.Draw(img)
    draw.rectangle((int(60*random.random()), int(70*random.random()), int(30*random.random()), int(120*random.random())), outline=random_color(), width=int(10*random.random()),fill=color)
    draw.rectangle((int(80*random.random()), int(90*random.random()), int(75*random.random()), int(100*random.random())), outline=random_color(), width=int(10*random.random()),fill=color)
    img = img.rotate(random.randint(0, 360), expand=1, fillcolor = color)
    img.save(img_path)

def files_count(folder):
    return len(os.listdir(dir))



width = 128 
height = 128
images_per_classes = 1000
classes_count = 2

color_range_csv = 'color_range.csv' 

if len(sys.argv) > 1:
    color_range_csv = 'color_range_with_gray.csv'

f_csv = open(os.path.join('images', color_range_csv), 'w', newline='')
writer_csv = csv.writer(f_csv)
writer_csv.writerow(['red','green','blue','color'])



df = pd.read_csv (os.path.join('dataset', color_range_csv), sep=';', usecols= ['color', 'color_name','red', 'green', 'blue', 'lig']) 

print(color_range_csv)
print(df.color.unique())
print(sleep(5))

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

            if lig > 86 or lig < 24:
                continue

            path = os.path.join('images', f, color)
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)

            if sum([len(files) for r, d, files in os.walk(path)]) < images_per_classes:
                img = Image.new('RGB', (width, height), color = (red, green, blue))

                
                
                image_name = color_name + '_' + str(i) +'.png'
                image_full_path = os.path.join( path, image_name)

                
                
                img.save(image_full_path)

                if i%7 != 0:
                    transform(image_full_path, i, color = (red, green, blue))

                writer_csv.writerow([red,green,blue,color])
                print(i, image_full_path)
                i = i+1
    
    if sum([len(files) for r, d, files in os.walk('./images')]) > (images_per_classes-10) * classes_count * 3:
        break

 

f_csv.close()

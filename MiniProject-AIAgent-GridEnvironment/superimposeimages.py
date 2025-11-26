import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import image as mp_image


# Required magic to display matplotlib plots in notebooks


from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import shutil
     



import os
import random
from PIL import Image
from PIL import ImageOps 

def superimpose_random_bg(foreground_rgba_path, bg_folder):
    """
    foreground_rgba_path : str  – your agent-produced 128×128 RGBA frame
    bg_folder           : str  – folder with 128×128 JPG/PNG background images
    returns             : PIL.Image – 128×128 RGB
    """
    # 1. pick a random background
    bg_files = [f for f in os.listdir(bg_folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not bg_files:
        raise FileNotFoundError("No images found in background folder")
    bg_path = os.path.join(bg_folder, random.choice(bg_files))
    bg = Image.open(bg_path).convert('RGB')   # RGB 128×128

    # 2. load foreground
    fg = Image.open(foreground_rgba_path).convert('RGBA')  # keep alpha

    # 3. composite (alpha is the mask)
    new_frame = Image.alpha_composite(bg.convert('RGBA'), fg).convert('RGB')

    return new_frame

def resize_image(src_image, size=(128,128)): 
    
    
    # resize the image so the longest dimension matches our target size
    return src_image.thumbnail(size, Image.LANCZOS)


def main():
    data_dir = 'data/bg' 


    device = torch.device("cpu")

    train_loader = torch.utils.data.DataLoader(data_dir, batch_size=4, shuffle=True)

    classes = sorted(os.listdir(data_dir))
    print(classes)
    training_folder_name = 'data/assets/imagen1'

    # New location for the resized images
    train_folder = 'data/bg'


    # PLEASE SEE > MOVE FLOOR, LAVA AND WALL INTO DATA/NOIMPOSE, THEN RUN THIS KERNEL: SUPERIMPOSE, THEN RUN RESIZE ON FLOOR, LAVA AND WALL
    noimpose = 'data/noimpose'
    bg_folder = 'data/noimpose/floor'


    # Create resized copies of all of the source images
    size = (58,58)

    # Create the output folder if it doesn't already exist


    # Loop through each subfolder in the input folder
    print('Transforming images...')
    for root, folders, files in os.walk(training_folder_name):
        for sub_folder in folders:
            print('processing folder ' + sub_folder)
            # Create a matching subfolder in the output dir
            saveFolder = os.path.join(train_folder,sub_folder)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            # Loop through the files in the subfolder
            file_names = os.listdir(os.path.join(root,sub_folder))
            for file_name in file_names:
                # Open the file
                file_path = os.path.join(root,sub_folder, file_name)
            
                # Create a bg version and save it
                new_image = superimpose_random_bg(file_path, bg_folder)
                resize_image(new_image, (58,58))
                saveAs = os.path.join(saveFolder, file_name)
                #print("writing " + saveAs)
                new_image.save(saveAs)

    for root, folders, files in os.walk(noimpose):
        for sub_folder in folders:
            print('processing folder ' + sub_folder)
            # Create a matching subfolder in the output dir
            saveFolder = os.path.join(train_folder,sub_folder)
            if not os.path.exists(saveFolder):
                os.makedirs(saveFolder)
            # Loop through the files in the subfolder
            file_names = os.listdir(os.path.join(root,sub_folder))
            for file_name in file_names:
                # Open the file
                file_path = os.path.join(root,sub_folder, file_name)
            
                # Create a bg version and save it
                '''new_image = superimpose_random_bg(file_path, bg_folder)'''
                new_image = Image.open(file_path)
                resize_image(new_image, (58,58))
                saveAs = os.path.join(saveFolder, file_name)
                #print("writing " + saveAs)
                new_image.save(saveAs)

    print('Done.')

main()
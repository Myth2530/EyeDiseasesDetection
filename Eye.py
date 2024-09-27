import glob
import os
import time
import shutil
import pathlib
import itertools
from idlelib import testing

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import shutil
from tensorflow.python.keras.utils.version_utils import training

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers


print ('modules loaded')
training_directory='C:/Users/jacob/PycharmProjects/pythonProject/dataset/training'
testing_directory='C:/Users/jacob/PycharmProjects/pythonProject/dataset/testing'

training_folders=[var for var in os.listdir(training_directory) if os.path.isdir(os.path.join(training_directory,var))]
print(training_folders)
image_extensions=['.jpg','.jpeg','.png']
for folder in training_folders:
    image_files=[
        var for var in os.listdir(os.path.join(training_directory,folder))
        if os.path.isfile(os.path.join(os.path.join(training_directory,folder),var))
        and any(var.lower().endswith(ext) for ext in image_extensions)
    ]
    print(image_files)

testing_folder=[var for var in os.listdir(testing_directory) if os.path.isdir(os.path.join(testing_directory,var))]
print(testing_folder)
image_extensions=['.jpg','.jpeg','.png']
for folder in testing_folder:
    image_files=[
        var for var in os.listdir(os.path.join(testing_directory,folder))
        if os.path.isfile(os.path.join(os.path.join(testing_directory,folder),var))
        and any(var.lower().endswith(ext) for ext in image_extensions)
    ]
    print(image_files)

print("section 2")

print(training_folders[0])

cataract_training_directory=os.path.join(training_directory,training_folders[0])
cataract_training_directory=os.path.normpath(cataract_training_directory)
print(cataract_training_directory)

cataract_testing_directory=os.path.join(testing_directory,testing_folder[0])
cataract_testing_directory=os.path.normpath(cataract_testing_directory)
print(cataract_testing_directory)

current_directory=os.getcwd()
print(current_directory)

temp_directory=os.path.join(current_directory,'temp')
print(temp_directory)

if not os.path.exists(temp_directory):
    print('temp directory does not exist, creating it')
    os.makedirs(temp_directory)
    print('temp directory created')
else:
    print('temp directory already exists')

for filename in os.listdir(cataract_training_directory):
    source=os.path.join(cataract_training_directory,filename)
    destination=os.path.join(temp_directory,filename)
    if os.path.isfile(source):
        shutil.copy(source,destination)
print("All files copied from cataract training to temp directory")

for filename in os.listdir(cataract_testing_directory):
    source=os.path.join(cataract_testing_directory,filename)
    destination=os.path.join(temp_directory,filename)
    if os.path.isfile(source):
        shutil.copy(source,destination)
print("All files copied from cataract testing to temp directory")


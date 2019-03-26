"""
For ImageDataGenerator(keras), we should divide images into each class folder.
For exmaple, a image belonging to classA should be placed in classA folder.
data/org/train/train_xxxxx.png -> data/gen/train/classA/train_xxxxx.png
Also, we need to divide Images into train and validation.
In this code, we decide the number of train samples with train_ratio(variable)
"""

import os 
import numpy as np
import pandas as pd
from skimage import io
from tqdm import tqdm
import shutil

input_path='../../data/org/train/'
label_path='../../data/org/train_master.tsv'
train_path='../../data/gen/train/'
validation_path='../../data/gen/validation/'


"""
Generate class folders
Attention:
When you name folders by figure, for exmaple, when generating 0, 2, 10 folder
> mkdir 0 2 10
note that sorted folder list is [0,10,2].('10' locates ahead of '2')
This cause some problems when connecting label and folder name in ImageDataGenerator.
Avoiding it, we need to name 0 00, 2 02 , respectively. 
In Python, using ''{:0=2}'.format(i)'
as below.
"""
for i in range(20):
    os.makedirs(os.path.join(train_path,'{:0=2}'.format(i)),exist_ok=True)
    os.makedirs(os.path.join(validation_path,'{:0=2}'.format(i)),exist_ok=True)

"""
Move images
"""
train_ratio=0.9
master=pd.read_csv(label_path,sep='\t')
labels=np.array(master['label_id'])
for i in tqdm(range(len(master))):
    if i<len(master)*train_ratio:
        shutil.move(os.path.join(input_path,master['file_name'][i]),os.path.join(train_path,'{:0=2}'.format(labels[i])))
    else:
        shutil.move(os.path.join(input_path,master['file_name'][i]),os.path.join(validation_path,'{:0=2}'.format(labels[i])))



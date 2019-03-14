"""
Print the number of each class samples  in train floder and validation folder.
"""

import os
from glob import glob

train_path='../../data/gen/train/'
validation_path='../../data/gen/validation/'

print('for train')
for i in range(20):
    print('class {}:{}'.format(i,len(glob(os.path.join(train_path+str(i),'*.png')))))

print('for validation')
for i in range(20):
    print('class {}:{}'.format(i,len(glob(os.path.join(validation_path+str(i),'*.png')))))

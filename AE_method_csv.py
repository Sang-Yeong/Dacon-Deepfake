import pandas as pd
import numpy as np
from natsort import natsorted
import glob
import os

image_path = glob.glob('C:/Users/mmclab1/Desktop/deepfake_compt/confirm_attack2img/AE-real_fake/train/**/*.*', recursive=True)
image_path = natsorted(image_path)

image_name = []
face_name = []
method = []


for i in image_path:
    image_name.append(os.path.split(i)[-1])

for i in image_name:
    face_name.append(i.split('_')[1])
    method.append(i.split('_')[3])


df1 = pd.DataFrame(image_name, columns=['image_name'])
df1['face_name'] = face_name
df1['target'] = method


# save cvs file
df1.to_csv('C:/Users/mmclab1/Desktop/deepfake_compt/confirm_attack2img/AE-real_fake/train.csv')
print('success')
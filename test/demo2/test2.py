import os

import numpy as np
from PIL import Image

from tensorflow.python.keras.api.keras import layers,models

model = models.load_model('dataDemo/cats_and_dogs_small_1.h5',compile=False)
picPath = "H:\\Machine Learning\\CNN\\CNNProjects\\data\\train3\\test\\cats"
paths=[]
names=[]
err_paths=[]
err_name=[]

for i in os.listdir(picPath):
    path = os.path.join(picPath,i)
    name = i.split('.')[0]
    names.append(name)
    paths.append(path)

for j in range(0,13):
    img = Image.open(paths[j])
    img = img.resize((150,150))
    img_arr = np.array(img)/255.
    img_arr = img_arr.reshape(1,150,150,3)
    pre = model.predict(img_arr)
    # if pre[0][0]>pre[0][1]:
    #     print('cat')
    # else:
    #     print('dog')
    print(pre)
from tensorflow.python.keras.api.keras.models import load_model
import os, random
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

def read_image():
    # path='H:\\Machine Learning\\CNN\\CNNProjects\\data\\'
    path='H:\\Machine Learning\\CNN\\CNNProjects\\data\\test\\248.jpg'
    # file_path = path + random.choice(os.listdir(path))
    # pil = Image.open(file_path,'r')
    pil = Image.open(path,'r')
    return pil

def preditCatAndDog(pil,model):
    t = 150
    w = t
    h = t
    #缩放图片
    pil = pil.resize((w,h))
    #转array格式
    array = np.array(pil) /255.
    #预测
    array = array.reshape(1,w,h,3)
    res = model.predict(array)
    print(res)
    if res[0][0]>0.5:
        print('dog')
    else:
        print('cat')

#载入模型
path='data\cats_and_dogs_small_2.h5'
model = load_model(path)

pil = read_image();
imshow(np.asarray(pil))
preditCatAndDog(pil,model)

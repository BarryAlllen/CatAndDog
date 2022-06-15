from tensorflow.python.keras.api.keras.models import load_model
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

def read_image():
    path='H:\\Machine Learning\\CNN\\CNNProjects\\data\\test\\111.jpg'
    # path='./resources/cat.jpg'
    pil = Image.open(path,'r')
    return pil

def preditCatAndDog(pil,model):
    t = 200
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
path='data\catDogFight02-4.h5'
model = load_model(path)

pil = read_image();
# imshow(np.asarray(pil))
preditCatAndDog(pil,model)

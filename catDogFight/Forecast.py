import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras.models import load_model
import numpy as np
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 去掉加载GPU的警告

def read_image():
    path='H:\\Machine Learning\\CNN\\CNNProjects\\data\\test\\1380.jpg'
    # path='E:\\Desktop\\dog2.jpg'
    # path='E:\\1\\微信图片_20190817133709.jpg'
    # path='./resources/cat.jpg'
    pil = Image.open(path,'r')
    return pil

def formatRes(r):
    r = round(r,2)
    r = format(r,'.2%')
    return r

def preditCatAndDog(pil,model):
    t = 200
    w = t
    h = t
    #缩放图片
    pil = pil.resize((w,h))
    #转array格式
    array = np.array(pil) /255.
    array = array.reshape(1,w,h,3)
    # 预测
    res = model.predict(array)
    # print(res)
    if res[0][0]<res[0][1]:
        r = res[0][1]
        r = formatRes(r)
        print('预测结果: 狗')
        print('概率为: ',r)
        return 'dog',r
    else:
        r = res[0][0]
        r = formatRes(r)
        print('预测结果: 猫')
        print('概率为: ',r)
        return 'cat',r

def getModel():
    path = 'data\catDogFight13-DenseNet121-f.h5'
    model = load_model(path)
    return model

# def predit_more():
#     plt.figure()
#     for i in range(1,9):
#         path = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\test\\'+str(i)+'.jpg'
#         img = Image.open(path, 'r')
#         plt.subplot(2, 4, i + 1)
#         plt.imshow(img)
#         model = getModel()
#         res = preditCatAndDog(img, model)
#         # plt.xticks([res])
#     plt.show()

def predit_one():
    # 载入模型
    model = getModel()
    pil = read_image()
    res,r = preditCatAndDog(pil, model)
    plt.imshow(np.asarray(pil))
    plt.xlabel(res+'  '+str(r))
    plt.show()

predit_one()
# predit_more()






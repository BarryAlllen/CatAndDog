import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras.models import load_model
import numpy as np
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 去掉加载GPU的警告

def read_image():
    path='H:\\Machine Learning\\CNN\\CNNProjects\\data\\1.jpg'
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
    path = 'data/catDogFight13-DenseNet121-f.h5'
    model = load_model(path)
    return model

def predit_more():
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\test'

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(200, 200),
        batch_size=50,
        shuffle=True,
        class_mode='categorical'
    )
    model = getModel()
    y_pred = model.predict(test_generator, batch_size=50, verbose=1)
    for i in range(1,11):
        if y_pred[i][0]>y_pred[i][1]:
            r = y_pred[i][0]
            r = formatRes(r)
            print("预测结果: 狗(dog)  概率为:",r)
        else:
            r = y_pred[i][1]
            r = formatRes(r)
            print("预测结果: 猫(cat)  概率为:",r)

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






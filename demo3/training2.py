# 网络模型构建
import os

import matplotlib.pyplot as plt

from tensorflow.python.keras.api.keras import layers, models
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api.keras import optimizers
from tensorflow.python.keras.api.keras.preprocessing import image

from tensorflow.python.keras.api.keras.layers import Conv2D
from tensorflow.python.keras.api.keras.layers import MaxPooling2D
from tensorflow.python.keras.api.keras.layers import Flatten
from tensorflow.python.keras.api.keras.layers import Dropout
from tensorflow.python.keras.api.keras.layers import Dense
from tensorflow.python.keras.api.keras.optimizers import SGD
from tensorflow.python.keras.api.keras.utils import plot_model


# 1.构建网络模型
def define_cnn_model():
    # keras的序贯模型
    model = models.Sequential()
    # 卷积层，卷积核是3*3，激活函数relu
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))
    # 卷积层，卷积核2*2，激活函数relu
    model.add(Conv2D(64, (3, 3), activation='relu'))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))
    # 卷积层，卷积核是3*3，激活函数relu
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))
    # 卷积层，卷积核是3*3，激活函数relu
    model.add(Conv2D(128, (3, 3), activation='relu'))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))
    # flatten层，用于将多维的输入一维化，用于卷积层和全连接层的过渡
    model.add(Flatten())
    # 退出层
    model.add(Dropout(0.5))
    # 全连接，激活函数relu
    model.add(Dense(128, activation='relu'))
    # 全连接，激活函数sigmoid
    model.add(Dense(1, activation='sigmoid'))

    # 输出模型各层的参数状况
    model.summary()

    # 编译模型
    opt = SGD(lr=0.001, momentum=0.9)

    # 2.配置优化器
    model.compile(loss='binary_crossentropy',
                  # optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    return model


# 生成神经网络结构图片
# plot_model(model,to_file='cnn_model.png',dpi=100,show_shapes=True,show_layer_names=True)

def train_cnn_model():
    # 生成模型
    model = define_cnn_model()
    # 创建图片生成器
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    teain_target = datagen.flow_from_directory('H:\\Machine Learning\\CNN\\CNNProjects\\data\\train3\\train',
                                               class_mode='binary',
                                               batch_size=64,
                                               target_size=(200, 200)
                                               )
    # 训练模型
    model.fit_generator(teain_target,
                        steps_per_epoch=(teain_target),
                        epochs=1,
                        verbose=1
                        )

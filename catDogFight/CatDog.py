import os
import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras import layers, models
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api.keras import optimizers

from tensorflow.python.keras.api.keras.layers import Conv2D
from tensorflow.python.keras.api.keras.layers import MaxPooling2D
from tensorflow.python.keras.api.keras.layers import Flatten
from tensorflow.python.keras.api.keras.layers import Dropout
from tensorflow.python.keras.api.keras.layers import Dense

from tensorflow.python.keras.api.keras.utils import plot_model

# 神经网络模型构建
def cnn_model():
    # 初始化序列模型
    model = models.Sequential()

    #卷积层 卷积核 3x3 输入为200x200的RGB图片
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(200, 200, 3)))
    #最大池化层
    model.add(MaxPooling2D((2, 2)))

    #卷积层 卷积核 3x3 输出维度64
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    #最大池化层
    model.add(MaxPooling2D((2, 2)))

    #卷积层 卷积核 3x3 输出维度128
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #最大池化层
    model.add(MaxPooling2D((2, 2)))

    #卷积层 卷积核 3x3 输出维度128
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    #最大池化层
    model.add(MaxPooling2D((2, 2)))

    # flatten层
    model.add(Flatten())

    # # 退出层
    model.add(Dropout(0.5))

    # 全连接层
    model.add(Dense(128, activation='relu'))
    # 全连接层 sigmoid
    model.add(Dense(1, activation='sigmoid'))

    # 输出各层参数
    model.summary()

    # 配置优化器
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['accuracy'])

    return model

# 生成神经网络结构图片
def get_plot_model():
    model = cnn_model()
    plot_model(model, to_file='resources/cnn_model.png', dpi=100, show_shapes=True, show_layer_names=True)

# 训练模型
def train_cnn_model():
    # 生成模型
    model = cnn_model()

    #测试数据图片格式化 增强数据
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, # 归一化
        rotation_range=40, # 旋转角度
        width_shift_range=0.2, # 水平偏移
        height_shift_range=0.2, # 垂直偏移
        shear_range=0.2, # 随机错切变换的角度
        zoom_range=0.2, # 随机缩放的范围
        horizontal_flip=True #随机将一半图像水平翻转
    )

    # 测试数据图片归一化
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 训练文件路径
    train_dir='H:\\Machine Learning\\CNN\\CNNProjects\\data\\train4\\train'
    # 验证文件路径
    validation_dir='H:\\Machine Learning\\CNN\\CNNProjects\\data\\train4\\validation'

    train_generator = train_datagen.flow_from_directory(
        # 训练文件路径
        train_dir,
        # 图像统一尺寸
        target_size=(200, 200),
        # batch数据大小 一次输入64张图片进行训练
        batch_size=32,
        # 返回标签数组形式 二进制
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        # 验证文件路径
        validation_dir,
        target_size=(200, 200),
        batch_size=32,
        class_mode='binary'
    )

    # 训练模型
    history = model.fit_generator(
        train_generator, # 定义的图片生成器
        steps_per_epoch=100,
        epochs=100, # 数据迭代的轮数
        validation_data=validation_generator,
        validation_steps=50
    )

    # 保存训练得到的的模型
    model.save('data\catDogFight02-4.h5')

    plt_result(history)

# 结果可视化
def plt_result(history):

    # 准确性
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # 损失值
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Train accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Train loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.show()

train_cnn_model()
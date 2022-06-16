import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.api.keras import layers, models
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras.layers import Conv2D
from tensorflow.python.keras.api.keras.layers import MaxPooling2D
from tensorflow.python.keras.api.keras.layers import Flatten
from tensorflow.python.keras.api.keras.layers import Dropout
from tensorflow.python.keras.api.keras.layers import Dense

from tensorflow.python.keras.api.keras.utils import plot_model

# 神经网络模型构建
from tensorflow.python.ops.confusion_matrix import confusion_matrix

wh = 200
batch_size = 32
tn = 0

def cnn_model():
    # 初始化序列模型
    model = models.Sequential()

    # # 官方搭建的VGG19网络
    # conv_base = keras.applications.VGG19(weights='imagenet', include_top=False)
    # # 设置为不可训练
    # conv_base.trainable = False

    # 卷积层 卷积核 3x3 输入为200x200的RGB图片
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(wh, wh, 3)))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))

    # 卷积层 卷积核 3x3 输出维度64
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))

    # 卷积层 卷积核 3x3 输出维度128
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # 最大池化层
    model.add(MaxPooling2D((2, 2)))

    # # 卷积层 卷积核 3x3 输出维度128
    # model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    # # 最大池化层
    # model.add(MaxPooling2D((2, 2)))

    # flatten层
    model.add(Flatten())

    # 全连接层
    model.add(Dense(512, activation='relu'))
    # # 退出层
    model.add(Dropout(0.5))

    # 全连接层 sigmoid
    # model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(2, activation='softmax'))

    # 输出各层参数
    model.summary()

    # 配置优化器
    # model.compile(loss='binary_crossentropy',
    #               optimizer=optimizers.RMSprop(lr=1e-4),
    #               metrics=['accuracy'])
    #动态学习率为指数衰减型
    lr_sch = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=50,
        decay_rate=0.96,
        staircase=True
    )

    gen_optimizer = keras.optimizers.Adam(learning_rate=lr_sch) # adam优化器
    # 编译模型
    model.compile(
        optimizer=gen_optimizer,
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


# 生成神经网络结构图片
def get_plot_model():
    model = cnn_model()
    plot_model(model, to_file='resources/cnn_model.png', dpi=100, show_shapes=True, show_layer_names=True)


# 训练模型
def train_cnn_model():
    # 生成模型
    model = cnn_model()

    # 测试数据图片格式化 增强数据
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,  # 归一化
        rotation_range=40,  # 旋转角度
        width_shift_range=0.2,  # 水平偏移
        height_shift_range=0.2,  # 垂直偏移
        shear_range=0.2,  # 随机错切变换的角度
        zoom_range=0.2,  # 随机缩放的范围
        horizontal_flip=True  # 随机将一半图像水平翻转
    )

    # 测试数据图片归一化
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # 训练文件路径
    train_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train4\\train'
    # 验证文件路径
    validation_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train4\\validation'

    train_generator = train_datagen.flow_from_directory(
        # 训练文件路径
        train_dir,
        # 图像统一尺寸
        target_size=(wh, wh),
        # batch数据大小 一次输入64张图片进行训练
        batch_size=batch_size,
        # 返回标签数组形式 二进制
        # class_mode='binary'
        class_mode='categorical'
    )

    tg = train_generator.n

    validation_generator = test_datagen.flow_from_directory(
        # 验证文件路径
        validation_dir,
        target_size=(wh, wh),
        batch_size=batch_size,
        # class_mode='binary'
        class_mode = 'categorical'
    )

    # 训练模型
    history = model.fit_generator(
        train_generator,  # 定义的图片生成器
        steps_per_epoch=100,
        epochs=5,  # 数据迭代的轮数
        validation_data=validation_generator,
        validation_steps=50
    )

    labels = ['cats', 'dogs']
    y_pred = model.predict(train_generator, tg // batch_size + 1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    confusion_mtx = confusion_matrix(y_true=train_generator.classes, y_pred=y_pred_classes)
    plot_confusion_matrix(confusion_mtx,normalize=True, target_names=labels)

    # 保存训练得到的的模型
    model.save('data\catDogFight08-3.h5')

    # labels = ['cat', 'dogs']
    # y_pred = model.predict(test_datagen)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    # confusion_mtx = confusion_matrix(y_true=test_datagen.classes, y_pred=y_pred_classes)
    # plot_confusion_matrix(confusion_mtx,normalize=True,target_names=labels)

    plt_result(history)


# 绘制混淆矩阵的
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=False):
    accuracy = np.trace(cm) / float(np.sum(cm))  # 计算准确率
    misclass = 1 - accuracy  # 计算错误率
    if cmap is None:
        plt.get_cmap('Blues')  # 颜色设置成蓝色
    plt.figure(figsize=(10, 8))  # 设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示图片
    plt.title(title)  # 显示标题
    plt.colorbar()  # 绘制颜色条

    # 设置x,y坐标标签
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)  # x坐标标签旋转45度
        plt.yticks(tick_marks, target_names)  # y坐标

    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)
        cm = np.round(cm, 2)  # 对数字保留两位小数

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):  # 将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize:  # 标准化
            plt.text(j, i, "{:0.2f}".format(cm[i, j]),  # 保留两位小数
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色
        else:  # 非标准化
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色

    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label')  # y方向上的标签
    plt.xlabel("Predicted label\naccuracy={:0.4f}\n misclass={:0.4f}".format(accuracy, misclass))  # x方向上的标签
    plt.show()  # 显示图片

# 使曲线变得顺滑
def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

# 结果可视化
def plt_result(history):
    # 准确性
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # 损失值
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure()
    plt.plot(epochs, smooth_curve(acc), 'r', label='Train accuracy')
    plt.plot(epochs, smooth_curve(val_acc), 'b', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.figure()
    plt.plot(epochs, smooth_curve(loss), 'r', label='Train loss')
    plt.plot(epochs, smooth_curve(val_loss), 'b', label='Validation loss')
    plt.title('Loss')
    plt.legend()

    plt.show()


train_cnn_model()

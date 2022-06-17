import os

import matplotlib.pyplot as plt
from tensorflow.python.keras.api.keras.models import load_model
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import itertools
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.api.keras import layers, models
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.api import keras
from tensorflow.python.keras.api.keras.layers import Conv2D
from tensorflow.python.keras.api.keras.layers import MaxPooling2D
from tensorflow.python.keras.api.keras.layers import Flatten
from tensorflow.python.keras.api.keras.layers import Dropout
from tensorflow.python.keras.api.keras.layers import Dense

from tensorflow.python.keras.api.keras.utils import plot_model


# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap='BuGn', normalize=False):
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[1][0]
    fn = cm[0][1]
    print(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))  # 准确率
    misclass = 1 - accuracy  # 错误率
    recall = (tp) / (tp + fn)  # 召回率
    precision = (tp) / (tp + fp)  # 精确率
    F1score = 2 * precision * recall / (precision + recall)  # F1score
    print("准确率:" + str(round(accuracy, 4)))
    print("错误率:" + str(round(misclass, 4)))
    print("召回率:" + str(round(recall, 4)))
    print("精确率:" + str(round(precision, 4)))
    print("F1score:" + str(round(F1score, 4)))

    if cmap is None:
        plt.get_cmap('Greens')  # 颜色设置成蓝色
    plt.figure(figsize=(22, 20))  # 设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示图片
    plt.title(title)  # 显示标题
    plt.colorbar()  # 绘制颜色条

    # 设置x,y坐标标签
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)  # x坐标
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
    plt.xlabel(
        "Predicted label\n accuracy={:0.4f}\n misclass={:0.4f}\n recall={:0.4f}\n precision={:0.4f}\n F1score={:0.4f}".format(
            accuracy, misclass, recall, precision,
            F1score))  # x方向上的标签
    plt.show()  # 显示图片

wh = 200
batch_size = 50

# 测试数据图片归一化
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 训练文件路径
train_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\train'
# 验证文件路径
validation_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\validation'

test_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\test'

test_generator = test_datagen.flow_from_directory(
    # 验证文件路径
    test_dir,
    target_size=(wh, wh),
    batch_size=batch_size,
    shuffle=False,  # 不打乱标签
    class_mode='categorical'
)

path = 'data\catDogFight13-DenseNet121.h5'
model = load_model(path)

y_true = test_generator.classes
y_pred = model.predict(test_generator, batch_size=50, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=['cats', 'dogs'])
# y_pred = model.predict(test_generator, batch_size=32, verbose=1)
# y_pred_classes = np.argmax(y_pred, axis=1)
# print(y_pred)
# print(y_pred.shape)
# print(y_pred_classes)
# print(test_generator.classes)
# confusion_mtx = confusion_matrix(y_true=test_generator.classes, y_pred=y_pred_classes)
# plot_confusion_matrix(confusion_mtx, normalize=True, target_names=['cats', 'dogs'])

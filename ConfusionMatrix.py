import os
from tensorflow.python.keras.api.keras.models import load_model
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 去掉加载GPU的警告

# 绘制混淆矩阵
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap='BuGn', normalize=False):
    tp = cm[0][0]
    tn = cm[1][1]
    fp = cm[1][0]
    fn = cm[0][1]
    print('\n混淆矩阵:')
    print(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))  # 准确率
    misclass = 1 - accuracy  # 错误率
    recall = (tp) / (tp + fn)  # 召回率
    precision = (tp) / (tp + fp)  # 精确率
    F1score = 2 * precision * recall / (precision + recall)  # F1score
    print("\n准确率:" + str(round(accuracy, 4)))
    print("错误率:" + str(round(misclass, 4)))
    print("召回率:" + str(round(recall, 4)))
    print("精确率:" + str(round(precision, 4)))
    print("F1 Score:" + str(round(F1score, 4)))

    if cmap is None:
        plt.get_cmap('Greens')  # 颜色设置成蓝色
    plt.figure(figsize=(6, 6))  # 设置窗口尺寸
    plt.imshow(cm, interpolation='nearest', cmap=cmap)  # 显示图片
    plt.title(title)  # 显示标题
    plt.colorbar()  # 绘制颜色条

    # 设置x,y坐标标签
    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)  # x坐标
        plt.yticks(tick_marks, target_names)  # y坐标

    if normalize:
        cm1 = cm.astype('float32') / cm.sum(axis=1)
        cm1 = np.round(cm, 2)  # 对数字保留两位小数

    thresh = cm1.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):  # 将cm.shape[0]、cm.shape[1]中的元素组成元组，遍历元组中每一个数字
        if normalize:  # 标准化
            plt.text(j, i, format(cm[i, j]),  # 保留两位小数
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色
        else:  # 非标准化
            plt.text(j, i, format(cm[i, j]),
                     horizontalalignment="center",  # 数字在方框中间
                     color="white" if cm[i, j] > thresh else "black")  # 设置字体颜色

    # plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域
    plt.ylabel('True label')  # y方向上的标签
    plt.xlabel(
        "Predicted label\n\n accuracy={:0.4f}    misclass={:0.4f}\nrecall={:0.4f}    precision={:0.4f}    F1score={:0.4f}".format(
            accuracy, misclass, recall, precision,
            F1score))  # x方向上的标签
    plt.show()  # 显示图片

wh = 200

# 测试数据图片归一化
test_datagen = ImageDataGenerator(rescale=1. / 255)

# 测试文件路径
test_dir = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\test'

test_generator = test_datagen.flow_from_directory(
    # 验证文件路径
    test_dir,
    target_size=(wh, wh),
    batch_size=50,
    shuffle=False,  # 不打乱标签
    class_mode='categorical'
)

path = 'data/catDogFight13-DenseNet121-f.h5'
model = load_model(path)

print('正在预测...')
y_true = test_generator.classes
y_pred = model.predict(test_generator, batch_size=50, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
confusion_mtx = confusion_matrix(y_true=y_true, y_pred=y_pred)
plot_confusion_matrix(confusion_mtx, normalize=True, target_names=['cats', 'dogs'])

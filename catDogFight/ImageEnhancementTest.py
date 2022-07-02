import os
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tensorflow.python.keras.api.keras.preprocessing.image import ImageDataGenerator

dir = "H:\\Machine Learning\\CNN\\CNNProjects\\data\\train_data\\test\\cat"
fnames = [os.path.join(dir, fname) for fname in os.listdir(dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(200, 200))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0

datagen = ImageDataGenerator(
    rescale=1. / 255,  # 归一化
    rotation_range=40,  # 旋转角度
    width_shift_range=0.2,  # 水平偏移
    height_shift_range=0.2,  # 垂直偏移
    shear_range=0.2,  # 随机错切变换的角度
    zoom_range=0.2,  # 随机缩放的范围
    horizontal_flip=True  # 随机将一半图像水平翻转
)

for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

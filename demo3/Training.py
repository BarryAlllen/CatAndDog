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

# 1.构建网络模型
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

#编译模型
opt = SGD(lr=0.001, momentum=0.9)

# 2.配置优化器
model.compile(loss='binary_crossentropy',
              # optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# 查看数据增强后的效果
# train_cats_dir="H:\\Machine Learning\\CNN\\CNNProjects\\data\\train2\\train\\cats"
# # This is module with image preprocessing utilities
# fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
# # We pick one image to "augment"
# img_path = fnames[3]
# # Read the image and resize it
# img = image.load_img(img_path, target_size=(150, 150))
# # Convert it to a Numpy array with shape (150, 150, 3)
# x = image.img_to_array(img)
# # Reshape it to (1, 150, 150, 3)
# x = x.reshape((1,) + x.shape)
# # The .flow() command below generates batches of randomly transformed images.
# # It will loop indefinitely, so we need to `break` the loop at some point!
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(image.array_to_img(batch[0]))
#     i += 1
#     if i % 4 == 0:
#         break
# plt.show()


# 3.图片格式转化
# 所有图像将按1/255重新缩放
# train_datagen = ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)
#
train_dir = "H:\\Machine Learning\\CNN\\CNNProjects\\data\\train3\\train"
validation_dir = "H:\\Machine Learning\\CNN\\CNNProjects\\data\\train3\\validation"
# train_generator = train_datagen.flow_from_directory(
#         # 这是目标目录
#         train_dir,
#         # 所有图像将调整为150x150
#         target_size=(150, 150),
#         batch_size=20,
#         # 因为我们使用二元交叉熵损失，我们需要二元标签
#         class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=20,
#         class_mode='binary')

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True, )
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)
# train_generator = train_datagen.flow_from_directory(
#         # This is the target directory
#         train_dir,
#         # All images will be resized to 150x150
#         target_size=(150, 150),
#         batch_size=32,
#         # Since we use binary_crossentropy loss, we need binary labels
#         class_mode='binary')
# validation_generator = test_datagen.flow_from_directory(
#         validation_dir,
#         target_size=(150, 150),
#         batch_size=32,
#         class_mode='binary')
train_generator = train_datagen.flow_from_directory(
    # 这是目标目录
    train_dir,
    # 所有图像将调整为150x150
    target_size=(150, 150),
    batch_size=20,
    # 因为我们使用二元交叉熵损失，我们需要二元标签
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

# 查看上面对于图片预处理的处理结果
# for data_batch, labels_batch in train_generator:
#     print('data batch shape:', data_batch.shape)
#     print('labels batch shape:', labels_batch.shape)
#     break

# 4.开始训练模型
# 模型训练过程
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)

# 5.保存模型
# 保存训练得到的的模型
model.save('data\cats_and_dogs_small_4.h5')

# 6.结果可视化
# 对于模型进行评估，查看预测的准确性
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

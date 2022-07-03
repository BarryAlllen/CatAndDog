#预处理
import os, shutil

# 原始路径
original_path = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train'

# 分类后路径
sort_path = 'H:\\Machine Learning\\CNN\\CNNProjects\\data\\train4'
os.mkdir(sort_path)

# 创建训练路径 train
train_path = os.path.join(sort_path, 'train')
os.mkdir(train_path)

# 创建验证路径 validation
validation_path = os.path.join(sort_path, 'validation')
os.mkdir(validation_path)

# 创建测试路径 test
test_path = os.path.join(sort_path, 'test')
os.mkdir(test_path)

# 创建猫训练路径 train cat
train_cat_path = os.path.join(train_path, 'cat')
os.mkdir(train_cat_path)

# 狗训练路径 train dog
train_dog_path = os.path.join(train_path, 'dog')
os.mkdir(train_dog_path)

# 猫验证路径 validation cat
validation_cat_path = os.path.join(validation_path, 'cat')
os.mkdir(validation_cat_path)

# 狗验证路径 validation dog
validation_dog_path = os.path.join(validation_path, 'dog')
os.mkdir(validation_dog_path)

# 猫测试路径 test cat
test_cat_path = os.path.join(test_path, 'cat')
os.mkdir(test_cat_path)

# 狗测试路径 test dog
test_dog_path = os.path.join(test_path, 'dog')
os.mkdir(test_dog_path)

# 7500张为训练
amount_train = 7500
# 猫
file_name = ['cat.{}.jpg'.format(i) for i in range(amount_train)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(train_cat_path, n)
    shutil.copyfile(src, dst)

# 狗
file_name = ['dog.{}.jpg'.format(i) for i in range(amount_train)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(train_dog_path, n)
    shutil.copyfile(src, dst)

# 2500张为验证
amount_validation = amount_train + 2500
# 猫
file_name = ['cat.{}.jpg'.format(i) for i in range(amount_train, amount_validation)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(validation_cat_path, n)
    shutil.copyfile(src, dst)

# 狗
file_name = ['dog.{}.jpg'.format(i) for i in range(amount_train, amount_validation)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(validation_dog_path, n)
    shutil.copyfile(src, dst)

# 2500张为测试
amount_test = amount_validation + 2500
# 猫
file_name = ['cat.{}.jpg'.format(i) for i in range(amount_validation, amount_test)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(test_cat_path, n)
    shutil.copyfile(src, dst)

# 狗
file_name = ['dog.{}.jpg'.format(i) for i in range(amount_validation, amount_test)]
for n in file_name:
    src = os.path.join(original_path, n)
    dst = os.path.join(test_dog_path, n)
    shutil.copyfile(src, dst)


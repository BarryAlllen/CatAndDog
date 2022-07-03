#基于DenseNet网络的猫狗识别
#### *Cat and dog recognition based on DenseNet —— CatAndDog猫狗大战*
___
<div align=center>
<img src="https://storage.googleapis.com/kaggle-competitions/kaggle/3362/media/woof_meow.jpg" width="300" height="185">
</div>

1. ###概述
>利用DenseNet121神经网络模型，搭配Adam优化器，对猫狗进行二分类识别。

2. ###文件描述
> + CatDog.py：训练模型主程序
>
> + ConfusionMatrix.py：输出混淆矩阵、准确率、召回率、精确率、F1 Score
> 
> + Forecast.py：利用训练好的模型预测猫狗图片
> 
> + Histogram.py：输出数据集的直方图
> 
> + ImageEnhancementTest.py：用于测试可视化图像增强的效果
> 
> + Pretreatment.py：用于对拿到的数据集进行预处理
>
> >数据集是由Kaggle提供的猫狗图片，共计25000张，猫与狗各12500张。
> >
> >数据集网址：https://www.kaggle.com/c/dogs-vs-cats
____

__作者：*Barry*__

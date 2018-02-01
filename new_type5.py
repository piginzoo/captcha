# __author__ = 'lllcho'
# __date__ = '2015/9/29'
import cv2
import h5py
import codecs
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2

letters = list('0123456789abcdefghijklmnopqrstuvwxyz')
weight_decay = 0.001

# -*- coding:utf-8 -*-  
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


# batch_size 太小会导致训练慢，过拟合等问题，太大会导致欠拟合。所以要适当选择
batch_size = 128
# 0-9手写数字一个有10个类别
num_classes = 10
# 12次完整迭代，差不多够了
epochs = 4
# 输入的图片是28*28像素的灰度图
img_rows, img_cols = 28, 28
# 训练集，测试集收集非常方便
(x_train, y_train), (x_test, y_test) = mnist.load_data()
 
# keras输入数据有两种格式，一种是通道数放在前面，一种是通道数放在后面，
# 其实就是格式差别而已
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
# 把数据变成float32更精确
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# 把类别0-9变成2进制，方便训练
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)


# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
num_model = Sequential()
# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
# 卷积核的窗口选用3*3像素窗口
num_model.add(Conv2D(32,activation='relu',input_shape=input_shape,nb_row=3,nb_col=3))
# 64个通道的卷积层
num_model.add(Conv2D(64, activation='relu',nb_row=3,nb_col=3))
# 池化层是2*2像素的
num_model.add(MaxPooling2D(pool_size=(2, 2)))
# 对于池化层的输出，采用0.35概率的Dropout
num_model.add(Dropout(0.35))
# 展平所有像素，比如[28*28] -> [784]
num_model.add(Flatten())
# 对所有像素使用全连接层，输出为128，激活函数选用relu
num_model.add(Dense(128, activation='relu'))
# 对输入采用0.5概率的Dropout
num_model.add(Dropout(0.5))
# 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
num_model.add(Dense(num_classes, activation='softmax'))
# 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
num_model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 牛逼的Sequential类可以让我们灵活地插入不同的神经网络层
char_model = Sequential()
# 加上一个2D卷积层， 32个输出（也就是卷积通道），激活函数选用relu，
# 卷积核的窗口选用3*3像素窗口
char_model.add(Conv2D(32,activation='relu',input_shape=input_shape,nb_row=3,nb_col=3))
# 64个通道的卷积层
char_model.add(Conv2D(64, activation='relu',nb_row=3,nb_col=3))
# 池化层是2*2像素的
char_model.add(MaxPooling2D(pool_size=(2, 2)))
# 对于池化层的输出，采用0.35概率的Dropout
char_model.add(Dropout(0.35))
# 展平所有像素，比如[28*28] -> [784]
char_model.add(Flatten())
# 对所有像素使用全连接层，输出为128，激活函数选用relu
char_model.add(Dense(128, activation='relu'))
# 对输入采用0.5概率的Dropout
char_model.add(Dropout(0.5))
# 对刚才Dropout的输出采用softmax激活函数，得到最后结果0-9
char_model.add(Dense(num_classes, activation='softmax'))
# 模型我们使用交叉熵损失函数，最优化方法选用Adadelta
char_model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

comp = 'type5_test1'
img_dir = './image/' + comp + '/'
f_csv = codecs.open("result/" + comp + '.csv', 'w', 'utf-8')
# for nb_img in range(1,20001):
#     name=comp+'_'+str(nb_img)+'.png'
import os,csv
filename = os.path.join(os.getcwd(), 'type5_train.csv')
y_train = []
if os.path.exists(filename):
	with open(filename, 'r') as f:
	  reader = csv.reader(f)
	  for item in reader:
          #print item[1],len(item[1])#识别码
          y_train.append(len(item[1]))



names = os.listdir(img_dir)
for name in names:



    imgname = img_dir + name
    img = cv2.imread(imgname, cv2.IMREAD_GRAYSCALE)
    retval, t = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY_INV)
    s = t.sum(axis=0)
    y1, y2 = (s > np.median(s) + 5).nonzero()[0][0], (s > np.median(s) + 5).nonzero()[0][-1]
    x1, x2 = 0, 36
    im = img[x1:x2, y1 - 2:y2 + 3]
    retval, im = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
    im0 = im[x1:x2, 1:-1]
    if im.shape[1] < 100:
        im = np.concatenate((im, np.zeros((36, 100 - im.shape[1]), dtype='uint8')), axis=1)
    else:
        im = cv2.resize(im, (100, 36))
    I = im > 127
    I = I.astype(np.float32).reshape((1, 1, 36, 100))

    #n = num_model.predict_classes(I, verbose=0) + 4
    num_model.fit(I, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))

    #1.该如何训练呢？
    #   1.1 该把数据准备好呢？准备成啥样啥格式呢？
    #   1.2 批次是不是不用考虑？一次加载进去，然后tf/keras自己靠batch参数控制？keras不是有批次输入么？
    #2.该如何中途保存呢？
    #3.该如何输出呢？
    #4.如何验证正确率和loss？loss是错误率么？是一个batch评价一次么？
    #其他问题：
    #   - 去研究一下minst的数据加载，如何填充(？, 1, 36, 100)第一个维度的？
    #       应该是一次都加载进去么？


    n = num_model.predict_classes(I, verbose=0) + 4
    im1 = np.zeros((36, 150), dtype=np.uint8)
    im1[:, 10:im0.shape[1] + 10] = im0
    step = im0.shape[1] / float(n)
    center = [i + step / 2 for i in np.arange(0, im0.shape[1], step).tolist()]
    imgs = np.zeros((n, 1, 36, 20), dtype=np.float32)
    for i, c in enumerate(center):
        imgs[i, 0, :, :] = im1[:, c:c + 20]
    classes = char_model.predict_classes(imgs.astype('float32') / 255.0, verbose=0)
    result = []
    for c in classes:
        result.append(letters[c])
    print(name, ''.join(result).upper())
    f_csv.write(name + ',' + ''.join(result).upper() + '\n')
f_csv.close()

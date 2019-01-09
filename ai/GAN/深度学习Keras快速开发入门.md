```
深度学习：Keras快速开发入门

https://github.com/yanchao727/keras_book
```

```
第1章　Keras概述	1
1.1 Keras簡介	1
1.1.1 Keras 2	1
1.1.2 Keras功能構成	4
1.2 Keras特點	6
1.3 主要深度學習框架	8
1.3.1 Caffe	8
1.3.2 Torch	10
1.3.3 Keras	12
1.3.4 MXNet	12
1.3.5 TensorFlow	13
1.3.5 CNTK	14
1.3.6 Theano	14

第2章　Keras的安裝與配置	16
2.1 Windows環境下安裝Keras	16
2.1.1 硬體設定	16
2.1.2 Windows版本	18
2.1.3 Microsoft Visual Studio版本	18
2.1.4 Python環境	18
2.1.5 CUDA	18
2.1.6 加速庫CuDNN	19
2.1.7 Keras框架的安裝	19
2.2 Linux環境下的安裝	20
2.2.1 硬體設定	20
2.2.2 Linux版本	21
2.2.3 Ubuntu環境的設置	22
2.2.4 CUDA開發環境	22
2.2.5 加速庫cuDNN	23
2.2.6 Keras框架安裝	24

第3章　Keras快速上手	25
3.1 基本概念	25
3.2 初識Sequential模型	29
3.3 一個MNIST手寫數位實例	30
3.3.1 MNIST數據準備	30
3.3.2 建立模型	31
3.3.3 訓練模型	32

第4章　Keras模型的定義	36
4.1 Keras模型	36
4.2 Sequential模型	38
4.2.1 Sequential模型介面	38
4.2.2 Sequential模型的資料登錄	48
4.2.3 模型編譯	49
4.2.4 模型訓練	50
4.3 函數式模型	51
4.3.1 全連接網路	52
4.3.2 函數模型介面	53
4.3.3 多輸入和多輸出模型	63
4.3.4 共用層模型	67


第5章　Keras網路結構	71
5.1 Keras層物件方法	71
5.2 常用層	72
5.2.1 Dense層	72
5.2.2 Activation層	74
5.2.3 Dropout層	75
5.2.4 Flatten層	75
5.2.5 Reshape層	76
5.2.6 Permute層	77
5.2.7 RepeatVector層	78
5.2.8 Lambda層	79
5.2.9 ActivityRegularizer層	80
5.2.10 Masking層	81
5.3 卷積層	82
5.3.1 Conv1D層	82
5.3.2 Conv2D層	84
5.3.3 SeparableConv2D層	87
5.3.4 Conv2DTranspose層	91
5.3.5 Conv3D層	94
5.3.6 Cropping1D層	97
5.3.6 Cropping2D層	97
5.3.7 Cropping3D層	98
5.3.8 UpSampling1D層	99
5.3.9 UpSampling2D層	100
5.3.10 UpSampling3D層	101
5.3.11 ZeroPadding1D層	102
5.3.12 ZeroPadding2D層	103
5.3.13 ZeroPadding3D層	104
5.4 池化層	105
5.4.1 MaxPooling1D層	105
5.4.2 MaxPooling2D層	106
5.4.3 MaxPooling3D層	108
5.4.4 AveragePooling1D層	109
5.4.5 AveragePooling2D層	110
5.4.6 AveragePooling3D層	111
5.4.7 GlobalMaxPooling1D層	112
5.4.8 GlobalAveragePooling1D層	113
5.4.9 GlobalMaxPooling2D層	113
5.4.10 GlobalAveragePooling2D層	114
5.5 局部連接層	115
5.5.1 LocallyConnected1D層	115
5.5.2 LocallyConnected2D層	117
5.6 迴圈層	120
5.6.1 Recurrent層	120
5.6.2 SimpleRNN層	124
5.6.3 GRU層	126
5.6.4 LSTM層	127
5.7 嵌入層	129
5.8 融合層	131
5.9 啟動層	134
5.9.1 LeakyReLU層	134
5.9.2 PReLU層	134
5.9.3 ELU層	135
5.9.4 ThresholdedReLU層	136
5.10 規範層	137
5.11 雜訊層	139
5.11.1 GaussianNoise層	139
5.11.2 GaussianDropout層	139
5.12 包裝器Wrapper	140
5.12.1 TimeDistributed層	140
5.12.2 Bidirectional層	141
5.13 自訂層	142

第6章　Keras數據預處理	144
6.1 序列數據預處理	145
6.1.1 序列數據填充	145
6.1.2 提取序列跳字樣本	148
6.1.3 生成序列抽樣概率表	151
6.2 文本預處理	153
6.2.1 分割句子獲得單詞序列	153
6.2.2 OneHot序列編碼器	154
6.2.3 單詞向量化	155
6.3 圖像預處理	159


第7章　Keras內置網路配置	167
7.1 模型性能評估模組	168
7.1.1 Keras內置性能評估方法	168
7.1.2 使用Keras內置性能評估	170
7.1.3 自訂性能評估函數	171
7.2 損失函數	171
7.3 優化器函數	174
7.3.1 Keras優化器使用	174
7.3.2 Keras內置優化器	176
7.4 啟動函數	180
7.4.1 添加啟動函數方法	180
7.4.2 Keras內置啟動函數	181
7.4.3 Keras高級啟動函數	185
7.5 初始化參數	189
7.5.1 使用初始化方法	189
7.5.2 Keras內置初始化方法	190
7.5.3 自訂Keras初始化方法	196
7.6 正則項	196
7.6.1 使用正則項	197
7.6.2 Keras內置正則項	198
7.6.3 自訂Keras正則項	198
7.7 參數約束項	199
7.7.1 使用參數約束項	199
7.7.2 Keras內置參數約束項	200

第8章　Keras實用技巧和視覺化	202
8.1 Keras調試與排錯	202
8.1.1 Keras Callback回呼函數與調試技巧	202
8.1.2 備份和還原Keras模型	215
8.2 Keras內置Scikit-Learn介面包裝器	217
8.3 Keras內置視覺化工具	224

第9章　Keras實戰	227
9.1 訓練一個準確率高於90%的Cifar-10預測模型	227
9.1.1 數據預處理	232
9.1.2 訓練	233

9.2 在Keras模型中使用預訓練詞向量判定文本類別	239
9.2.1 資料下載和實驗方法	240
9.2.2 數據預處理	241
9.2.3 訓練	245

9.3 用Keras實現DCGAN生成對抗網路還原MNIST樣本	247
9.3.1 DCGAN網路拓撲結構	250
9.3.2 訓練	254


```

# 9.3 用Keras實現DCGAN生成對抗網路還原MNIST樣本	
```
# -*- coding:utf-8 -*-
'''
DCGAN on MNIST using Keras
Author: Rowel Atienza
Project: https://github.com/roatienza/Deep-Learning-Experiments
Dependencies: tensorflow 1.0 and keras 2.0
Usage: python3 dcgan_mnist.py
'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class DCGAN(object):
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4
        # In: 28 x 28 x 1, depth = 1
        # Out: 10 x 10 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same', activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same',\
                activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same',\
                activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same',\
                activation=LeakyReLU(alpha=0.2)))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()
        return self.D

    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM:
            return self.DM
        optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM:
            return self.AM
        optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return self.AM

class MNIST_DCGAN(object):
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        self.x_train = input_data.read_data_sets("mnist",\
                one_hot=True).train.images
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
                self.img_cols, 1).astype(np.float32)

        self.DCGAN = DCGAN()
        self.discriminator =  self.DCGAN.discriminator_model()
        self.adversarial = self.DCGAN.adversarial_model()
        self.generator = self.DCGAN.generator()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])
        for i in range(train_steps):
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2*batch_size, 1])
            y[batch_size:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_loss = self.adversarial.train_on_batch(noise, y)
            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)
            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))

    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step
            images = self.generator.predict(noise)
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))
        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    mnist_dcgan = MNIST_DCGAN()
    timer = ElapsedTimer()
    mnist_dcgan.train(train_steps=10000, batch_size=256, save_interval=500)
    timer.elapsed_time()
    mnist_dcgan.plot_images(fake=False, save2file=True)
```

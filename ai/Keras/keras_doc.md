```
Keras

在Google Colab中運行

在GitHub上查看源代碼
Keras是一個用於構建和培訓深度學習模型的高級API。它用於快速原型設計，高級研究和生產，具有三個主要優勢：

用戶友好的
Keras具有針對常見用例優化的簡單，一致的界面。它為用戶錯誤提供清晰且可操作的反饋。
模塊化和可組合的
Keras模型是通過將可配置的構建塊連接在一起而製定的，幾乎沒有限制。
易於擴展
編寫自定義構建塊以表達研究的新想法。創建新圖層，損失函數並開發最先進的模型。
導入tf.keras
tf.keras是TensorFlow實現的 Keras API規範。這是一個用於構建和訓練模型的高級API，其中包括對TensorFlow特定功能的一流支持，例如急切執行， tf.data管道和估算器。 tf.keras使TensorFlow更易於使用而不犧牲靈活性和性能。

要開始使用，tf.keras請將其作為TensorFlow程序設置的一部分導入：

!pip install -q pyyaml  # Required to save models in YAML format
import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)
1.12.0
2.1.6-TF
tf.keras 可以運行任何與Keras兼容的代碼，但請記住：

在tf.keras最新發布TensorFlow版本可能不一樣的最新keras一封來自PyPI版本。檢查。tf.keras.version
當保存模型的權重，tf.keras默認為 檢查點的格式。通過save_format='h5'使用HDF5。
構建一個簡單的模型
順序模型
在Keras中，您可以組裝圖層來構建模型。模型（通常）是圖層圖。最常見的模型類型是一疊層： tf.keras.Sequential模型。

構建一個簡單的，完全連接的網絡（即多層感知器）：

model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))
配置圖層
有許多tf.keras.layers可用的常見構造函數參數：

activation：設置圖層的激活功能。此參數由內置函數的名稱或可調用對象指定。默認情況下，不應用任何激活。
kernel_initializer和bias_initializer：創建圖層權重（內核和偏差）的初始化方案。此參數是名稱或可調用對象。這默認為"Glorot uniform"初始化程序。
kernel_regularizer和bias_regularizer：應用圖層權重（內核和偏差）的正則化方案，例如L1或L2正則化。默認情況下，不應用正則化。
以下tf.keras.layers.Dense使用構造函數參數實例化圖層：

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
訓練和評估
設置培訓
構建模型後，通過調用compile方法配置其學習過程 ：

model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
tf.keras.Model.compile 有三個重要論點：

optimizer：此對象指定訓練過程。通過它優化器實例tf.train模塊，例如 tf.train.AdamOptimizer，tf.train.RMSPropOptimizer，或 tf.train.GradientDescentOptimizer。
loss：優化期間最小化的功能。常見的選擇包括均方誤差（mse）categorical_crossentropy，和 binary_crossentropy。損失函數由名稱指定或通過從tf.keras.losses模塊傳遞可調用對象來指定。
metrics：用於監控培訓。這些是tf.keras.metrics模塊中的字符串名稱或可調用項。
以下顯示了配置培訓模型的幾個示例：

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
輸入NumPy數據
對於小型數據集，請使用內存中的NumPy 陣列來訓練和評估模型。該模型使用以下fit方法“適合”訓練數據：

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
大紀元1/10
1000/1000 [==============================]  -  0s 191us /步 - 損失：11.6073  -  categorical_accuracy：0.1050
大紀元2/10
1000/1000 [==============================]  -  0s 51us /步 - 損失：11.5352  -  categorical_accuracy：0.1050
大紀元3/10
1000/1000 [==============================]  -  0s 57us / step  - 損失：11.5283  -  categorical_accuracy：0.1170
大紀元4/10
1000/1000 [==============================]  -  0s 61us /步 - 損失：11.5251  -  categorical_accuracy：0.1220
大紀元5/10
1000/1000 [==============================]  -  0s 63us / step  -  loss：11.5231  -  categorical_accuracy：0.1010
大紀元6/10
1000/1000 [==============================]  -  0s 57us / step  - 損失：11.5182  -  categorical_accuracy：0.1090
大紀元7/10
1000/1000 [==============================]  -  0s 58us / step  -  loss：11.5221  -  categorical_accuracy：0.0930
大紀元8/10
1000/1000 [==============================]  -  0s 61us /步 - 損失：11.5129  -  categorical_accuracy：0.1350
大紀元9/10
1000/1000 [==============================]  -  0s 62us / step  - 損失：11.5150  -  categorical_accuracy：0.1260
Epoch 10/10
1000/1000 [==============================]  -  0s 60us / step  - 損失：11.5071  -  categorical_accuracy：0.1240

tf.keras.Model.fit 有三個重要論點：

epochs：培訓結構為時代。一個紀元是對整個輸入數據的一次迭代（這是以較小的批次完成的）。
batch_size：當傳遞NumPy數據時，模型將數據分成較小的批次，並在訓練期間迭代這些批次。此整數指定每個批次的大小。請注意，如果樣本總數不能被批量大小整除，則最後一批可能會更小。
validation_data：在對模型進行原型設計時，您希望輕鬆監控其在某些驗證數據上的性能。傳遞這個參數 - 輸入和標籤的元組 - 允許模型在每個紀元的末尾以傳遞數據的推理模式顯示損失和度量。
這是一個使用示例validation_data：

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
訓練1000個樣本，驗證100個樣本
大紀元1/10
1000/1000 [==============================]  -  0s 106us /步 - 損失：11.4556  -  categorical_accuracy：0.0950  -  val_loss ：11.5346  -  val_categorical_accuracy：0.0800
大紀元2/10
1000/1000 [==============================]  -  0s 63us /步 - 損失：11.4518  -  categorical_accuracy：0.0970  -  val_loss ：11.5400  -  val_categorical_accuracy：0.1700
大紀元3/10
1000/1000 [==============================]  -  0s 71us /步 - 損失：11.4525  -  categorical_accuracy：0.0920  -  val_loss ：11.5251  -  val_categorical_accuracy：0.0800
大紀元4/10
1000/1000 [==============================]  -  0s 62us / step  - 損失：11.4499  -  categorical_accuracy：0.1000  -  val_loss ：11.5561  -  val_categorical_accuracy：0.1300
大紀元5/10
1000/1000 [==============================]  -  0s 66us /步 - 損失：11.4492  -  categorical_accuracy：0.1110  -  val_loss ：11.5293  -  val_categorical_accuracy：0.0600
大紀元6/10
1000/1000 [==============================]  -  0s 62us /步 - 損失：11.4474  -  categorical_accuracy：0.1130  -  val_loss ：11.5684  -  val_categorical_accuracy：0.0600
大紀元7/10
1000/1000 [==============================]  -  0s 67us /步 - 損失：11.4504  -  categorical_accuracy：0.1150  -  val_loss ：11.5319  -  val_categorical_accuracy：0.1200
大紀元8/10
1000/1000 [==============================]  -  0s 64us /步 - 損失：11.4453  -  categorical_accuracy：0.1210  -  val_loss ：11.5535  -  val_categorical_accuracy：0.1100
大紀元9/10
1000/1000 [==============================]  -  0s 68us / step  - 損失：11.4421  -  categorical_accuracy：0.1290  -  val_loss ：11.5439  -  val_categorical_accuracy：0.1000
Epoch 10/10
1000/1000 [==============================]  -  0s 65us / step  - 損失：11.4432  -  categorical_accuracy：0.1000  -  val_loss ：11.5436  -  val_categorical_accuracy：0.1300

輸入tf.data數據集
使用數據集API可擴展到大型數據集或多設備培訓。將tf.data.Dataset實例傳遞給fit 方法：

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
大紀元1/10
30/30 [==============================]  -  0s 5ms /步 - 損失：11.4307  -  categorical_accuracy：0.1125
大紀元2/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4335  -  categorical_accuracy：0.1260
大紀元3/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4212  -  categorical_accuracy：0.1146
大紀元4/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4330  -  categorical_accuracy：0.1292
大紀元5/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4540  -  categorical_accuracy：0.1260
大紀元6/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4235  -  categorical_accuracy：0.1365
大紀元7/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4397  -  categorical_accuracy：0.1250
大紀元8/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4397  -  categorical_accuracy：0.1562
大紀元9/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4412  -  categorical_accuracy：0.1406
Epoch 10/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4156  -  categorical_accuracy：0.1417

這裡，該fit方法使用steps_per_epoch參數 - 這是模型在移動到下一個紀元之前運行的訓練步數。由於 Dataset產生批量數據，此代碼段不需要batch_size。

數據集也可用於驗證：

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
大紀元1/10
30/30 [==============================]  -  0s 6ms /步 - 損失：11.3982  -  categorical_accuracy：0.1500  -  val_loss ：11.5993  -  val_categorical_accuracy：0.1042
大紀元2/10
30/30 [==============================]  -  0s 1ms /步 - 損失：11.3960  -  categorical_accuracy：0.1521  -  val_loss ：11.2608  -  val_categorical_accuracy：0.0312
大紀元3/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.3888  -  categorical_accuracy：0.1562  -  val_loss ：11.5409  -  val_categorical_accuracy：0.1979
大紀元4/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.3999  -  categorical_accuracy：0.1594  -  val_loss ：11.4413  -  val_categorical_accuracy：0.0417
大紀元5/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4191  -  categorical_accuracy：0.1865  -  val_loss ：11.6075  -  val_categorical_accuracy：0.1354
大紀元6/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.3852  -  categorical_accuracy：0.1781  -  val_loss ：11.2979  -  val_categorical_accuracy：0.0312
大紀元7/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4046  -  categorical_accuracy：0.1583  -  val_loss ：11.6355  -  val_categorical_accuracy：0.0938
大紀元8/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4117  -  categorical_accuracy：0.1552  -  val_loss ：11.6064  -  val_categorical_accuracy：0.1667
大紀元9/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.4087  -  categorical_accuracy：0.1635  -  val_loss ：11.6565  -  val_categorical_accuracy：0.0833
Epoch 10/10
30/30 [==============================]  -  0s 2ms /步 - 損失：11.3840  -  categorical_accuracy：0.1740  -  val_loss ：11.3825  -  val_categorical_accuracy：0.1354

評估和預測
該tf.keras.Model.evaluate和tf.keras.Model.predict方法可以使用NumPy的數據和tf.data.Dataset。

要評估所提供數據的推理模式損失和指標：

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
1000/1000 [==============================]  -  0s 62us /步
30/30 [==============================]  -  0s 3ms /步

[11.394975598653158,0.165625]
並且作為NumPy數組，預測所提供數據的推斷中最後一層的輸出：

result = model.predict(data, batch_size=32)
print(result.shape)
（1000,10）
構建高級模型
功能API
該tf.keras.Sequential模型是一個簡單的圖層堆棧，不能代表任意模型。使用 Keras功能API 構建複雜的模型拓撲，例如：

多輸入型號，
多輸出型號，
具有共享層的模型（同一層被稱為多次），
具有非順序數據流的模型（例如，殘餘連接）。
使用功能API構建模型的工作方式如下：

圖層實例可調用並返回張量。
輸入張量和輸出張量用於定義tf.keras.Model 實例。
這個模型就像Sequential模型一樣訓練。
以下示例使用功能API構建一個簡單，完全連接的網絡：

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
實例化給定輸入和輸出的模型。

model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
大紀元1/5
1000/1000 [==============================]  -  0s 211us /步 - 損失：11.6775  -  acc：0.1060
大紀元2/5
1000/1000 [==============================]  -  0s 61us /步 - 損失：11.5813  -  acc：0.1020
大紀元3/5
1000/1000 [==============================]  -  0s 69us /步 - 損失：11.5461  -  acc：0.1010
大紀元4/5
1000/1000 [==============================]  -  0s 60us /步 - 損失：11.5359  -  acc：0.1070
大紀元5/5
1000/1000 [==============================]  -  0s 65us /步 - 損失：11.5273  -  acc：0.1130

模型子類化
通過子類化tf.keras.Model和定義自己的前向傳遞來構建完全可自定義的模型。在__init__方法中創建圖層並將它們設置為類實例的屬性。在call方法中定義正向傳遞。

當啟用eager執行時，模型子類化特別有用， 因為可以強制寫入正向傳遞。

關鍵點：為工作使用正確的API。雖然模型子類化提供了靈活性，但其代價是更高的複雜性和更多的用戶錯誤機會。如果可能，請選擇功能API。
以下示例顯示了tf.keras.Model使用自定義正向傳遞的子類：

class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)
實例化新的模型類：

model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
大紀元1/5
1000/1000 [==============================]  -  0s 193us /步 - 損失：11.6119  -  acc：0.0970
大紀元2/5
1000/1000 [==============================]  -  0s 47us / step  - 損失：11.5825  -  acc：0.0900
大紀元3/5
1000/1000 [==============================]  -  0s 48us / step  - 損失：11.5576  -  acc：0.1010
大紀元4/5
1000/1000 [==============================]  -  0s 64us /步 - 損失：11.5455  -  acc：0.1040
大紀元5/5
1000/1000 [==============================]  -  0s 60us /步 - 損失：11.5378  -  acc：0.1200

自定義圖層
通過子類化tf.keras.layers.Layer並實現以下方法來創建自定義層：

build：創建圖層的權重。使用該add_weight 方法添加權重。
call：定義前進傳球。
compute_output_shape：指定在給定輸入形狀的情況下如何計算圖層的輸出形狀。
可選地，可以通過實現get_config方法和from_config類方法來序列化層。
以下是matmul使用內核矩陣實現輸入的自定義層的示例：

class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Make sure to call the `build` method at the end
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
使用自定義圖層創建模型：

model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# The compile step specifies the training configuration
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
大紀元1/5
1000/1000 [==============================]  -  0s 186us / step  - 損失：11.5429  -  acc：0.0930
大紀元2/5
1000/1000 [==============================]  -  0s 56us /步 - 損失：11.5306  -  acc：0.1010
大紀元3/5
1000/1000 [==============================]  -  0s 49us /步 - 損失：11.5256  -  acc：0.0910
大紀元4/5
1000/1000 [==============================]  -  0s 60us /步 - 損失：11.5241  -  acc：0.0970
大紀元5/5
1000/1000 [==============================]  -  0s 54us /步 - 損失：11.5222  -  acc：0.1160

回調
回調是傳遞給模型的對象，用於在培訓期間自定義和擴展其行為。您可以編寫自己的自定義回調，或使用包含以下內置的內置 tf.keras.callbacks：

tf.keras.callbacks.ModelCheckpoint：定期保存模型的檢查點。
tf.keras.callbacks.LearningRateScheduler：動態改變學習率。
tf.keras.callbacks.EarlyStopping：驗證性能停止改進時的中斷培訓。
tf.keras.callbacks.TensorBoard：使用TensorBoard監控模型的行為 。
要使用a tf.keras.callbacks.Callback，請將其傳遞給模型的fit方法：

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
訓練1000個樣本，驗證100個樣本
大紀元1/5
1000/1000 [==============================]  -  0s 141us /步 - 損失：11.5211  -  acc：0.1190  -  val_loss ：11.5258  -  val_acc：0.1200
大紀元2/5
1000/1000 [==============================]  -  0s 75us /步 - 損失：11.5190  -  acc：0.1140  -  val_loss ：11.5249  -  val_acc：0.1300
大紀元3/5
1000/1000 [==============================]  -  0s 82us /步 - 損失：11.5180  -  acc：0.1060  -  val_loss ：11.5226  -  val_acc：0.1200
大紀元4/5
1000/1000 [==============================]  -  0s 85us /步 - 損失：11.5163  -  acc：0.1150  -  val_loss ：11.5294  -  val_acc：0.0700
大紀元5/5
1000/1000 [==============================]  -  0s 80us /步 - 損失：11.5153  -  acc：0.1190  -  val_loss ：11.5245  -  val_acc：0.0400


保存並恢復
僅重量
使用以下方法保存並加載模型的權重tf.keras.Model.save_weights：

model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')
默認情況下，這會以TensorFlow檢查點文件格式保存模型的權重 。權重也可以保存為Keras HDF5格式（Keras的多後端實現的默認值）：

# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')
僅配置
可以保存模型的配置 - 這可以在沒有任何權重的情況下序列化模型體系結構。即使沒有定義原始模型的代碼，保存的配置也可以重新創建和初始化相同的模型。Keras支持JSON和YAML序列化格式：

# Serialize a model to JSON format
json_string = model.to_json()
json_string
'{“keras_version”：“2.1.6-tf”，“config”：{“layers”：[{“config”：{“units”：64，“bias_initializer”：{“config”：{“dtype”： “float32”}，“class_name”：“Zeros”}，“activation”：“relu”，“trainable”：true，“dtype”：null，“name”：“dense_17”，“activity_regularizer”：null，“kernel_regularizer” “：null，”kernel_constraint“：null，”kernel_initializer“：{”config“：{”seed“：null，”dtype“：”float32“}，”class_name“：”GlorotUniform“}，”bias_constraint“：null， “use_bias”：true，“bias_regularizer”：null}，“class_name”：“密集”}，{“config“：{”units“：10，”bias_initializer“：{”config“：{”dtype“：”float32“}，”class_name“：”Zeros“}，”激活“：”softmax“，”trainable“： true，“dtype”：null，“name”：“dense_18”，“activity_regularizer”：null，“kernel_regularizer”：null，“kernel_constraint”：null，“kernel_initializer”：{“config”：{“seed”：null， “dtype”：“float32”}，“class_name”：“GlorotUniform”}，“bias_constraint”：null，“use_bias”：true，“bias_regularizer”：null}，“class_name”：“密集”}]，“名稱” ：“sequential_3”}，“backend”：“tensorflow”，“class_name”：“順序”}“
import json
import pprint
pprint.pprint(json.loads(json_string))
{'後端'：'tensorflow'，
 'class_name'：'順序'，
 'config'：{'layers'：[{'class_name'：'Dense'，
                        'config'：{'激活'：'relu'，
                                   'activity_regularizer'：沒有，
                                   'bias_constraint'：沒有，
                                   'bias_initializer'：{'class_name'：'Zeros'，
                                                        'config'：{'dtype'：'float32'}}，
                                   'bias_regularizer'：沒有，
                                   'dtype'：沒有，
                                   'kernel_constraint'：沒有，
                                   'kernel_initializer'：{'class_name'：'GlorotUniform'，
                                                          'config'：{'dtype'：'float32'，
                                                                     '種子'：無}}，
                                   'kernel_regularizer'：沒有，
                                   'name'：'dense_17'，
                                   '可訓練'：是的，
                                   '單位'：64，
                                   'use_bias'：True}}，
                       {'class_name'：'密集'，
                        'config'：{'激活'：'softmax'，
                                   'activity_regularizer'：沒有，
                                   'bias_constraint'：沒有，
                                   'bias_initializer'：{'class_name'：'Zeros'，
                                                        'config'：{'dtype'：'float32'}}，
                                   'bias_regularizer'：沒有，
                                   'dtype'：沒有，
                                   'kernel_constraint'：沒有，
                                   'kernel_initializer'：{'class_name'：'GlorotUniform'，
                                                          'config'：{'dtype'：'float32'，
                                                                     '種子'：無}}，
                                   'kernel_regularizer'：沒有，
                                   'name'：'dense_18'，
                                   '可訓練'：是的，
                                   '單位'：10，
                                   'use_bias'：True}}]，
            'name'：'sequential_3'}，
 'keras_version'：'2.1.6-tf'}
從JSON重新創建模型（新初始化）：

fresh_model = tf.keras.models.model_from_json(json_string)
將模型序列化為YAML格式要求pyyaml 在導入TensorFlow之前進行安裝：

yaml_string = model.to_yaml()
print(yaml_string)
後端：tensorflow
class_name：順序
配置：
  層：
  -  class_name：密集
    配置：
      激活：relu
      activity_regularizer：null
      bias_constraint：null
      bias_initializer：
        class_name：零
        config：{dtype：float32}
      bias_regularizer：null
      dtype：null
      kernel_constraint：null
      kernel_initializer：
        class_name：GlorotUniform
        config：{dtype：float32，seed：null}
      kernel_regularizer：null
      名稱：dense_17
      可訓練的：真的
      單位：64
      use_bias：true
  -  class_name：密集
    配置：
      激活：softmax
      activity_regularizer：null
      bias_constraint：null
      bias_initializer：
        class_name：零
        config：{dtype：float32}
      bias_regularizer：null
      dtype：null
      kernel_constraint：null
      kernel_initializer：
        class_name：GlorotUniform
        config：{dtype：float32，seed：null}
      kernel_regularizer：null
      名稱：dense_18
      可訓練的：真的
      單位：10
      use_bias：true
  name：sequential_3
keras_version：2.1.6-tf

從YAML重新創建模型：

fresh_model = tf.keras.models.model_from_yaml(yaml_string)
注意：子類模型不可序列化，因為它們的體系結構由call方法體中的Python代碼定義。
整個模型
整個模型可以保存到包含權重值，模型配置甚至優化器配置的文件中。這允許您檢查模型並稍後從完全相同的狀態恢復培訓 - 無需訪問原始代碼。

# Create a trivial model
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')
大紀元1/5
1000/1000 [==============================]  -  0s 261us /步 - 損失：11.5380  -  acc：0.0940
大紀元2/5
1000/1000 [==============================]  -  0s 53us /步 - 損失：11.5247  -  acc：0.1050
大紀元3/5
1000/1000 [==============================]  -  0s 66us /步 - 損失：11.5221  -  acc：0.1080
大紀元4/5
1000/1000 [==============================]  -  0s 65us /步 - 損失：11.5211  -  acc：0.1020
大紀元5/5
1000/1000 [==============================]  -  0s 55us /步 - 損失：11.5204  -  acc：0.1040
急切執行
急切執行是一個必要的編程環境，可以立即評估操作。這不是Keras所必需的，但是tf.keras對於檢查程序和調試是有用的，也是有用的。

所有tf.keras模型構建API都與急切執行兼容。雖然Sequential可以使用和功能API，但是熱切的執行特別有利於模型子類化和構建自定義層 - 需要您將前向傳遞作為代碼編寫的API（而不是通過組合現有層來創建模型的API）。

有關使用具有自定義訓練循環的Keras模型的示例，請參閱急切的執行指南tf.GradientTape。

分配
估計
該估計 API用於培訓模式，為分佈式環境。這針對行業用例，例如可以導出模型進行生產的大型數據集的分佈式培訓。

阿tf.keras.Model可與被訓練tf.estimatorAPI由模型轉換為tf.estimator.Estimator與對象 tf.keras.estimator.model_to_estimator。請參閱 從Keras模型創建估算器。

model = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)
信息：tensorflow：使用默認配置。
警告：tensorflow：使用臨時文件夾作為模型目錄：/ tmp / tmpvv9dwp02
INFO：tensorflow：使用提供的Keras模型。
信息：tensorflow：使用config：{'_ keep_checkpoint_max'：5，'_ evaluation_master'：''，'_ session_config'：allow_soft_placement：true
graph_options {
  rewrite_options {
    meta_optimizer_iterations：ONE
  }
}
，'_mo​​del_dir'：'/ tmp / tmpvv9dwp02'，'_ cluster_pec'：<tensorflow.python.training.server_lib.ClusterSpec對象位於0x7fb500688ac8>，'_ master'：''，'_ save_checkpoints_secs'：600，'_ log_step_count_steps'：100， '_keep_checkpoint_every_n_hours'：10000，'_ tf_random_seed'：無，'_ save_summary_steps'：100，'_ num_ps_replicas'：0，'_ is_chief'：是，'_ prototocol'：無，'_ _ train_distribute'：無，'_ eval_distribute'：無，'_ expperimental_distribute '：無，'_ num_worker_replicas'：1，'_ save_checkpoints_steps'：無，'_ global_id_in_cluster'：0，'_ task_id'：0，'_ service'：無，'_ task_type'：'worker'，'_ device_fn'：無}
注意：啟用預先執行以調試 Estimator輸入函數 和檢查數據。
多個GPU
tf.keras模型可以使用多個GPU運行 tf.contrib.distribute.DistributionStrategy。此API在多個GPU上提供分佈式培訓，幾乎不對現有代碼進行任何更改。

目前，tf.contrib.distribute.MirroredStrategy是唯一支持的分銷策略。MirroredStrategy在單台機器上使用all-reduce進行同步訓練的圖形內復制。要使用 DistributionStrategy與Keras，轉換tf.keras.Model為 tf.estimator.Estimator與tf.keras.estimator.model_to_estimator，則訓練估計

以下示例tf.keras.Model在一台計算機上分佈多個GPU。

首先，定義一個簡單的模型：

model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
_________________________________________________________________
圖層（類型）輸出形狀參數＃   
================================================== ===============
dense_23（密集）（無，16）176       
_________________________________________________________________
dense_24（密集）（無，1）17        
================================================== ===============
總參數：193
可訓練的參數：193
不可訓練的參數：0
_________________________________________________________________
定義輸入管道。所述input_fn返回一個tf.data.Dataset對象用來跨多個數據設備-與每個設備處理輸入批次的切片分配。

def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset
接下來，創建一個tf.estimator.RunConfig並train_distribute為tf.contrib.distribute.MirroredStrategy實例設置參數。創建時 MirroredStrategy，您可以指定設備列表或設置num_gpus 參數。默認使用所有可用的GPU，如下所示：

strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
INFO：tensorflow：使用分發策略初始化RunConfig。
INFO：tensorflow：不使用Distribute Coordinator。
將Keras模型轉換為tf.estimator.Estimator實例：

keras_estimator = tf.keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/tmp/model_dir')
INFO：tensorflow：使用提供的Keras模型。
信息：tensorflow：使用config：{'_ keep_checkpoint_max'：5，'_ evaluation_master'：''，'_ session_config'：allow_soft_placement：true
graph_options {
  rewrite_options {
    meta_optimizer_iterations：ONE
  }
}
，'_ model_dir'：'/ tmp / model_dir'，'_distribute_coordinator_mode'：無，'_ cluster_pec'：<tensorflow.python.training.server_lib.ClusterSpec對象位於0x7fb549ac6cf8>，'_ master'：''，'_ save_checkpoints_secs'：600， '_log_step_count_steps'：100，'_ keep_checkpoint_every_n_hours'：10000，'_ tf_random_seed'：無，'_ save_summary_steps'：100，'_ num_ps_replicas'：0，'_ is_chief'：是的，'_ prototocol'：無，'_ _ train_distribute'：<tensorflow.contrib .distribute.python.mirrored_strategy.MirroredStrategy對象位於0x7fb549ac6c50>，'_ eval_distribute'：無，'_ instperimental_distribute'：無，'_ num_worker_replicas'：1，'_ save_checkpoints_steps'：無，'_ global_id_in_cluster'：0，'_ task_id'：0，'_ service'：無，'_ task_type'：'worker'，'_ device_fn'：無}
最後，Estimator通過提供input_fn和steps 參數訓練實例：

keras_estimator.train(input_fn=input_fn, steps=10)
INFO：tensorflow：設備可用但分發策略不使用：/ device：XLA_CPU：0
警告：tensorflow：並非DistributionStrategy中的所有設備都對TensorFlow會話可見。
信息：tensorflow：調用model_fn。
信息：tensorflow：完成調用model_fn。
INFO：tensorflow：使用WarmStartSettings熱啟動：WarmStartSettings（ckpt_to_initialize_from ='/ tmp / model_dir / keras / keras_model.ckpt'，vars_to_warm_start ='。*'，var_name_to_vocab_info = {}，var_name_to_prev_var_name = {}）
INFO：tensorflow：熱啟動:('/ tmp/model_dir/keras/keras_model.ckpt'，）
INFO：tensorflow：熱啟動變量：dense_24 / bias; prev_var_name：不變
INFO：tensorflow：熱啟動變量：dense_23 / kernel; prev_var_name：不變
INFO：tensorflow：熱啟動變量：dense_23 / bias; prev_var_name：不變
INFO：tensorflow：熱啟動變量：dense_24 / kernel; prev_var_name：不變
信息：tensorflow：創建CheckpointSaverHook。
信息：tensorflow：圖表已完成。
信息：tensorflow：運行local_init_op。
信息：tensorflow：完成運行local_init_op。
INFO：tensorflow：將0的檢查點保存到/tmp/mo

```

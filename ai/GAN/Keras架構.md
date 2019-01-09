# Keras
```
Keras 是一個用 Python 編寫的高級神經網路 API
它能夠以 TensorFlow, CNTK, 或者 Theano 作為後端運行。

Keras 的開發重點是支持快速的實驗。能夠以最小的時延把你的想法轉換為實驗結果，是做好研究的關鍵。

如果你在以下情況下需要深度學習庫，請使用 Keras：
允許簡單而快速的原型設計（由於用戶友好，高度模組化，可擴展性）。
同時支援卷積神經網路和迴圈神經網路，以及兩者的組合。
在 CPU 和 GPU 上無縫運行。
```
# Keras參考書目
```
https://keras.io/zh/

https://detail.tmall.com/item.htm?spm=a230r.1.14.75.4a3e3dc1BgMDSW&id=582584961300&ns=1&abbucket=16
https://github.com/yanchao727/keras_book
```
# Keras 模型
```
在 Keras 中有兩類主要的模型：
[1]Sequential 順序模型 
[2]使用函數式 API 的 Model 類模型。

這些模型有許多共同的方法和屬性：
model.layers 是包含模型網路層的展平清單。
model.inputs 是模型輸入張量的清單。
model.outputs 是模型輸出張量的清單。
model.summary() 列印出模型概述資訊。 它是 utils.print_summary 的簡捷調用。

model.get_config() 返回包含模型配置資訊的字典。通過以下代碼，就可以根據這些配置資訊重新產生實體模型：
model.get_weights() 返回模型中所有權重張量的清單，類型為 Numpy 陣列。
model.set_weights(weights) 從 Numpy 陣列中為模型設置權重。清單中的陣列必須與 get_weights() 返回的權重具有相同的尺寸。

model.to_json() 以 JSON 字串的形式返回模型的表示。
  請注意，該表示不包括權重，僅包含結構。你可以通過以下方式從 JSON 字串重新產生實體同一模型（使用重新初始化的權重）：
model.to_yaml() 以 YAML 字串的形式返回模型的表示。請注意，該表示不包括權重，只包含結構。你可以通過以下代碼，從 YAML 字串中重新產生實體相同的模型（使用重新初始化的權重）：

model.save_weights(filepath) 將模型權重存儲為 HDF5 檔。
model.load_weights(filepath, by_name=False): 從 HDF5 檔（由 save_weights 創建）中載入權重。
預設情況下，模型的結構應該是不變的。 如果想將權重載入不同的模型（部分層相同）， 設置 by_name=True 來載入那些名字相同的層的權重。
注意：另請參閱如何安裝 HDF5 或 h5py 以保存 Keras 模型，在常見問題中瞭解如何安裝 h5py 的說明。
```
### Model 類繼承
```
除了這兩類模型之外，你還可以通過繼承 Model 類並在 call 方法中實現你自己的前向傳播，以創建你自己的完全定制化的模型，
（Model 類繼承 API 引入於 Keras 2.2.0）。
```
https://colab.research.google.com/drive/1I9zhxEbNvZ6_9iyKe524EoiMr_f5Nn5T

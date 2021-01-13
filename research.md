### 損失関数
損失関数 ⇒ 2乗和誤差, 交差エントロピー誤差 などが多い
損失 ⇒ 「目標」と「実際」の出力の語さ

参考文献：https://rightcode.co.jp/blog/information-technology/loss-function-neural-network-learning-theory

### NumPy
numpy.ndarray.reshape()
⇒二次元配列にしたりする構造変化のやつ

## Keras
### models.Sequential()
```
model = keras.Sequential([
    layers.Dense(2, activation="relu", name="layer1")
    layers.Dense(3, activation="relu", name="layer2")
    layers.Dense(4, name="layer3")
])

x = tf.ones((3,3))
y = model(x)
```
下のコードと等価になる。
```
layer1 = layers.Dense(2, activation="relu", name="layer1")
layer2 = layers.Dense(3, activation="relu", name="layer2")
layer3 = layers.Dense(4, name="layer3")

x = tf.ones((3,3))
y = layer3(layer2(layer1(x)))
```

### model.add(layers.Dense())
model.Sequential()の別の記述方法としてmodel.add()がある。
```
model = keras.Sequantial()
model.add(layers.Dense(2, activation="relu, name="layer1"))
model.add(layers.Dense(3, activation="relu, name="layer2"))
model.add(layers.Dense(4, name="layer3"))
```
<b><font color = "yellow">Sequentialモデル</font></b>を段階的に構築する場合は、現在の出力形状を含む、これまでのモデルの概要を表示できると非常に便利です。この場合に、<b><font color = "yellow">Inputオブジェクト</font></b>をモデルに渡してモデルを開始し、モデルが最初から入力形状を認識できるようにする必要があります。  
Inputオブジェクトはレイヤーではないため、model.layersの一部として表示されない。よって、簡単な代替補法として、<b><font color = "yellow">input_shape引数</font></b>を最初のレイヤーに渡すことで解決される。

### 重み
Kerasの全てのレイヤーは、最初は重みがない。
重みの形状は入力の形状に依存するため、入力で最初に呼び出されたときに重みを作成します。

Sequentialモデルも同様である。
model.weight():　重みの確認
model.summary(): モデルの構築内容確認
### loss
https://keras.io/ja/losses/  
categorical_crossentropy: 多クラス交差エントロピー  
sparese_categorical_crossentropy  
binary_crossentoropy: 二値交差エントロピー(BCE)  
mean_squared_error: 平均二乗誤差(MSE)  
mean_absolute_error: 平均絶対誤差(MAE)  
mean_absolute_percentage_error: 平均絶対誤差率  
mean_squared_logarithmic_error  
squared_hinge  
hinge  
categorical_hinge  
kullback_leibler_divergence  
poisson  


### model.compile(optimaizer, loss)
### callbacks.EarlyStopping(monitor, patience, mode)
monitor: Quantity to be monitored.  
patience: NUmber of epochs with no improvement after which training will be stopped.  
mode: One of {"auto", "min",  "max"}. In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode, it will stop when the wuantity monitored has stopped increasing; in auto mode, the directio is automatically inferred from the name of the monitored quantity.
### model.fit()
### model.evaluate()
### models.clone_model()
### models.pop()
レイヤーを削除するだけ


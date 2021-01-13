"""
AutoEncoderの実装
入力ユニット数N: 10 <= N <= 100
出力ユニット数N
中間層のユニット数M: 5 <= M <= 50

学習データ
ランダムに作ったNbitのデータ100個(0と1の発生確率はいずれも0.5)

テストデータ
同上

評価関数：出力と正解データの間の平均2乗誤差
活性化関数：シグモイド関数
"""

"""
(input: N-dimensional)
[Dense (M units, sigmoid activation)]
(output: N-dimensional)
"""
"""
参考文献
https://blog.amedama.jp/entry/keras-auto-encoder
https://www.tensorflow.org/tutorials/quickstart/beginner?hl=ja
https://qiita.com/omiita/items/1735c1d048fe5f611f80
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import callbacks

def ae(N:int, M:int):
    #データセット読み込み
    train_data = np.random.randint(0,2,(100,N))
    train_data = train_data.astype("float32")
    #print(train_data)
    #ラベル付けは一緒
    train_label = train_data
    #保存
    np.savetxt("train_data.txt", train_data, fmt="%.0f")

    #テストデータとして新しく乱数生成する必要はない？
    test_data = np.random.randint(0,2,(100,N))
    test_data = test_data.astype("float32")
    test_label = test_data
    #保存
    np.savetxt("test_data.txt", test_data, fmt="%.0f")
    
    #中間層で圧縮される次元数 = M
    encoding_dim = M

    #Flatten

    #Min-Max Normalization

    #中間層が一層だけのAutoEncoder
    model = models.Sequential()
    model.add(layers.Dense(encoding_dim, activation="relu", input_shape=(N,)))
    model.add(layers.Dense(N, activation="sigmoid"))

    #モデルの構造を確認する
    print(model.summary())
    
    model.compile(optimizer="adam", loss="mean_squared_error")

    fit_callbacs = [callbacks.EarlyStopping(monitor="loss", patience=5, mode="min")]

    #モデルを学習させる
    #epochs,batch_sizeは実験で変更させる
    model.fit(train_data, train_label, epochs=1000, batch_size=32,shuffle=True,callbacks=fit_callbacs)
    #テストデータの損失を確認しておく
    score = model.evaluate(test_data, test_label, verbose=2)
    #score = model.evaluate(train_data,train_label, verbose=2)
    print("test mean_squared_erorr:", score)

    """
    #学習済みのモデルを元に、次元圧縮だけするモデルを用意する
    encoder = models.clone_model(model)
    encoder.compile(optimizer="adam", loss="mean_squared_error")
    encoder.set_weights(model.get_weights())

    #最終段のレイヤーを取り除く
    encoder.pop()
    """

    #選び出したサンプルをAutoEncoderにかける
    test_pred = model.predict_proba(test_data,verbose=2)
    #test_pred = model.predict_proba(train_data,verbose=2)
    #四捨五入してintにする
    test_pred = np.round(test_pred)
    print(test_pred)
    np.savetxt("test_pred.txt", test_pred, fmt="%.0f")


    #精度
    precision = 0
    for i in range(100):
        if all(test_pred[i] == test_data[i]) == True:
            precision += 1
    
    print("precision: "+ str(precision))

    #次元圧縮だけする場合



if __name__ == "__main__":
    print("AutoEncoder...")
    N = int(input("入出力ユニット数： N = "))
    M = int(input("中間層のユニット数： M = "))

    ae(N, M)
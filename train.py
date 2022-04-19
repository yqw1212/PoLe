from operator import mod
from random import random
from xml.etree.ElementInclude import XINCLUDE_FALLBACK
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.datasets import cifar10
from keras.datasets import mnist
import matplotlib.pyplot as plt
import keras
import numpy as np
import random

				
def vgg16(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model


def leNet5(input_shape):
    model = Sequential([
        Conv2D(filters=6,kernel_size=(5,5),padding="valid",activation="tanh",input_shape=input_shape),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(filters=16,kernel_size=(5,5),padding="valid",activation="tanh"),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(120,activation="tanh"),
        Dense(84,activation="tanh"),
        Dense(10,activation="softmax")
    ])
    
    return model


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

# (x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print(len(x_train))  # 50000
# print(len(x_test))  # 10000

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)

# tmp = []
# for x in x_train:
#     m = []
#     for k in range(3):
#         mat = [[0 for i in range(32)] for _ in range(32)]
#         # 32*32*3
#         for i in range(32):
#             for j in range(32):
#                 mat[i][j] = x[i][j][k]
#         m.append(mat)
#     tmp.append(m)
# tmp = np.array(tmp)
# print(tmp.shape)
    
# x_train = tmp

# tmp = []
# for x in x_test:
#     m = []
#     for k in range(3):
#         mat = [[0 for i in range(32)] for _ in range(32)]
#         # 32*32*3
#         for i in range(32):
#             for j in range(32):
#                 mat[i][j] = x[i][j][k]
#         m.append(mat)
#     tmp.append(m)
# tmp = np.array(tmp)
# print(tmp.shape)
    
# x_test = tmp

# # 生成Z
# Z = [ [ [ random.random() for j in range(32)] for i in range(32)] for _ in range(3)]
# Z = np.array(Z)

# # 训练集×Z
# # 遍历训练集
# for i in range(len(x_train)):
#     for j in range(3):
#         # print(x_train[i][j].shape)
#         x_train[i][j] = np.matmul(x_train[i][j], Z[j])


# # 测试集×Z
# # 遍历测试集
# for i in range(len(x_test)):
#     for j in range(3):
#         x_test[i][j] = np.matmul(x_test[i][j], Z[j])


# # 还原数据集格式
# # 3*32*32 -> 32*32*3

# tmp = []
# for x in x_train:
#     m = [[[] for i in range(32)] for _ in range(32)]
#     for i in range(32):
#         for j in range(32):
#             mat = []
#             for k in range(3):
#                 mat.append(x[k][i][j])
#             m[i][j].append(mat)
#     tmp.append(m)
# tmp = np.array(tmp)
# print(tmp.shape)
# x_train = tmp


# tmp = []
# for x in x_test:
#     m = [[[] for i in range(32)] for _ in range(32)]
#     for i in range(32):
#         for j in range(32):
#             mat = []
#             for k in range(3):
#                 mat.append(x[k][i][j])
#             m[i][j].append(mat)
#     tmp.append(m)
# tmp = np.array(tmp)
# print(tmp.shape)
# x_test = tmp


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = x_train / 255.0
x_test = x_test / 255.0

input_shape = (28, 28)

# model = vgg16()
model = leNet5(input_shape)

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# one-hot独热映射
y_label_train = keras.utils.to_categorical(y_train, 10)
y_label_test = keras.utils.to_categorical(y_test, 10)

model.fit(x_train, y_label_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_label_test)
print('Test acc: %f' % test_acc)

model.save("./vgg16_model")
print("模型保存成功")

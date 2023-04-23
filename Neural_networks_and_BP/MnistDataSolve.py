#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import struct
from bp import *
from datetime import datetime
# 数据加载器基类
import numpy as np

#全连接层实现类
class FullConnectedLayer(object):
    def __init__(self,input_size,output_size,activator):
        '''
        :param input_size: 本层输入向量的维度
        :param output_size: 本层输出向量的维度
        :param activator: 激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # 权重数组W 以后要更新的就是这玩意
        self.W = np.random.uniform(-0.1,0.1,(output_size, input_size)) # random.uniform (x, y) 方法将随机生成一个实数，它在 [x,y] 范围内
        #偏置值
        self.b = np.zeros((output_size,1))
        #输出向量
        self.output = np.zeros((output_size,1))

    def forward(self,input_array):
        '''
        向前计算
        :param input_array: 输入向量
        :return:
        '''
        self.input = input_array
        self.output = self.activator.forward(self,np.dot(self.W, input_array) + self.b)

    def backward(self,delta_array):
        '''
        反向计算w和b的梯度
        :param delta_array:从上一次传过来的误差项
        :return:
        '''
        self.delta = self.activator.backward(self,self.input) *np.dot(self.W.T,delta_array) # W.T是矩阵转置
        self.W_grad = np.dot(delta_array,self.input.T)
        self.b_grad = delta_array

    def update(self,learning_rate):
        '''
        用梯度下降法更新权重
        :param learning_rate:
        :return:
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

class SigmoidActivator(object):
    def forward(self,weighted_input):
        return 1.0/ (1.0 + np.exp(-weighted_input))
    def backward(self,output):
        return output * (1 - output) # sigmoid函数求导，从而得到梯度，用来做反向传播

class Network(object):
    def __init__(self,layers):
        # 注意layers 和 self.layers的区别
        # 这里就是将layers 装入到 self.layers当中，并进行全连接的操作
        self.layers = []
        for i in range(len(layers) -1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i],layers[i+1],
                    SigmoidActivator
                )
            )

    def predict(self,sample):
        '''
        使用神经网络实现预测 ： 正向输出
        :param sample: 输入样本
        :return:
        '''
        output = sample
        # 每次都向前传递一层的参数，传递的方式就是 矩阵乘向量
        for layer in self.layers:
            layer.forward(output)
            output = layer.output
        return output

    def train(self,labels,data_set,rate,epoch):
        '''
        训练函数
        :param labels: 样本标签
        :param data_set: 数据集
        :param rate: 学习速率
        :param epoch: 训练轮数
        :return:
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(
                    labels[d],
                    data_set[d],
                    rate
                )

    def train_one_sample(self,label,sample,rate):
        self.predict(sample) # 前向计算得到输出
        self.calc_gradient(label) # 计算梯度
        self.update_weight(rate) # 更新权重

    def calc_gradient(self,label):
        label = np.atleast_2d(label).T # 将纯向量转成 矩阵化向量，方便计算 即 [1,2,3,4] -> [[1],[2],[3],[4]]
        # self.layers[-1].output 是最后一层的输出向量
        delta = self.layers[-1].activator.backward(SigmoidActivator,self.layers[-1].output)*(label - self.layers[-1].output)
        # 式8计算每层误差
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self,rate):
        for layer in self.layers:
            layer.update(rate)


class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        '''
        self.path = path
        self.count = count
    def get_file_content(self):
        '''
        读取文件内容
        '''
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        return content
    def to_int(self, byte):
        '''
        将unsigned byte字符转换为整数
        '''
        return struct.unpack('B', byte)[0]
# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数，从文件中获取图像
        '''
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(self.to_int(bytes([content[start + i * 28 + j]])))
        return picture
    def get_one_sample(self, picture):
        '''
        内部函数，将图像转化为样本的输入向量
        二维的图像转换成一维的向量
        '''
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample
    def load(self):
        '''
        加载数据文件，获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set

class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件，获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels
    def norm(self, label):
        '''
        内部函数，将一个值转换为10维标签向量
        '''
        label_vec = []
        label_value = self.to_int(bytes([label]))
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec
def get_training_data_set():
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('./MNIST_data/train-images.idx3-ubyte', 60000) #加载训练集
    label_loader = LabelLoader('./MNIST_data/train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()
def get_test_data_set():
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('./MNIST_data/t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('./MNIST_data/t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

# 网络的输出是一个10维向量，这个向量第个(从0开始编号)元素的值最大，那么就是网络的识别结果。
def get_result(vec):
    # 我得到的其实是可能性最大的那个东西
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(test_labels[i]) # label集合
        predict = get_result(network.predict(test_data_set[i])) # 根据模型预测出来的
        if label != predict:
            error += 1
    return float(error) / float(total) # 计算错误率

def train_and_evaluate():
    last_error_ratio = 1.0 # 错误率
    epoch = 0 # 一个epoch , 表示： 所有的数据送入网络中， 完成了一次前向计算 + 反向传播的过程。
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()
    network = Network([784, 300, 10]) # 输入层 784个节点 hidden_layer 300个节点 output_layer 10个节点
    while True:
        epoch += 1
        network.train(train_labels, train_data_set, 0.3, 1) # 学习速率为0.1
        print ('%s epoch %d finished' % (datetime.now(), epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(network, test_data_set, test_labels)
            print ('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio
if __name__ == '__main__':
    train_and_evaluate()
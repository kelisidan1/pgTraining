import numpy as np
'''
    实现不了啊。。。。。。。。。
'''



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

        # 权重数组W
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
        self.output = self.activator.forward(
            np.dot(self.W, input_array) + self.b)

    def backward(self,delta_array):
        '''
        反向计算w和b的梯度
        :param delta_array:从上一次传过来的误差项
        :return:
        '''
        self.delta = self.activator.backward(self.input) *np.dot(self.W.T,delta_array) # W.T是矩阵转置
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
        使用神经网络实现预测
        :param sample: 输入样本
        :return:
        '''
        output = sample
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
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def calc_gradient(self,label):
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        )*(label - self.layers[-1].output)

        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self,rate):
        for layer in self.layers:
            layer.update(rate)




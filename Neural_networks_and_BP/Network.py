# Network 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。
from Neural_networks_and_BP.Connection import Connection
from Neural_networks_and_BP.Connections import Connections
from Neural_networks_and_BP.Layer import Layer


class Network(object):
    def __init__(self,layers):
        '''
        初始化一个全连接神经网络
        :param layers: 二维数组，面熟神经网络每层节点数量
        '''
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count - 1):
            connections = [
                Connection(upstream_node,downstream_node)
                for upstream_node in self.layers[layer].nodes
                for downstream_node in self.layers[layer + 1].nodes[:-1]
            ]

            for conn in connections:
                self.connections.add_connections(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
        '''
        训练神经网络
        :param labels: 每一个样本的标签
        :param data_set: 二维数组
        :param rate:
        :param iteration:
        :return:
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
        '''
        内部函数，用一个样本训练网络
        :param label:
        :param sample:
        :param rate:
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self,label):
        '''
        内部函数，计算每个节点的δ
        这个函数的实现分成了两个部分
        1. output Layer 的计算
        2. hidden Layer 的计算
        这俩的δ计算公式是不一样的
        还有一点就是，这个更新算法使用的是反向传播算法，从后往前进行迭代W vector的值
        :param label:
        :return:
        '''
        output_nodes = self.layers[-1].nodes # 最后一层的节点
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self,rate):
        '''
        内部函数，更新每个连接权重
        :param rate:
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        '''
        内部函数，计算每个连接的梯度
        :return:
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self,label,sample):
        '''
        获得网络再一个样本下，每个连接上的梯度
        :param label: 样本标签
        :param sample: 样本输入
        :return:
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self,sample):
        '''
        根据输入的样本预测输出值
        :param sample:输入向量
        :return:
        '''
        self.layers[0].set_output(sample)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node : node.output,self.layers[-1].nodes[-1])

    def dump(self):
        '''
        打印网络信息
        :return:
        '''
        for layer in self.layers:
            layer.dump()
            

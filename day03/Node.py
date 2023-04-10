# 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
from functools import reduce
import Utils

class Node(object):
    def __init__(self,layer_index,node_index):
        '''
        :param layer_index: 节点所属的层的编号
        :param node_index: 节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self,output):
        self.output = output

    def append_downstream_connection(self,conn):
        '''
        :param conn: 添加到下游节点的连接
        :return: void
        '''
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(conn)

    def calc_output(self):
        # 计算节点的输出 也就是 a4 = w1*x1+w2*x2+w3*x3
        output = reduce(lambda ret,conn:ret + conn.upstream_node.output * conn.weight,self.upstream,0)
        self.output = Utils.sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delda = reduce(
            lambda ret,conn : ret + conn.downstream_node.delta * conn.weight,
            self.downstream,
            0.0
        )
        self.delta = self.output * (1 - self.output) * downstream_delda

    def calc_output_layer_delta(self,label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


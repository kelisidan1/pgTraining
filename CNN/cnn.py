import numpy as np

#工具函数
def element_wise_op(array, op):
    '''
    对array当中的每一个参数都使用activator
    作用：对vector 或者 matrix都可以用这个东西
    :param array:
    :param op:
    :return:
    '''
    for i in np.nditer(array,op_flags=['readwrite']):
        i[...] = op(i)

def conv(input_array,kernel_array,output_array,stride,bias):
    '''
    计算卷积
    :param input_array:
    :param kernel_array:
    :param output_array:
    :param stride:
    :param bias:
    :return:
    '''
    channel_number = input_array.mdim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array,i,j,kernel_width,
                    kernel_height,stride) * kernel_array
            ).sum() + bias

def padding(input_array,zp):
    '''
    为数组增加Zero padding
    :param input_array:
    :param zp:
    :return:
    '''
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:   #ndim 是数组的维度
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            padded_array = np.zeros((
                input_depth,
                input_height + 2*zp,
                input_width + 2*zp
            ))
            padded_array[:,
                zp:zp + input_height,
                zp:zp +input_width
            ] = input_array
            return padded_array
        elif input_array.ndim==2:
            input_width = input_array.shape[1]
            input_height = input_array.shpe[0]
            padded_array = np.zeros((
                input_height + 2*zp,
                input_width + 2*zp
            ))
            padded_array[
                zp:zp+input_height+2*zp,
                zp:zp + input_width
            ] = input_array
            return padded_array


# 卷积层初始化
class ConvLayer(object):
    def __init__(self,input_width,input_height,
                 channel_number,filter_width,
                 filter_height,filter_number,
                 zero_padding,stride,activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(
            self.input_width,
            filter_width,
            zero_padding,
            stride
        )
        self.output_height = ConvLayer.calculate_output_size(
            self.input_height,
            filter_height,
            zero_padding,
            stride
        )
        self.output_array = np.zeros((self.filter_number,self.output_height,self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width,filter_height,self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2*zero_padding) / stride + 1

    def forward(self,input_array):
        '''
        计算卷积层的输出
        输出的结果保存在self.output_array
        :param input_array:
        :return:
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array,self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(
                self.padded_input_array,
                filter.get_weights(),
                self.output_array[f],
                self.stride,
                filter.get_bias()
            ) # 卷积操作
        element_wise_op(self.output_array,self.activator.forward)

    def bp_sensitivity_map(self,sensitivity_array,activator): #公式
        '''
        计算传递到上一层的sensitivity map
        :param sensitivity_array: 本层的sensitivity map
        :param activator: 上一层的激活函数
        :return:
        '''
        # 处理卷积步长，对原始的sensitivity map进行扩展
        expanded_array = self.expend_sensitivity_map(sensitivity_array)

        #full卷积，用sensitivity map 进行zero padding
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        #初始化 delta_array，用于保存上一层的sensitivity map
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter 权重反转180度
            flipped_weights = np.array(map(
                lambda i:np.rot90(i,2), # 将矩阵img逆时针旋转2*90° = 180°
                filter.get_weights()
            ))
            #计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f],flipped_weights[d],delta_array[d],1,0)
            self.delta_array += delta_array
          # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op((derivative_array,activator.backward))
        self.delta_array *= derivative_array

    def enpand_sensitivity_map(self,sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 确定stride为1的时候 sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding +1)
        # 构造新的sensitivity_map
        expand_array = np.zeros((depth,expanded_height,expanded_width))
        # 从原始sensitivity map拷贝误差值
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:,i_pos,j_pos] = sensitivity_array[:i,j]

        return expand_array

    def create_delta_array(self):
        return np.zeros((self.channel_number,self.input_height,self.input_width))

    def bp_gradient(self,sensitivity_array):
        # 处理卷积步长，对原始的sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],expanded_array[f],filter.weights_grad[d],1,0)

            filter.bias_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)


class Filter(object):
    def __init__(self,width,height,depth):
        # numpy.random.uniform(low,high,size) [low,high)中间，取出来size个随机数
        # size=(m,n,k),时则输出m*n*k个样本，缺省时输出1个值。
        self.weights = np.random.uniform(-1e-4,1e-4,(depth,height,width))
        self.bias = 0
        self.weight_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))
    def get_weights(self):
        return self.weights
    def get_bias(self):
        return self.bias
    def update(self,learning_rate):
        # 梯度下降法更新权重
        self.weights -= learning_rate * self.weight_grad
        self.bias -= learning_rate * self.bias_grad

class ReluActivator(object):
    def forward(self,weighted_input):
        return max(0,weighted_input)
    def backward(self,output):
        return 1 if output > 0 else 0


class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = (input_width -
                             filter_width) / self.stride + 1
        self.output_height = (input_height -
                              filter_height) / self.stride + 1
        self.output_array = np.zeros((self.channel_number,
                                      self.output_height, self.output_width))

    def forward(self,input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d,i,j] = (
                        get_patch(
                            input_array[d],i,j,
                            self.filter_width,
                            self.filter_height,
                            self.stride
                        ).max()
                    )

    def backward(self,input_array,sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    k, l = get_max_index(patch_array)
                    self.delta_array[d,
                                     i * self.stride + k,
                                     j * self.stride + l] = \
                        sensitivity_array[d, i, j]





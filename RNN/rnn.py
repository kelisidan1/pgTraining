import numpy as np
from CNN.cnn import ReluActivator, IdentityActivator, element_wise_op

class RecurrentLayer(object):
    def __init__(self,input_width,state_width,activator,learning_rate):
        self.input_width = input_width
        self.state_width = state_width
        self.activator = activator
        self.learning_rate = learning_rate
        self.times = 0  # 当前时刻初始化为t0
        self.state_list = []  # 保存各个时刻的state
        self.state_list.append(np.zeros((state_width,1))) # 初始化s0
        self.U = np.random.uniform(-1e-4, 1e-4,(state_width, input_width))  # 初始化U
        self.W = np.random.uniform(-1e-4, 1e-4,(state_width, state_width))  # 初始化W

    def forward(self,input_array):
        '''
        前向计算
        :param input_array:
        :return:
        '''
        self.times += 1
        # dot是矩阵乘法运算，也可以用来乘向量
        state = (np.dot(self.U,input_array) + np.dot(self.W,self.state_list[-1])) # 先权值+求和
        element_wise_op(state,self.activator.forward) # 后经过激活函数，从而完整地经过一个“层”
        self.state_list.append(state)

    def backward(self,sensitivity_array,activator):
        '''
        实现BPTT算法
        :param sensitivity_array:
        :param activator:
        :return:
        '''
        self.calc_delta(sensitivity_array,activator) # 计算delta值
        self.calc_gradient() # 计算梯度，BP

    def calc_delta(self, sensitivity_array, activator):
        self.delta_list = [] # 保存各个时刻的误差项
        for i in range(self.times):
            self.delta_list.append(np.zeros(self.state_width,1))
        self.delta_list.append(sensitivity_array)
        # 迭代计算每个时刻的误差项
        for k in range(self.times-1,0,-1):
            self.calc_delta_k(k,activator)

    def calc_delta_k(self,k,activator):
        '''
        根据k+1时刻的delta计算k时刻的delta
        :param k:
        :param activator:
        :return:
        '''
        state = self.state_list[k+1].copy()
        element_wise_op(self.state_list[k+1],activator.backward)
        self.delta_list[k] = np.dot(
            np.dot(self.delta_list[k+1].T,self.W),
            np.diag(state[:,0].T)
        )

    def calc_gradient(self):
        self.gradient_list = [] # 保存各个时刻的权重梯度
        for t in range(self.times + 1):
            self.gradient_list.append(np.zeros(self.state_width,self.state_width))

        for t in range(self.times,0,-1):
            self.calc_gradient_t(t)

        #实际的梯度是各个时刻的梯度和
        self.gradient = reduce(
            lambda a,b:a+b,self.gradient_list,
            self.gradient_list[0]
        )

    def calc_gradient(self,t):
        '''
        计算每个时刻t权重的梯度
        :param t:
        :return:
        '''
        gradient = np.dot(self.delta_list[t],self.state_list[t-1].T)
        self.gradient_list[t] = gradient

    def update(self):
        '''
        梯度下降法，更新权重
        :return:
        '''
        self.W -= self.learning_rate * self.gradient

    def reset_state(self):
        self.times = 0
        self.state_list = []
        self.state_list.append(np.zeros((self.state_width,1)))

    def gradient_check(self):
        '''
        梯度检查
        :return:
        '''
        error_function = lambda o : o.sum()
        rl = RecurrentLayer(3,2,IdentityActivator(),1e-3)

        x,d = data_set()
        rl.forward(x[0])
        rl.forward(x[1])
        sensitivity_array = np.ones(rl.state_list[-1].shape,dtype=np.float64)
        rl.backward(sensitivity_array,IdentityActivator())

        #检查梯度
        epsilon = 10e-4
        for i in range(rl.W.shape[0]):
            for j in range(rl.W.shape[1]):
                rl.W[i, j] += epsilon
                rl.reset_state()
                rl.forward(x[0])
                rl.forward(x[1])
                err1 = error_function(rl.state_list[-1])
                rl.W[i, j] -= 2 * epsilon
                rl.reset_state()
                rl.forward(x[0])
                rl.forward(x[1])
                err2 = error_function(rl.state_list[-1])
                expect_grad = (err1 - err2) / (2 * epsilon)
                rl.W[i, j] += epsilon
                print('weights(%d,%d): expected - actural %f - %f' % (
                    i, j, expect_grad, rl.gradient[i, j]))



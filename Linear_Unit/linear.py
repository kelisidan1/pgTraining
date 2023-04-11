from perceptron.perceptron import Perceptron
'''
    这个项目是 有监督学习
    通过公司当中几个人的收入
    预测一下公司当中其他人的收入
'''


#继承Perceptron
f = lambda x: x
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    #虚构的数据
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels

def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs,labels = get_training_dataset()
    # 10轮迭代，学习速率为0.01
    lu.train(input_vecs,labels,10,0.01)
    return lu

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))

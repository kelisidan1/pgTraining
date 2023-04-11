from functools import reduce


class Perceptron(object):
    # activator是激活函数
    def __init__(self,input_num,activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        # 得到 weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.bias = 0.0

    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (list(self.weights) ,self.bias)

    def predict(self,input_vec):
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda xw: xw[0] * xw[1],
                       zip(input_vec, self.weights))
                   , 0.0) + self.bias)


    def train(self,input_vecs,labels,iteration,rate):
        #循环iteration这么多次，每一次都是对模型的一次更新
        for i in range(iteration):
            self._one_iteration(input_vecs,labels,rate)

    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs,labels)
        for(input_vecs,label) in samples:
            output = self.predict(input_vecs)
            self._update_weights(input_vecs,output,label,rate)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map(
            lambda xw:xw[1]+rate*delta*xw[0],
            zip(input_vec,self.weights)
        ))
        self.bias += rate*delta

def f(x):
    return 1 if x>0 else 0

def get_training_dataset():
    input_vecs = [[1,1], [0,0], [1,0], [0,1]]
    labels = [1, 0, 0, 0]
    return input_vecs, labels

def train_and_perceptron():
    #创建感知器，输入参数个数为2，激活函数为f
    p = Perceptron(2,f)
    input_vecs,labels = get_training_dataset()
    p.train(input_vecs,labels,10,0.1)
    return p

if __name__ == '__main__':
    and_perception = train_and_perceptron()
    print(and_perception)
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))


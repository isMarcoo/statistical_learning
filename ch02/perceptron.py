# Author: Marco
# Created on: 2024-11-08
# Description: 感知机算法的原始形式的实现

import numpy as np

class Perceptron:
    def __init__(self, dim):
        self.dim = dim
    
    def initialize(self, w=None, b=None):
        '''
        w: shape [1, dim]
        b: a number
        '''
        if w and b:
            self.w = w
            self.b = b
        else:
            self.w = np.zeros((1,self.dim))
            self.b = 0

    def train(self, X, Y, eta=0.1):
        '''
        x: shape [dim, n_sample]
        y: shape [1, n_sample]
        '''
        flag = True
        n_sample = X.shape[1]
        while flag:
            for i in range(n_sample):
                x = X[:, i]
                y = Y[:, i]
                result = self.w @ x + self.b
                if i == n_sample-1:
                    flag = False
                if result*y <= 0:
                    self.w = self.w + eta * y * x
                    self.b = self.b + eta * y
                    flag = True

    def predict(self, x):
        y = self.w @ x + self.b
        y = self.sign(y)
        return y

    def sign(self, num):
        if num > 0:
            return 1
        else:
            return -1
        
if __name__ == "__main__":
    '''
    x shape: [dim, n_sample]
    y shape: [1, n_sample]
    数据为书上的例子
    '''
    x = np.array([
        [3, 3],
        [4, 3],
        [1, 1]
    ])
    y = np.array([1,1,-1])
    x = x.transpose()
    y = y.reshape(1, 3)

    perceptron = Perceptron(x.shape[0])
    perceptron.initialize()
    perceptron.train(x, y, 1)
    print(perceptron.w, perceptron.b)
# Author: Marco
# Created on: 2024-11-14
# Description: 实现朴素贝叶斯算法

import numpy as np

class NaiveBayes:
    def __init__(self):
        '''
        self.conditional:用于存放条件分布
        self.dim:用于存放特征维数
        self.labels:用于存放标签
        self.prior:用于存放先验
        '''
        self.conditional = {}
        self.dim =None
        self.labels = None
        self.prior = None

    def train(self, X, Y):
        '''
        计算先验以及条件分布
        '''
        n_sample = X.shape[0]
        self.dim = X.shape[1]

        self.labels, counts = np.unique(Y, return_counts=True)
        self.prior = counts / n_sample

        # 对于每个类别计算
        for label in self.labels:
            self.conditional[label] = {}
            # 对该类别计数
            total = np.count_nonzero(Y==label)
            indexs = np.where(Y==label)
            # 对于每个特征
            for i in range(self.dim):
                self.conditional[label][i] = {}
                # 计算该类别下的同一特征不同特征类别的条件概率分布
                feature_labels, count = np.unique(X[:, i][indexs], return_counts=True)
                result = count / total
                # 对于每个特征的每个维度
                for feature_label in feature_labels:
                    k = np.where(feature_labels == feature_label)
                    # 将其条件概率分布存到字典
                    self.conditional[label][i][feature_label] = result[k]
                    print(f"label:{label}, feature:{i}, feature_label:{feature_label}, result:{result[k]}")

    def predict(self, x):
        '''
        预测新的数据
        '''
        result = np.zeros(self.labels.shape[0])
        for label in self.labels:
            index = np.where(self.labels == label)
            prob = self.prior[index]
            for i in range(self.dim):
                prob *= self.conditional[label][i][x[i]]
            
            result[index] = prob

        index = np.argmax(result)
        print(f"该数据的标签为{self.labels[index]}")

if __name__ == '__main__':
    X = np.array([
        [1,1],
        [1,2],
        [1,2],
        [1,1],
        [1,1],
        [2,1],
        [2,2],
        [2,2],
        [2,3],
        [2,3],
        [3,3],
        [3,2],
        [3,2],
        [3,3],
        [3,3]
    ])
    Y = np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])

    bayes = NaiveBayes()
    bayes.train(X, Y)

    x = np.array([2,1])
    bayes.predict(x)
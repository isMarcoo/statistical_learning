# Author: Marco
# Created on: 2024-11-27
# Description: ID3 and C4.5, 默认是ID3，如果需要更换成C4.5，请将fit函数中所有用到information_gain的地方替换成information_gain_rate函数

import numpy as np

def entropy(P):
    '''
    输入P为一个分布，是一个numpy中的向量，如：[0.1,0.3,0.6]
    '''
    entropy = 0
    for item in P:
        if item == 0:
            cal = 0
        else:
            cal = item*np.log2(item)
    
        entropy -= cal
    return entropy

def empirical_entropy(D, A):
    '''
    输入为一组数据，需要计算其经验分布，并计算其指定特征的经验熵
    形状为[n_sample, feature]
    当A=-1时表明是标签分布的经验熵
    '''
    unique, counts = np.unique(D[:, -1], return_counts=True)
    empirical_distribution = counts / D.shape[0]
    emp_entropy = entropy(empirical_distribution)
    return emp_entropy

def empirical_conditional_entropy(D, A):
    '''
    D: 数据
    A: 某一特征
    '''
    A_column = D[:, A]
    unique = np.unique(A_column)
    grouped_data = [D[A_column==key] for key in unique]

    emp_cond_entropy = 0
    for samples in grouped_data:
        if samples.size == 0:
            print("空")
        else:
            emp_cond_entropy += samples.shape[0] / D.shape[0] * empirical_entropy(samples, -1)

    return emp_cond_entropy


def information_gain(D, A):
    '''
    计算信息增益
    '''
    return empirical_entropy(D, -1) - empirical_conditional_entropy(D, A)

def information_gain_rate(D, A):
    '''
    计算信息增益比
    '''
    return information_gain(D, A) / empirical_entropy(D, A)

class Node():
    def __init__(self, feature=None, label=None):
        self.feature = feature
        self.label = label
        self.childs = {}

class DesicianTree():
    def __init__(self):
        self.root = None
        self.num_leaf = 0

    def fit(self, D, A, e):
        '''
        D: 用于构建的数据集
        A: 用于构建的特征集
        '''
        # 如果所有实例属于同一类，则返回单节点树
        unique, counts = np.unique(D[:, -1], return_counts=True)
        index_max_count = np.argmax(counts)
        if len(unique) == 1:
            root = Node(label=unique[0])
            self.num_leaf += 1
            return root
        
        # 如果特征集为空，则选择数量最多的标签作为叶子节点的标签
        if len(A) == 0:
            root = Node(label=unique[index_max_count])
            self.num_leaf += 1
            return root

        # 选择最大信息增益的特征
        information_gains = np.zeros(len(A))
        for feature in A:
            information_gains[A.index(feature)] = information_gain(D, feature)
        index = np.argmax(information_gains)

        # 若信息增益小于阈值，则简化为单节点树
        if information_gains[index] < e:
            root = Node(label=unique[index_max_count])
            return root
        
        selected_feature = A[index]
        selected_column = D[:, selected_feature]
        root = Node(feature=selected_feature, label=unique[index_max_count])

        unique, counts = np.unique(selected_column, return_counts=True)
        grouped_data = [(key, D[selected_column==key]) for key in unique]
        A.remove(selected_feature)
        for (key, data) in grouped_data:
            root.childs[key] = self.fit(data, A, e)

        self.root = root
        return root
    
    def printTree(self, tree):
        '''
        先序遍历
        '''
        print(tree.feature)
        if tree.childs:
            for key, child in tree.childs.items():
                self.printTree(child)

    def predict(self, tree, x):
        '''
        预测
        '''
        featureLabel = x[tree.feature]
        if not tree.childs:
            label = tree.label
        else:
            label = self.predict(tree.childs[featureLabel], x)
        
        return label
    
    def prune(self, tree, D, alpha):
        '''
        根据损失函数后剪枝
        '''
        emp_entropy = empirical_entropy(D, -1)
        selected_column = D[:, tree.feature]
        child_emp_entropy = 0
        if tree.childs:
            for key, child in tree.childs.items():
                child_emp_entropy += self.prune(child, D[selected_column==key], alpha)
                
            # 一旦经验熵的增大小于复杂度的减小，则可以剪枝 
            if emp_entropy - child_emp_entropy <= alpha * (len(tree.childs) - 1):
                # todo 裁剪
                unique, counts = np.unique(selected_column, return_counts=True)
                tree.label = unique[np.argmax(counts)]
                tree.childs = []
                self.root = tree
                return emp_entropy
            else:
                # 若不需要剪枝，则该节点的父节点考虑损失时应该用该节点的子节点返回的数值
                return child_emp_entropy

        self.root = tree
        return emp_entropy
    

if __name__ == '__main__':
    data = np.array([
        [0,0,0,0,0],
        [0,0,0,1,0],
        [0,1,0,1,1],
        [0,1,1,0,1],
        [0,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,1,0],
        [1,1,1,1,1],
        [1,0,1,2,1],
        [1,0,1,2,1],
        [2,0,1,2,1],
        [2,0,1,1,1],
        [2,1,0,1,1],
        [2,1,0,2,1],
        [2,0,0,0,0]
    ])

    desicianTree = DesicianTree()
    tree = desicianTree.fit(data, [0,1,2,3], 0.001)
    desicianTree.printTree(tree)
    desicianTree.prune(tree, data, 1)
    desicianTree.printTree(desicianTree.root)
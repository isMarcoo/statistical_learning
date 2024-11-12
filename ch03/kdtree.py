# Author: Marco
# Created on: 2024-11-12
# Description: Describe the purpose of this file here

import numpy as np

class KDNode:
    def __init__(self, value, split):
        self.value = value
        self.split = split
        self.left = None
        self.right = None

class KDTree:
    def __init__(self):
        self.root = None

    def createTree(self, X, split):
        '''
        创建KD树
        '''
        n_sample = X.shape[0]
        dim = X.shape[1]
        mid = n_sample // 2

        sortedIndex = np.argsort(X[:, split])
        sort_X = X[sortedIndex]
        root = KDNode(sort_X[mid], split)
        
        if np.any(sort_X[0:mid]):
            root.left = self.createTree(sort_X[0:mid], (split+1)%dim)
        else:
            root.left = None
        
        if np.any(sort_X[mid+1:]):
            root.right = self.createTree(sort_X[mid+1:], (split+1)%dim)
        else:
            root.right = None

        return root
    
    def printTree(self, node):
        if node is not None:
            print(node.value)
            self.printTree(node.left)
            self.printTree(node.right)


if __name__ == '__main__':
    X = np.array([
        [2,3],
        [5,4],
        [9,6],
        [4,7],
        [8,1],
        [7,2]
    ])
    kdtree = KDTree()
    kdtree.root = kdtree.createTree(X, 0)
    kdtree.printTree(kdtree.root)
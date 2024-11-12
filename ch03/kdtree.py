# Author: Marco
# Created on: 2024-11-12
# Description: kd树的最近邻搜索算法

import numpy as np

class KDNode:
    def __init__(self, value, split):
        '''
        value:用于存储分割节点的值
        split:用于存储分割的维度
        left:左子树
        right:右子树
        isvisit:用于判断是否被访问（用于判断另一未访问节点）
        '''
        self.value = value
        self.split = split
        self.left = None
        self.right = None
        self.isvisit = None

class KDTree:
    def __init__(self):
        '''
        root:根节点
        nearest_node:最近邻节点
        nearest_distance:最短距离
        '''
        self.root = None
        self.nearest_node = None
        self.nearest_distance = None

    def createTree(self, X, split):
        '''
        创建KD树
        '''
        n_sample = X.shape[0]
        dim = X.shape[1]
        mid = n_sample // 2

        # 根据选定的维度排序
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
        '''
        中序遍历输出kd树
        '''
        if node is not None:
            print(node.value)
            self.printTree(node.left)
            self.printTree(node.right)

    def searchTree(self, node, x):
        node.isvisit = 1
        print(f"{node.value}正在被访问")
        # 递归搜索目标点所属区域的叶子节点
        if x[node.split] <= node.value[node.split] and node.left is not None:
            self.searchTree(node.left, x)
        elif x[node.split] > node.value[node.split] and node.right is not None:
            self.searchTree(node.right, x)

        # 计算距离并判断当前节点是否更近
        distance = self.cal_distance(node.value, x)
        print(f"{node.value}的距离为{distance}")
        if self.nearest_node is None:
            print(f"正在第一次获取最近邻：{node.value}")
            self.nearest_node = node
            self.nearest_distance = distance
        elif self.nearest_distance > distance:
            print(f"当前节点为{node.value},距离更短，更新最近邻")
            self.nearest_node = node
            self.nearest_distance = distance

        # 判断是否相交，并根据情况移动到另一子节点
        ccross_distance = np.abs(node.value[node.split] - x[node.split])
        if ccross_distance < self.nearest_distance:
            print(f"与分割超平面的距离为{ccross_distance}")
            if node.left is not None and node.left.isvisit is None:
                print(f"与当前节点的另一子节点区域有相交,是其左节点")
                self.searchTree(node.left, x)
            if node.right is not None and node.right.isvisit is None:
                print(f"与当前节点的另一子节点区域有相交,是其右节点")
                self.searchTree(node.right, x)

    def cal_distance(self, x1, x2):
        '''
        计算欧式距离
        '''
        return np.sqrt(np.sum((x1-x2)**2))

if __name__ == '__main__':
    X = np.array([
        [2,3],
        [5,4],
        [9,6],
        [4,7],
        [8,1],
        [7,2]
    ])
    x = np.array([3,4.5])
    kdtree = KDTree()
    kdtree.root = kdtree.createTree(X, 0)
    # kdtree.printTree(kdtree.root)
    kdtree.searchTree(kdtree.root, x)
    print(kdtree.nearest_node.value)
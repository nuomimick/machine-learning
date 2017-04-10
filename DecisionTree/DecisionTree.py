# 简单分类决策树的实现
from collections import Counter
from math import log2
import numpy as np

class generate_node():
    def __init__(self):
        self.label = None
        self.data = None
        self.feature = None
        self.path = {}

    def __str__(self):
        return "label:{},data:{},feature:{},\npath:{}".format(self.label,self.data,self.feature,self.path)

class DecisionTree:
    def __init__(self):
        pass
    def fit(self,x_train,y_train):
        self.D = np.column_stack((x_train,y_train))
        A = list(range(self.D.shape[1]-1))
        self.node =  generate_node()
        self.__tree_generate(self.D,A,self.node)

    # 决策树生成
    def __tree_generate(self, D, A, node):
        label = D[:, -1]
        node.data = D[:,:-1]
        if len(set(label)) == 1:
            node.label = label[0]
            return
        if len(A) == 0 or (D[:,:-1] == D[0,:-1]).all():
            node.label = self.__get_label_D(label)
            return
        a = self.__find_best_feature(D, A)
        node.feature = a# 预测时有用，减1防止数组越界
        Ap = A.copy()
        Ap.remove(a)
        c = D[:, a]
        for v in set(self.D[:, a]):
            Dv = D[c == v]
            if len(Dv) == 0:
                next_node = generate_node()
                next_node.label = self.__get_label_D(label)
                next_node.data = []
                node.path[v] = next_node
                return
            else:
                next_node = generate_node()
                self.__tree_generate(Dv, Ap, next_node)
                node.path[v] = next_node

    def __get_label_D(self,label):
        cnt = Counter()
        for lb in label:
            cnt[lb] += 1
        return max(cnt.items(), key=lambda x: x[1])[0]

    def predict(self,x_test):
        y_preds = []
        for arr in x_test:
            node = self.node
            while(node.label == None):
                node = node.path[arr[node.feature]]
            y_preds.append(node.label)
        return y_preds

    def __info_entropy(self,D):
        #信息熵
        label = D[:,-1]
        cnt = Counter()
        length = len(label)
        for lb in label:
            cnt[lb] += 1
        ent = 0.
        for lb,count in cnt.items():
            p = count / length
            ent += p * log2(p) if p != 0 else 0
        return -ent
        
    def __info_gain(self,D,a):
        #信息增益
        c = D[:,a]#属性a的列
        vs = set(c)#属性的值
        gain = 0.
        for v in vs:
            Dv = D[c==v]
            gain += self.__info_entropy(Dv) * len(Dv) / len(D)
        return self.__info_entropy(D) - gain
    
    def __find_best_feature(self,D,A):
        gain_max = max([(self.__info_gain(D,i),i) for i in A])
        return gain_max[1]#最佳划分属性   

def main():
    dt = DecisionTree()
    x_train = np.array([[0,0,0,0,0,0],[1,0,1,0,0,0],[1,0,0,0,0,0],[0,1,0,0,1,1],[1,1,0,1,1,1],
                        [0,2,2,0,2,1],[2,1,1,1,0,0],[1,1,0,0,1,1],[2,0,0,2,2,0],[0,0,1,1,1,0]])
    y_train = np.array([1,1,1,1,1,0,0,0,0,0])
    x_test = np.array([[0,0,1,0,0,0],[2,0,0,0,0,0],[1,1,0,0,1,0],[1,1,1,1,1,0],[2,2,2,2,2,0],[2,0,0,2,2,1],[0,1,0,1,0,0]])
    y_test = np.array([1,1,1,0,0,0,0])
    dt.fit(x_train,y_train)
    print(dt.predict(x_test))

if __name__ == '__main__':
    main()


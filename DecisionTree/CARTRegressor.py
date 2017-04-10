# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:54:16 2017

@author: ruihuanz
"""
import numpy as np
class CARTRegressor:
    def __init__(self,gamma,min_sample):
        # gamma:允许的误差下降值
        # min_sample:每个节点的最小样本数
        self.gamma = gamma 
        self.min_sample = min_sample 

    def createTree(self,dataSet):
        feat,val = self.chooseBestSplit(dataSet)
        if feat == None:return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val 
        lSet,rSet = self.binSplitDataSet(dataSet,feat,val)
        retTree['left'] = self.createTree(lSet)
        retTree['right'] = self.createTree(rSet)
        return retTree
        
    def binSplitDataSet(self,dataSet,feature,value):
        mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
        mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:] 
        return mat0,mat1

    def regLeaf(self,dataSet):
        return np.mean(dataSet[:,-1])

    def regErr(self,dataSet):
        return np.var(dataSet[:-1]) * dataSet.shape[0]

    def chooseBestSplit(self,dataSet):
        if len(set(dataSet[:,-1])) == 1:
            return None,self.regLeaf(dataSet)
        m,n = dataSet.shape
        S = self.regErr(dataSet)#划分前
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        for featIndex in range(n-1):
            for splitVal in set(dataSet[:,featIndex]):
                mat0,mat1 = self.binSplitDataSet(dataSet,featIndex,splitVal)
                if mat0.shape[0] < self.min_sample or mat1.shape[0] < self.min_sample:
                    continue
                newS = self.regErr(mat0) + self.regErr(mat1)#划分后
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
        if S - bestS < self.gamma:
            return None,self.regLeaf(dataSet)
        mat0,mat1 = self.binSplitDataSet(dataSet,bestIndex,bestValue)
        if mat0.shape[0] < self.min_sample or mat1.shape[1] < self.min_sample:
            return None,self.regLeaf(dataSet)
        return bestIndex,bestValue

        
    def fit(self,x_train,y_train,eval_sets):
        dataSet = np.column_stack((x_train,y_train))
        dataSet_eval = np.column_stack(eval_sets)
        self.tree = self.createTree(dataSet)
        self.tree = self.prune(self.tree,dataSet_eval)

    def getMean(self,tree):
        if isinstance(tree['left'],dict):#dict说明不是叶子节点
            tree['left'] = self.getMean(tree['left'])
        if isinstance(tree['right'],dict):
            tree['right'] = self.getMean(tree['right'])
        return (tree['left'] + tree['right']) / 2 

    def prune(self,tree,eval_sets):#后剪枝
        if eval_sets.shape[0] == 0:return self.getMean(tree)
        if isinstance(tree['right'],dict) or isinstance(tree['left'],dict):
            lSet,rSet = self.binSplitDataSet(eval_sets,tree['spInd'],tree['spVal'])
        if isinstance(tree['left'],dict):
            tree['left'] = self.prune(tree['left'],lSet)
        if isinstance(tree['right'],dict):
            tree['right'] = self.prune(tree['right'],rSet)
        if not isinstance(tree['left'],dict) and not isinstance(tree['right'],dict):
            lSet,rSet = self.binSplitDataSet(eval_sets,tree['spInd'],tree['spVal'])
            errorNoMerge = np.sum(np.power(lSet[:,-1]-tree['left'],2)) + np.sum(np.power(rSet[:,-1]-tree['right'],2))
            treeMean = (tree['left'] + tree['right']) / 2 
            errorMerge = np.sum(np.power(eval_sets[:,-1]-treeMean,2))
            if errorMerge < errorNoMerge:
                return treeMean
            else:
                return tree
        else:
            return tree

    def predict(self,x_test):
        result = []
        for x in x_test:
            tree = self.tree
            while isinstance(tree,dict):
                if x[tree['spInd']] > tree['spVal']:
                    tree = tree['left']
                else:
                    tree = tree['right']
            result.append(tree)
        return np.array(result)

def main():
    from sklearn.datasets import load_boston
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeRegressor

    bst = load_boston()
    data = bst.data 
    target = bst.target
    x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)
    cart = CARTRegressor(5,5)
    cart.fit(x_train,y_train,eval_sets=(x_test,y_test))
    print(np.mean(np.abs(cart.predict(x_test)-y_test)))

    # regressor = DecisionTreeRegressor(random_state=0)
    # regressor.fit(x_train,y_train)
    # print(np.mean(np.abs(regressor.predict(x_test)-y_test)))
    

if __name__=='__main__':
    main()

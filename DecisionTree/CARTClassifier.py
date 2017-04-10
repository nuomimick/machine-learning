# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 10:54:16 2017

@author: ruihuanz
"""
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

class CARTClassifier:
    def __init__(self,min_sample):
        # min_sample:每个节点的最小样本数
        self.min_sample = min_sample 

    def createTree(self,dataSet,eval_sets):
        feat,val = self.chooseBestSplit(dataSet,eval_sets)
        if feat == None:return val
        retTree = {}
        retTree['spInd'] = feat
        retTree['spVal'] = val 
        lSet,rSet = self.binSplitDataSet(dataSet,feat,val)
        leSet,reSet = self.binSplitDataSet(eval_sets,feat,val)
        retTree['left'] = self.createTree(lSet,leSet)
        retTree['right'] = self.createTree(rSet,reSet)
        return retTree
        
    def binSplitDataSet(self,dataSet,feature,value):
        mat0 = dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
        mat1 = dataSet[np.nonzero(dataSet[:,feature] <= value)[0],:] 
        return mat0,mat1

    def clfLeaf(self,dataSet):
        labels = dataSet[:,-1]
        cnt = Counter()
        for lb in labels:
            cnt[lb] += 1
        return max(cnt.items(),key=lambda tp:tp[1])[0]

    def clfErr(self,dataSet,eval_sets):
        label = self.clfLeaf(dataSet)
        return len(eval_sets[eval_sets[:,-1] == label])

    def gini(self,dataSet):
        gn = 0.
        for lb in set(dataSet[:,-1]):
            gn += (len(dataSet[dataSet[:,-1] == lb]) / len(dataSet))**2
        return 1 - gn

    def chooseBestSplit(self,dataSet,eval_sets):
        if len(set(dataSet[:,-1])) == 1:
            return None,self.clfLeaf(dataSet)
        m,n = dataSet.shape
        S = self.gini(dataSet)#划分前的gini指数
        pS = self.clfErr(dataSet,eval_sets)#划分前的准确率
        bestS = np.inf
        bestIndex = 0
        bestValue = 0
        best_p = 0
        for featIndex in range(n-1):
            for splitVal in set(dataSet[:,featIndex]):
                mat0,mat1 = self.binSplitDataSet(dataSet,featIndex,splitVal)
                mat0_eval,mat1_eval = self.binSplitDataSet(eval_sets,featIndex,splitVal)
                if mat0.shape[0] < self.min_sample or mat1.shape[0] < self.min_sample:
                    continue
                newS = len(mat0) / len(dataSet) * self.gini(mat0) + len(mat1) / len(dataSet) * self.gini(mat1)#划分后的gini指数             
                p_newS = (self.clfErr(mat0,mat0_eval) + self.clfErr(mat1,mat1_eval))#划分后的准确率
                if newS < bestS:
                    bestIndex = featIndex
                    bestValue = splitVal
                    bestS = newS
                    best_p = p_newS
        if best_p <= pS:#预剪枝
            return None,self.clfLeaf(dataSet)
        return bestIndex,bestValue

        
    def fit(self,x_train,y_train):
        x_train0,x_test1,y_train0,y_test1 = train_test_split(x_train,y_train,test_size=0.1)
        dataSet = np.column_stack((x_train0,y_train0))
        dataSet_eval = np.column_stack((x_test1,y_test1))
        self.tree = self.createTree(dataSet,dataSet_eval)
        print(self.tree)
        self.tree = self.prune(self.tree,dataSet,dataSet_eval)
        print(self.tree)

    def getMean(self,tree,dataSet):
        if tree['spInd'] == None:
            return self.clfLeaf(dataSet)
        lSet, rSet = self.binSplitDataSet(dataSet, tree['spInd'], tree['spVal'])
        if isinstance(tree['left'],dict):#dict说明不是叶子节点
            tree['left'] = self.getMean(tree['left'],lSet)
        if isinstance(tree['right'],dict):
            tree['right'] = self.getMean(tree['right'],rSet)

    def prune(self,tree,dataSet,eval_sets):#后剪枝
        if eval_sets.shape[0] == 0:return self.getMean(tree,dataSet)
        if isinstance(tree['right'],dict) or isinstance(tree['left'],dict):
            lSet,rSet = self.binSplitDataSet(eval_sets,tree['spInd'],tree['spVal'])
            ldSet, rdSet = self.binSplitDataSet(dataSet, tree['spInd'], tree['spVal'])
        if isinstance(tree['left'],dict):
            tree['left'] = self.prune(tree['left'],ldSet,lSet)
        if isinstance(tree['right'],dict):
            tree['right'] = self.prune(tree['right'],rdSet,rSet)
        if not isinstance(tree['left'],dict) and not isinstance(tree['right'],dict):
            lSet,rSet = self.binSplitDataSet(eval_sets,tree['spInd'],tree['spVal'])
            errorNoMerge = np.sum((lSet[:,-1]-tree['left'])==0) + np.sum((rSet[:,-1]-tree['right'])==0)
            errorMerge = np.sum((eval_sets[:,-1]-self.clfLeaf(dataSet))==0)
            if errorMerge < errorNoMerge:
                return tree
            else:
                return self.clfLeaf(dataSet)
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
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    iris = load_digits()
    data = iris.data 
    target = iris.target
    x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.2,random_state=0)

    cart = CARTClassifier(5)
    cart.fit(x_train,y_train)
    print(np.sum((cart.predict(x_test)-y_test)==0))

    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(x_train,y_train)
    print(np.sum((classifier.predict(x_test) - y_test) == 0))
    

if __name__=='__main__':
    main()
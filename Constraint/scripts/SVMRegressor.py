__author__ = 'spijs'

from Regressor import Regressor as Reg
from sklearn import svm, preprocessing,linear_model

class SVMRegressor(Reg):

    def __init__(self):
        pass

    def train(self,train,corr):
        train = train.flatten()
        self.scaler = preprocessing.StandardScaler().fit(train)
        sX_train = self.scaler.transform(train)
        self.clf = svm.SVR()
        self.clf.fit(sX_train, corr)

    def test(self,test):
        test = test.flatten()
        sX_test = self.scaler.transform(test)
        pred = self.clf.predict(sX_test)
        return pred

class LinearRegressor(Reg):
    def __init__(self):
        pass

    def train(self,train,corr):
        self.clf = linear_model.LinearRegression()
        self.clf.fit(train, corr)

    def test(self,test):
        return self.clf.predict(test)
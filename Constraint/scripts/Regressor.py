__author__ = 'spijs'

from sklearn import svm, preprocessing,linear_model
from prices_data import *
import matplotlib.pyplot as plt
from sknn.mlp import *
from sknn.mlp import Regressor as Reg
from abc import ABCMeta,abstractmethod
import pickle
import numpy as np

''' Abstract regressor class. Each Regressor instance offers a test method'''
class Regressor:
    __metaclass__ = ABCMeta

Regressor.register(tuple)

assert issubclass(tuple, Regressor)
assert isinstance((), Regressor)


@abstractmethod
def test(self,test):
    '''
    :param test: array of test days
    :return: (should return array of result and array of correct values.'''
    pass

class LinearRegressor(Regressor):
    ''' Linear Regressor
    '''
    def __init__(self,useclassify,prev,train_days):
        '''
        :param useclassify: True if the classification feature should be used.
        :param prev: The number of previous days that should be included as features.
        :param train_days: The amount of days before each test day that should be used to train the regressor.
        :return: LinearRegressor class
        '''
        self.prev=prev
        self.train_days=train_days
        if useclassify:
            self.classifier=pickle.load(open('classifier.p'))
        else:
            self.classifier=None

    def test(self,test):
        '''
        :param test: array of days for which the following two weeks need to be predicted.
        :return: Predictions and correct values.
        '''
        column_features, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(self.train_days,test,self.prev)

        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day
            X_test, X_train, Y_test, y_train = get_data_for_day(self.classifier,self.prev,column_features,column_prev_features, column_predict, dat, day,
                                                                          historic_days)
            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            pred =  clf.predict(X_test)
            result.append(pred)
            correct.append(Y_test)

        return np.array(result), np.array(correct)


class EnsembleLinearRegressor(Regressor):
    '''
       Actually the same as LinearRegressor. Only now ten different 'linear regressors' with subsets of the used features.
       Final result is the average of the results of each regressor.
    '''
    def __init__(self,useclassify,prev,train_days):
        '''
        :param useclassify: True if peak class should be used as a feature
        :param prev: Number of previous days that need to be included as features
        :param train_days: Number of days that are used for training the regressors.
        :return: EnsembleLinearRegressor instance
        '''
        self.prev=prev
        self.train_days=train_days
        if useclassify:
            self.classifier=pickle.load(open('classifier.p'))
        else:
            self.classifier=None

    def test(self,test):
        '''
        Creates 10 linear regressor that each use 6 out of 8 randomly chosen features.
        Final result is the average result of each regressor.
        :param test: days to be used for testing
        :return: array with predicted values, array with correct values
        '''
        n = 10
        features = self.get_selected_features(n)
        print features
        _, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(self.train_days,test,self.prev)
        final = []
        for day in test:
            result = np.zeros(672)
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day
            for column_features in features:
                X_test, X_train, Y_test, y_train = get_data_for_day(self.classifier,self.prev,column_features,column_prev_features, column_predict, dat, day,
                                                                    historic_days)
                clf = linear_model.LinearRegression()
                clf.fit(X_train, y_train)
                pred =  np.array(clf.predict(X_test))
                result = np.add(pred,result)
            final.append(result/10)
            correct.append(Y_test)
        return np.array(final), np.array(correct)


    def get_selected_features(self,n):
        '''
        Returns an array containing n arrays of features.
        :param n: amount of featuresets this function should return
        :return: array containing n arrays of features
        '''
        all_elements = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                        'ORKTemperature', 'ORKWindspeed']
        result = []
        for i in range(n):
            temp = []
            sampled = random.sample(range(0, 8), 6)
            for j in sampled:
                temp.append(all_elements[j])
            result.append(temp)
        return result





''' Regression network'''
class Network(Regressor):
    '''
    Neural network for regression.
    '''

    def __init__(self,train_days,useclassify, prev,nbOfLayers, learning_rate,nb_iter,hidden,stable,rule,norm=False):
        '''
        :param train_days: Amount of days used for training
        :param useclassify: True if the peak class should be added as a feature
        :param prev: Amount of previous days that need to be included as a feature.
        :param nbOfLayers: Number of layers to use in the network.
        :param learning_rate: Learning rate of the network.
        :param nb_iter: Number of maximum iterations to train the network.
        :param hidden: Amount of hidden cells in each hidden layer.
        :param stable: Amount of stable iterations required to stop the training.
        :param rule: Learning method to be used.
        :param norm: True if the features should be normalized.
        :return: Network instance
        '''
        self.layers = []
        for i in range(nbOfLayers-1):
            self.layers.append(Layer("Tanh",name='hidden'+str(i),units=hidden))
        self.layers.append(Layer("Linear", name = 'output', units = 1))
        self.learning_rate=learning_rate
        self.n_iter=nb_iter
        self.n_stable=stable
        self.learning_rule=rule
        self.prev=prev
        self.norm = norm
        self.train_days=train_days
        if useclassify:
            self.classifier=pickle.load(open('classifier.p'))
        else:
            self.classifier=None


    def create_nn(self,valid_in,valid_out):
        '''
        Creates a Neural Network with a given validation set input and output.
        :param valid_in: validation set input
        :param valid_out: correct validaton set output
        :return: sknn Neural Network
        '''
        return Reg(
            layers = self.layers,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            valid_set=(valid_in,valid_out),
            n_stable=self.n_stable,
            learning_rule=self.learning_rule,
            verbose=True)

    def test(self, test):
        '''
        This function first fetches the correct data for training, testing and for validating the network.
        Then it creates the network based on this information, fits it to the data and finally predicts the test days.
        :param test: days for which the next two weeks need to be predicted
        :return: Array with the predicted values, Array with the correct values
        '''
        column_features, column_predict,column_prev_features, dat,historic_days, result,correct, test = load_data(self.train_days,test,self.prev)

        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            X_test, X_train, Y_test, y_train = get_data_for_day(self.classifier,self.prev,column_features, column_prev_features, column_predict, dat, day,
                                                                          historic_days)

            rows_val = get_data_days(dat, day, timedelta(28))[-14*48:]
            print len(rows_val)
            X_val = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_val]
            rows_before_test = get_data_prevdays(dat, day, timedelta(historic_days))
            additional_info = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_before_test]
            Y_val = [eval(row[column_predict]) for row in rows_val]
            additional_info_val = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_val]
            X_VAL = []
            if self.classifier:
                classifications = self.classifier.predict(X_val)
            for i in range(len(X_val)):
                extra = []
                for j in range (self.prev,0,-1):
                    if i-j*48 < 0:
                        row = additional_info[i-j*48]
                    else:
                        row = additional_info_val[i-j*48]
                    extra = extra + row
                if self.classifier:
                    extra = extra+[classifications[i]]
                X_VAL.append(X_val[i]+extra)
            print 'Val size ', np.array(X_VAL).shape

            scaler = preprocessing.StandardScaler().fit(X_train)
            # Normalize if necessary
            try :
                if self.norm:
                    X_train = scaler.transform(X_train)
                    X_test = scaler.transform(X_test)
                    X_VAL = scaler.transform(X_VAL)
            except AttributeError:
                pass

            nn = self.create_nn(np.array(X_VAL),np.array(Y_val))
            nn.fit(np.array(X_train), np.array(y_train))
            result.append(nn.predict(np.array(X_test)))
            correct.append(Y_test)

        return np.array(result), np.array(correct)

class SVMRegressor(Regressor):
    ''' Regressor using SVM'''

    def __init__(self,useclassify,prev,train_days):
        '''
        :param useclassify: Whether or not to use the peak class as a feature
        :param prev: Amount of previous days that should be included as a feature
        :param train_days: Amount of days required for training
        :return: SVMRegressor instance
        '''
        self.prev=prev
        self.train_days=train_days
        if useclassify:
            self.classifier=pickle.load(open('classifier.p'))
        else:
            self.classifier=None

    def test(self,test):
        '''
        First fetches all data, then trains SVM model for each separate test day.
        :param test: Array containing the test days
        :return: Array with predicted values, Array with correct values
        '''
        column_features, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(self.train_days,test,self.prev)
        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day
            preds = [] # [(model_name, predictions)]

            # method one: linear
            X_test, X_train, Y_test, y_train = get_data_for_day(self.classifier,self.prev,column_features,column_prev_features, column_predict, dat, day,
                                                                          historic_days)

            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            preds.append( ('lin', clf.predict(X_test)) )

            # same features but preprocess the data by scaling to 0..1
            scaler = preprocessing.StandardScaler().fit(X_train)
            sX_train = scaler.transform(X_train)
            sX_test = scaler.transform(X_test)
            clf = svm.SVR()
            clf.fit(sX_train, y_train)
            pred = clf.predict(sX_test)
            result.append(pred)
            correct.append(Y_test)

        return np.array(result), np.array(correct)


def get_data_for_day(classifier,prev,column_features,column_prev_features,column_predict,dat,day,historic_days):
    '''
    Fetches the required features for a given day.

    :param classifier: Whether or not the peak class should be included as a feature
    :param prev: Amount of previous days that need to be included as a feature
    :param column_features: Features to use for train data
    :param column_prev_features: Features to use for data of previous days
    :param column_predict: Features to use for test data
    :param dat: All the necessary data
    :param day: Day for which the required data is retrieved.
    :param historic_days: Amount of days used for training
    :return: train input, test input, train output, test output
    '''

    # Get train data

    rows_before_test = get_data_prevdays(dat, day, timedelta(historic_days))
    X_train = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_before_test]
    y_train = [eval(row[column_predict]) for row in rows_before_test]
    additional_info = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_before_test]
    X = []
    if classifier:
        classifications = classifier.predict(X_train)
    print np.array(X_train).shape
    # Add data from same time slot on previous day + peak classification of this time slot.
    for i in range(prev*48,len(X_train)):
        extra = []
        for j in range (prev,0,-1):
            extra = extra + additional_info[i-j*48]
        if classifier:
            extra = extra+[classifications[i]]
        X.append(X_train[i]+extra)
    print 'X train size: ' , np.array(X).shape

    # Get test data
    rows_tod = get_data_days(dat, day, timedelta(14))  # for next 2 weeks
    X_test = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_tod]
    print 'originele X_test size' , np.array(X_test).shape
    Y_test = [eval(row[column_predict]) for row in rows_tod]
    additional_info_test = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_tod]
    X_TEST = []
    if classifier:
        classifications = classifier.predict(X_test)
    # Add data from same time slot on previous day + peak classification of this time slot.
    for i in range(len(X_test)):
        extra = []
        for j in range (prev,0,-1):
            if i-j*48 < 0:
                row = additional_info[i-j*48]
            else:
                row = additional_info_test[i-j*48]
            extra = extra + row
        if classifier:
            extra = extra+[classifications[i]]
        X_TEST.append(X_test[i]+extra)
    print 'X test size:', np.array(X_TEST).shape
    print 'y_size:', np.array(y_train[prev*48:]).shape
    return X_TEST,X,Y_test,y_train[prev*48:]

def load_data(train_days,test,prev):
    '''
    Loads the required data and returns other important information.
    :param train_days: Amount of days used for training.
    :param test: Filename of the test days
    :param prev: Amount of previous days used as a feature.
    :return: Features for training, Features for testing, Features for previous days, prices data, number of historic days, [],[], array with test days
    '''
    datafile = '../data/cleanData.csv'
    dat = load_prices(datafile)
    column_features = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                        'ORKTemperature', 'ORKWindspeed']
    column_prev_features = ['HolidayFlag', 'DayOfWeek', 'ForecastWindProduction', 'SystemLoadEA', 'WeekOfYear', 'SMPEA',
                            'CO2Intensity', 'SMPEP2', 'ORKTemperature', 'ActualWindProduction', 'ORKWindspeed', 'SystemLoadEP2']
    column_predict = 'SMPEP2'
    historic_days = train_days+prev
    test = get_test_days(test)
    result = []
    return column_features, column_predict,column_prev_features, dat, historic_days, result,[], test


def get_test_days(testfile):
    '''
    :param testfile: filename without extension that contains the test data.
    :return: Array containing the days that should be tested
    '''
    print testfile
    f = open('../data/'+testfile+'.txt')
    days = f.readlines()
    return days

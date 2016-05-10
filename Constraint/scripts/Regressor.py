__author__ = 'spijs'

from sklearn import svm, preprocessing,linear_model
from prices_data import *
import matplotlib.pyplot as plt
from sknn.mlp import *
from sknn.mlp import Regressor as Reg
from abc import ABCMeta,abstractmethod
import numpy as np

''' Regressor abstract class'''
class Regressor:
    __metaclass__ = ABCMeta

Regressor.register(tuple)

assert issubclass(tuple, Regressor)
assert isinstance((), Regressor)

@abstractmethod
def test(self,test):
    pass

''' Linear regressor'''
class LinearRegressor(Regressor):
    def __init__(self,pred):
        self.pred=pred

    def test(self,test):
        column_features, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(test)

        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day

            # method one: linear
            X_test, X_train, Y_test, y_train = get_data_for_day(self.pred,column_features,column_prev_features, column_predict, dat, day,
                                                                          historic_days)

            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            pred =  clf.predict(X_test)
            result.append(pred)
            correct.append(Y_test)

        return np.array(result), np.array(correct)

''' Regression network'''
class Network(Regressor):

    def __init__(self, prev,nbOfLayers, learning_rate,nb_iter,hidden,stable,rule):
        self.layers = []
        for i in range(nbOfLayers-1):
            self.layers.append(Layer("Tanh",name='hidden'+str(i),units=hidden))
        self.layers.append(Layer("Linear", name = 'output', units = 1))
        self.learning_rate=learning_rate
        self.n_iter=nb_iter
        self.n_stable=stable
        self.learning_rule=rule
        self.prev=prev


    def create_nn(self,valid_in,valid_out):
        return Reg(
            layers = self.layers,
            learning_rate=self.learning_rate,
            n_iter=self.n_iter,
            valid_set=(valid_in,valid_out),
            n_stable=self.n_stable,
            learning_rule=self.learning_rule,
            verbose=True)

    def test(self, test):
        column_features, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(test)

        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            X_test, X_train, Y_test, y_train = get_data_for_day(self.prev,column_features, column_prev_features, column_predict, dat, day,
                                                                          historic_days)
            rows_val = get_data_days(dat, day, timedelta(1))
            X_val = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_val]
            rows_before_test = get_data_prevdays(dat, day, timedelta(historic_days))
            additional_info = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_before_test]
            Y_val = [eval(row[column_predict]) for row in rows_val]
            additional_info_val = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_val]
            X_VAL = []
            for i in range(len(X_val)):
                extra = []
                for j in range (self.prev,0,-1):
                    if i-j < 0:
                        row = additional_info[i-j]
                    else:
                        row = additional_info_val[i-j]
                    extra = extra + row
                X_VAL.append(X_val[i]+extra)
            print 'Val size ', np.array(X_VAL).shape
            nn = self.create_nn(np.array(X_VAL),np.array(Y_val))
            nn.fit(np.array(X_train), np.array(y_train))
            result.append(nn.predict(np.array(X_test)))
            correct.append(Y_test)

        return np.array(result), np.array(correct)

''' SVM Regressor'''
class SVMRegressor(Regressor):

    def __init__(self,prev):
        self.prev=prev

    def test(self,test):
        column_features, column_predict,column_prev_features, dat, historic_days, result,correct, test = load_data(test)
        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day
            preds = [] # [(model_name, predictions)]

            # method one: linear
            X_test, X_train, Y_test, y_train = get_data_for_day(self.prev,column_features,column_prev_features, column_predict, dat, day,
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

def get_data_for_test_day(column_features, column_predict, dat, day, historic_days):
    rows_prev = get_data_prevdays(dat, day, timedelta(historic_days))
    X_train = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_prev]
    y_train = [eval(row[column_predict]) for row in rows_prev]
    rows_tod = get_data_days(dat, day, timedelta(14))  # for next 2 weeks
    X_test = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_tod]
    Y_test = [eval(row[column_predict]) for row in rows_tod]
    return X_test, X_train, Y_test, y_train

def get_data_for_day(prev,column_features,column_prev_features,column_predict,dat,day,historic_days):
    print day
    rows_before_test = get_data_prevdays(dat, day, timedelta(historic_days))
    X_train = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_before_test]
    y_train = [eval(row[column_predict]) for row in rows_before_test]
    additional_info = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_before_test]
    train_size = (historic_days-prev)*48
    X = []
    print np.array(X_train).shape
    for i in range(prev*48,train_size):
        extra = []
        for j in range (prev,0,-1):
            extra = extra + additional_info[i-j*48]
        X.append(X_train[i]+extra)
    print 'X train size: ' , np.array(X).shape
    rows_tod = get_data_days(dat, day, timedelta(14))  # for next 2 weeks
    X_test = [[eval(v) for (k, v) in row.iteritems() if k in column_features] for row in rows_tod]
    print 'originele X_test size' , np.array(X_test).shape
    Y_test = [eval(row[column_predict]) for row in rows_tod]
    additional_info_test = [[eval(v) for (k, v) in row.iteritems() if k in column_prev_features] for row in rows_tod]
    X_TEST = []
    for i in range(len(X_test)):
        extra = []
        for j in range (prev,0,-1):
            if i-j*48 < 0:
                row = additional_info[i-j*48]
            else:
                row = additional_info_test[i-j*48]
            extra = extra + row
        X_TEST.append(X_test[i]+extra)
    print 'X test size:', np.array(X_TEST).shape
    print 'y_size:', np.array(y_train[prev:train_size*48]).shape
    return X_TEST,X,Y_test,y_train[prev:train_size*48]


def load_data(test):
    datafile = '../data/cleanData.csv'
    dat = load_prices(datafile)
    column_features = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                        'ORKTemperature', 'ORKWindspeed']
    column_prev_features = ['HolidayFlag', 'DayOfWeek', 'ForecastWindProduction', 'SystemLoadEA', 'WeekOfYear', 'SMPEA',
                            'CO2Intensity', 'SMPEP2', 'ORKTemperature', 'ActualWindProduction', 'ORKWindspeed', 'SystemLoadEP2']
    column_predict = 'SMPEP2'
    historic_days = 30
    test = get_test_days(test)
    result = []
    return column_features, column_predict,column_prev_features, dat, historic_days, result,[], test


def plot_preds(modelpreds, y_test):
    # Print the mean square errors
    print "Residual sum of squares:"
    for (name,preds) in modelpreds:
        print "%s: %.2f"%(name, np.mean((preds-y_test)**2))

    # Explained variance score: 1 is perfect prediction
    #print "Variance scores:"
    #for (name,clf) in clfs:
    #    pred = clf.predict(X_test)
    #    print "%s: %.2f"%(name, clf.score(X_test, y_test))

    # Plot price vs prediction
    plt.scatter(xrange(len(y_test)), y_test,  color='black', label='actual')
    for (name,preds) in modelpreds:
        plt.plot(xrange(len(y_test)), preds, linewidth=3, label=name)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()



def get_test_days(testfile):
    print testfile
    f = open('../data/'+testfile+'.txt')
    days = f.readlines()
    return days
__author__ = 'spijs'

from sklearn import svm, preprocessing,linear_model
from prices_data import *
import matplotlib.pyplot as plt
from sknn.mlp import *
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
    def __init__(self):
        pass

    def test(self,test):
        column_features, column_predict, dat, historic_days, result,correct, test = load_data(test)

        for i in test:
            day = get_data_day(dat,i)

            # method one: linear
            X_test, X_train, Y_test, y_train = get_data_for_test_day(column_features, column_predict, dat, day,
                                                                          historic_days)

            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            pred =  clf.predict(X_test)
            result.append(pred)
            correct.append(Y_test)

        return np.array(result), np.array(correct)

''' Regression network'''
class Network(Regressor):

    def __init__(self, nbOfLayers, learning_rate,nb_iter,valid_input,valid_output,hidden,stable,rule):
        layers = []
        for i in range(nbOfLayers-1):
            layers.append(Layer("Tanh",name='hidden'+str(i),units=hidden))
        layers.append(Layer("Linear", name = 'output', units = 48))
        self.nn = Regressor(
            layers = layers,
            learning_rate=learning_rate,
            n_iter=nb_iter,
            valid_set=(valid_input,valid_output),
            n_stable=stable,
            learning_rule=rule,
            verbose=True
        )

    def train(self, train, correct):
        self.nn.fit(train, correct)

    def test(self, test):
        result =  self.nn.predict(test)
        final = []
        d = 0
        current = []
        for i in range(len(test)):
            current.append(result[i])
            d+=1
            if d==14:
                final.append(np.array(current).flatten())
                current = []
                d=0
        #plot_preds(result.flatten()[0:48] , correct.flatten()[0:48])
        final = np.array(final)
        return final

''' SVM Regressor'''
class SVMRegressor(Regressor):

    def __init__(self):
        pass

    def test(self,test):
        column_features, column_predict, dat, historic_days, result,correct, test = load_data(test)
        for day in test:
            day = datetime.strptime(day.rstrip('\n'), '%Y-%m-%d').date()
            print day
            preds = [] # [(model_name, predictions)]

            # method one: linear
            X_test, X_train, Y_test, y_train = get_data_for_test_day(column_features, column_predict, dat, day,
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


def load_data(test):
    datafile = '../data/cleanData.csv'
    dat = load_prices(datafile)
    column_features = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                       'CO2Intensity', 'ORKTemperature', 'ORKWindspeed']
    column_predict = 'SMPEP2'
    historic_days = 25
    test = get_test_days(test)
    result = []
    return column_features, column_predict, dat, historic_days, result,[], test


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
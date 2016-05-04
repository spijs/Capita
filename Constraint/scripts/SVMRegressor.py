__author__ = 'spijs'

from Regressor import Regressor as Reg
from sklearn import svm, preprocessing,linear_model
from prices_data import *
import matplotlib.pyplot as plt

import numpy as np

class LinearRegressor(Reg):
    def __init__(self):
        pass

    def train(self,train,corr):
        pass

    def test(self,test):
        datafile = '../data/cleanData.csv'
        dat = load_prices(datafile)

        column_features = [ 'HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA','CO2Intensity', 'ORKTemperature', 'ORKWindspeed' ]
        column_predict = 'SMPEP2'
        historic_days = 30
        test = get_test_days()
        result = []

        for i in test:
            day = get_date_by_id(dat,i)

            # method one: linear
            rows_prev = get_data_prevdays(dat, day, timedelta(historic_days))
            X_train = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_prev]
            y_train = [ eval(row[column_predict]) for row in rows_prev ]
            rows_tod = get_data_days(dat, day, timedelta(14)) # for next 2 weeks
            X_test = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_tod]

            clf = linear_model.LinearRegression()
            clf.fit(X_train, y_train)
            pred =  clf.predict(X_test)
            result.append(pred)

        result = np.array(result)
        print result.shape
        return result


class SVMRegressor(Reg):

    def __init__(self):
        pass

    def train(self,train,corr):
        pass



    def test(self,test):
        datafile = '../data/cleanData.csv'
        dat = load_prices(datafile)

        column_features = [ 'HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA','CO2Intensity', 'ORKTemperature', 'ORKWindspeed' ]
        column_predict = 'SMPEP2'
        historic_days = 30
        test = get_test_days()
        result = []

        for i in test:
            day = get_date_by_id(dat,i)
            preds = [] # [(model_name, predictions)]

            # method one: linear
            rows_prev = get_data_prevdays(dat, day, timedelta(historic_days))
            X_train = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_prev]
            y_train = [ eval(row[column_predict]) for row in rows_prev ]
            rows_tod = get_data_days(dat, day, timedelta(14)) # for next 2 weeks
            X_test = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_tod]
            y_test = [ eval(row[column_predict]) for row in rows_tod]
            if i == 0:
                print y_test[0:48*14]
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

        result = np.array(result)
        print result.shape
        return result

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

def get_test_days():
    result = []
    for i in range(7,792): # loop over all days in dataset
        if i % 84 ==0:
            result.append(i)
    return result
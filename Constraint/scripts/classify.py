__author__  = "Wout & Thijs"

from prices_data import *
from datetime import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pickle


def strtodate(x):
    return datetime.strptime(x, '%Y-%m-%d').date()

def getData():
    '''
    This method loads the needed data from a csv file, and creates a training and test set.
    :return: data for training and testing
    '''
    datafile = '../data/cleanData.csv'
    dat = load_prices(datafile)
    column_features = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                       'ORKTemperature', 'ORKWindspeed']
    column_predict = 'peak'

    testdates = ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01']

    testdates = list(map(strtodate, testdates))
    train_data = []
    train_result = []
    test_data = []
    test_result = []
    for row in dat:
        vec = []
        currentDate = datetime.strptime(row['#DateTime'], '%a %d/%m/%Y %H:%M').date()
        for col in column_features:
            vec.append(float(row[col]))
        test = False
        for d in testdates:
            if (currentDate >= d) and (currentDate < d + timedelta(14)):
                test = True
        if test:
            test_data.append(vec)
            test_result.append(int(row[column_predict]))
        else:
            train_data.append(vec)
            train_result.append(int(row[column_predict]))
    return train_data, train_result, test_data, test_result


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    Plot a confusion matrix
    :param cm: the matrix to be plotted
    :param title: the title of the matrix
    :param cmap: the colors to be used
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(['0', '1']))
    plt.xticks(tick_marks, ['0', '1'])
    plt.yticks(tick_marks, ['0', '1'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def classify():
    '''
    This functon creates a random forest classifier and uses a provided dataset to train and test this classifier.
    After that is plots both the regular and normalized confusion matrix
    '''
    x, y, test_data, test_result = getData()
    print "training data collected"
    clf = RandomForestClassifier(n_estimators=50, max_features=None, min_samples_leaf=5)
    clf.fit(x, y)
    pickle.dump(clf, open('classifier.p', 'wb'))
    res = clf.predict(test_data)
    resfile = open('../results/classification', 'wb')
    pickle.dump(res, resfile)
    resfile.close()
    # Compute confusion matrix
    cm = confusion_matrix(test_result, res)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)
    # Normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    plt.show()


if __name__ == '__main__':
    classify()

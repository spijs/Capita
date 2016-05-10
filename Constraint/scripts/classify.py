from sklearn.svm import SVC
from prices_data import *
from datetime import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
from sklearn import tree


def strtodate(x):
    return datetime.strptime(x, '%Y-%m-%d').date()

def getData():
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




if __name__ == '__main__':
    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(['0', '1']))
        plt.xticks(tick_marks, ['0', '1'])
        plt.yticks(tick_marks, ['0', '1'])
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    column_features = ['HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA',
                           'ORKTemperature', 'ORKWindspeed']
    column_predict = 'peak'

    x, y, test_data, test_result = getData()
    print "training data collected"
    # clf = SVC(degree=4)
    # clf = DecisionTreeClassifier(min_samples_split = 1)
    clf = RandomForestClassifier(n_estimators= 15, max_features=None, min_samples_leaf=5)
    clf.fit(x,y)
    res = clf.predict(test_data)
    print "printing trees"
    for t in [clf.estimators_[0]]:
        dot_data = StringIO()
        tree.export_graphviz(t.tree_, out_file=dot_data)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())
        graph.write_png('tree.png')


    errmat = [0,0,0,0,0]
    error = np.absolute(res-test_result)
    for e in error:
        errmat[e] = errmat[e]+1
    print np.array(errmat)/(len(test_result)*1.0)

    # Compute confusion matrix
    cm = confusion_matrix(test_result, res)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    plot_confusion_matrix(cm)

    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
    # print "prediction error: ", str(100.0*error)

    plt.show()

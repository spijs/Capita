__author__ = 'spijs'

from createdatasubsets import getData
from sknn.mlp import *
import argparse
import matplotlib.pyplot as plt
import pickle
import numpy as np
from Regressor import Regressor as Reg
from Regressor import SVMRegressor, LinearRegressor, Network


def run_regression(params):
    if params['type']=='network':
        reg = Network(params['classifier'],params['prev'],params['layers'],params['learning_rate'],params['iterations'],params['hidden'],params['stable'],params['rule'],params['norm'])
    elif params['type']=='svm':
        reg = SVMRegressor(params['classifier'],params['prev'])
    elif params['type']=='linear':
        reg = LinearRegressor(params['classifier'],params['prev'])
    else:
        reg = SVMRegressor(params['classifier'],params['prev'])
    result,correct = reg.test(params['data'])
    score = evaluate(result,correct)
    pickle.dump(reg,open('../saved_regressors/%s_layers_%srate_%shidden_%s_SCORE_%s' % (params['type'],params['layers'],params['learning_rate'],params['hidden'],score),'wb'))

def compare(params):
    neural = Network(params['prev'],params['layers'],params['learning_rate'],params['iterations'],params['hidden'],params['stable'],params['rule'])
    svm = SVMRegressor(params['prev'])
    linear = LinearRegressor(params['prev'])
    neural_result,correct = neural.test(params['data'])
    svm_result,_ = svm.test(params['data'])
    linear_result,_ = linear.test(params['data'])
    preds=[]
    preds.append(('nn', neural_result.flatten()[0:672]))
    preds.append(('svm', svm_result.flatten()[0:672]))
    preds.append(('linear', linear_result.flatten()[0:672]))
    plot_preds(preds,correct.flatten()[0:672])

def evaluate(preds,y_test):
    preds = preds.flatten()
    y_test = y_test.flatten()[0:len(preds)]
    print "%.2f"%(np.mean((preds-y_test)**2))
    return "%.2f"%(np.mean((preds-y_test)**2))

def plot_pred(preds, y_test):
    # Print the mean square errors
    print "Residual sum of squares:"

    print "%.2f"%(np.mean((preds-y_test)**2))

    # Explained variance score: 1 is perfect prediction
    #print "Variance scores:"
    #for (name,clf) in clfs:
    #    pred = clf.predict(X_test)
    #    print "%s: %.2f"%(name, clf.score(X_test, y_test))

    # Plot price vs prediction
    plt.scatter(xrange(len(y_test)), y_test,  color='black', label='actual')
    plt.plot(xrange(len(y_test)), preds, linewidth=3, label='netwerk')
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()

def plot_preds(modelpreds, y_test):
    # Plot price vs prediction
    plt.scatter(xrange(len(y_test)), y_test,  color='black', label='actual')
    for (name,preds) in modelpreds:
        plt.plot(xrange(len(y_test)), preds, linewidth=3, label=name)
    plt.axis('tight')
    plt.legend(loc='upper left')
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--learning_rate', dest='learning_rate',type=float, default=.01, help='learning rate to be used')
    parser.add_argument('-hi', '--hidden', dest='hidden', type=int, default=256, help='Number of nodes in hidden layer')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default= 100, help='Number of iterations for training the network')
    parser.add_argument('-l', '--layers',dest='layers',type=int, default=5, help='number of hidden layers used')
    parser.add_argument('--learning_rule', dest = 'rule', default='sgd',type=str,help = 'Learning rule to be used: rmsprop, sgd, adagrad, ...')
    parser.add_argument('-s', '--stable', dest = 'stable', default=10,type=int, help = 'number of stable iterations')
    parser.add_argument('-t', '--type', dest='type', type=str, default='network',help = 'type of regression used')
    parser.add_argument('-c', '--compare', dest = 'comp', default=None,help = 'compare different methods')
    parser.add_argument('-d','--dataset',dest='data',default='val', help='dates to be used: test/val')
    parser.add_argument('-p','--previous_days',dest='prev',type=int,default=0,help='amount of previous days')
    parser.add_argument('-n','--name',dest='name',type=str,help='result file')
    parser.add_argument('--normalize',dest='norm',type=bool,default=True,help='result file')
    parser.add_argument('--use_classifier',dest='classifier',type=bool,default=False,help='use classifier True/False')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    if params['comp']:
        compare(params)
    else:
        run_regression(params)

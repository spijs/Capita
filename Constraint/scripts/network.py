__author__ = 'spijs'

from createdatasubsets import getData
from sknn.mlp import *
import argparse
import pickle
from Regressor import Regressor as Reg
from SVMRegressor import SVMRegressor, LinearRegressor
class Network(Reg):

    def __init__(self, nbOfLayers, learning_rate,nb_iter,valid_input,valid_output,hidden):
        layers = []
        for i in range(nbOfLayers-1):
            layers.append(Layer("Tanh",name='hidden'+str(i),units=hidden))
        layers.append(Layer("Linear", name = 'output', units = 48))
        self.nn = Regressor(
            layers = layers,
            learning_rate=learning_rate,
            n_iter=nb_iter,
            valid_set=(valid_input,valid_output),
            verbose=True
        )

    def train(self, train, correct):
        self.nn.fit(train, correct)

    def test(self, test,correct):
        result =  self.nn.predict(test)
        for i in range(len(test)):
            for j in range(len(test[i])):
                print 'Predicted: %f, Correct value: %f' % (result[i][j],correct[i][j])

def run_regression(params):
    train_x,train_y = getData('train')
    print train_x.shape
    print train_y.shape
    val_x,val_y = getData('val')
    print val_x.shape
    print val_y.shape
    test_x,test_y = getData('test')
    if params['type']=='network':
        reg = Network(params['layers'],params['learning_rate'],params['iterations'],val_x,val_y,params['hidden'])
    elif params['type']=='svm':
        reg = SVMRegressor()
    else:
        reg = LinearRegressor()
    reg.train(train_x,train_y)
    reg.test(test_x,test_y)
    pickle.dump(reg,open('learned_network.p','wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--learning_rate', dest='learning_rate',type=float, default=.01, help='learning rate to be used')
    parser.add_argument('-hi', '--hidden', dest='hidden', type=int, default=256, help='Number of nodes in hidden layer')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default= 2000, help='Number of iterations for training the network')
    parser.add_argument('-l', '--layers',dest='layers',type=int, default=5, help='number of hidden layers used')
    parser.add_argument('-t', '--type', dest='type', type=str, default='network',help = 'type of regression used')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    run_regression(params)

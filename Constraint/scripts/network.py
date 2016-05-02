__author__ = 'spijs'

from createdatasubsets import getData
from sknn.mlp import *
import argparse
import pickle
class Network:

    def __init__(self, nbOfLayers, learning_rate,nb_iter,valid_input,valid_output,hidden):
        layers = []
        for i in range(nbOfLayers):
            layers.append(Layer("Sigmoid",name='hidden'+str(i),units=hidden))
        self.nn = Regressor(
            layers = layers,
            learning_rate=learning_rate,
            nb_iter=nb_iter,
            valid_set=zip(valid_input,valid_output),
            verbose=True
        )

    def train(self, train, correct):
        self.nn.fit(train, correct)

    def test(self, train):
        return self.nn.predict(train)


def run_network(params):
    train_x,train_y = getData('train')
    val_x,val_y = getData('val')
    test_x,test_y = getData('test')
    nn = Network(params['layers'],params['learning_rate'],params['iterations'],val_x,val_y,params['hidden'])
    nn.train(train_x,train_y)
    pickle.dump(nn,open('learned_network.p','wb'))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--learning_rate', dest='learning_rate',type=int, default=.01, help='learning rate to be used')
    parser.add_argument('-hi', '--hidden', dest='hidden', type=int, default=256, help='Number of nodes in hidden layer')
    parser.add_argument('-i', '--iterations', dest='iterations', type=int, default= 10000, help='Number of iterations for training the network')
    parser.add_argument('-l', '--layers',dest='layers',type=int, default=5, help='number of hidden layers used')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    run_network(params)

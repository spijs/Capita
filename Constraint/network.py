__author__ = 'spijs'

from sknn.mlp import *

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




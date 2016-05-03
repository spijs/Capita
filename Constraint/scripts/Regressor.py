__author__ = 'spijs'

from abc import ABCMeta,abstractmethod

class Regressor:
    __metaclass__ = ABCMeta

Regressor.register(tuple)

assert issubclass(tuple, Regressor)
assert isinstance((), Regressor)

@abstractmethod
def train(self, train, correct):
    pass

@abstractmethod
def test(self,test):
    pass



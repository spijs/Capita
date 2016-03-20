__author__ = 'spijs'

from random import randint
import numpy as np

class Generator:

    def __init__(self):
        self.weekend_mean = 3
        self.weekend_dev = 2
        self.week_mean = 5
        self.week_dev  = 1
        self.t = 7

    def generate_general(self):

        d_min = randint(0,self.t)
        d_max = randint(d_min,self.t)
        o_min = randint(0,self.t)
        o_max = randint(o_min,self.t)
        b = self.generate_array(self.t)

        print ('t = %s, d^- = %s, d^+ = %s, o^- = %s, o^+ = %s' % (str(self.t), str(d_min), str(d_max), str(o_min),str(o_max)))
        print ('b = %s' % b)

    def generate_cyclic(self):
        ass = self.generate_array(self.t,1)
        b = self.generate_array(self.t)
        print('assignment = %s' % ass)
        print('b = %s' % b)

    def generate_array(self,t):
        b = []
        for i in range(5):
            b.append(round(np.asarray(np.random.normal(self.week_mean,self.week_dev,1))[0],0))
        for i in range(t-5):
            b.append(round(np.asarray(np.random.normal(self.weekend_mean,self.weekend_dev,1))[0],0))
        return b

if __name__ == "__main__":
    g = Generator()
    g.generate_general()
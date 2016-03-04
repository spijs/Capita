__author__ = 'spijs'

from random import randint

class Generator:
    max_t = 50
    max_demand = 5

    def __init__(self):
        self.t = randint(0,self.max_t)

    def generate_general(self):

        d_min = randint(0,self.t)
        d_max = randint(d_min,self.t)
        o_min = randint(0,self.t)
        o_max = randint(o_min,self.t)
        b = self.generate_array(self.t,self.max_demand)

        print ('t = %s, d^- = %s, d^+ = %s, o^- = %s, o^+ = %s' % (str(self.t), str(d_min), str(d_max), str(o_min),str(o_max)))
        print ('b = %s' % b)

    def generate_cyclic(self):
        ass = self.generate_array(self.t,1)
        b = self.generate_array(self.t,self.max_demand)
        print('assignment = %s' % ass)
        print('b = %s' % b)

    def generate_array(self,t,max):
        b = []
        for i in range(t):
            b.append(randint(0,max))
        return b
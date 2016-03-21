__author__ = 'spijs'

from random import randint
import numpy as np
from time import *
from Solution import *

class Generator:

    def __init__(self):
        self.weekend_mean = 3
        self.weekend_dev = 2
        self.week_mean = 5
        self.week_dev  = 1
        self.t = 7

    def generate_general(self):
        print "GENERATING NEW PROBLEM"
        d_min = self.t
        o_min = self.t
        while d_min + o_min > self.t:
            d_min = randint(1,self.t)
            d_max = randint(d_min,self.t)
            o_min = randint(1,self.t)
            o_max = randint(o_min,self.t)
            b = self.generate_array(self.t)
        g = GeneralProblem(d_min, d_max, o_min, o_max, self.t, b, None)
        if len(g.get_random_solution(time()).employees) == 0:
            "Creating New Problem"
            return self.generate_general()
        else:
            g.print_problem()
            return g


    def generate_cyclic(self):
        ass = self.generate_array(self.t,1)
        b = self.generate_array(self.t)
        c = CyclicProblem(ass, b)
        c.print_problem()
        return c

    def generate_array(self,t):
        b = []
        for i in range(5):
            b.append(round(np.asarray(np.random.normal(self.week_mean,self.week_dev,1))[0],0))
        for i in range(t-5):
            b.append(round(np.asarray(np.random.normal(self.weekend_mean,self.weekend_dev,1))[0],0))
        return np.array(b)



class GeneralProblem:
    def __init__(self, d_min, d_max, o_min, o_max, t, b, initial_solution):
        self.d_min = d_min
        self.d_max = d_max
        self.o_min = o_min
        self.o_max = o_max
        self.t = t
        self.b = b
        self.initial_solution = initial_solution

    def check_solution(self, solution):
        for e in solution.employees:
            if not (len(e) == self.t):
                print "length is not ok"
                return False
            if not self.checkDays(e):
                return False
        return self.check_sum_solution(solution)

    def check_sum_solution(self, solution):
        sum = np.zeros(self.t)
        for e in solution.employees:
            sum = sum + e
        for i in range(len(sum)):
            if sum[i] < self.b[i]:
                return False
        return True

    '''
    Returns true if the assignment for the given employee is valid.
    '''
    def checkDays(self,  employee):
        current_day = employee[0]
        current_count = 0
        for day in employee:
            if day == current_day:
                current_count+= 1
                # print "current count: " + str(current_count)
            else:
                if not self.checkBoundaries(current_count, current_day):
                    return False
                current_count = 1
                current_day = day
        return self.checkBoundaries(current_count, current_day)

    def get_random_solution(self, starttime = time()):
        print "trying random solution"
        e = self.generateRandomEmployee(time())
        if len(e) == 0:
            print "NO solution"
            return []
        else:
            first = [e]
            solution = first
            while not self.check_solution(Solution(solution)):
                if time() - starttime > 5:
                    return Solution([])
                else:
                    solution = np.append(solution, [self.generateRandomEmployee(time())], axis = 0)
        return Solution(solution)


    def generateRandomEmployee(self, start):
        candidate = np.random.randint(2, size=self.t)
        while not self.checkDays(candidate):
            if time() - start > 5:
                return []
            else:
            # print "Candidate is not ok"
                candidate = np.random.randint(2, size=self.t)
        #print "Candidate: " + str(candidate)
        return candidate

    '''
    Check whether the given number is between boudaries for on-off days, based on the day parameter (0 = off, 1 = on)
    '''
    def checkBoundaries(self, number, day):
        if day == 0:
            # print "Number of off days: " + str(number)
            return number >= self.o_min and number <= self.o_max
        else:
            # "Number of on days: " + str(number)
            return number >= self.d_min and number <= self.d_max


    def print_problem(self):
        print ('t = %s, d^- = %s, d^+ = %s, o^- = %s, o^+ = %s' % (str(self.t), str(self.d_min), str(self.d_max), str(self.o_min),str(self.o_max)))
        print ('b = %s' % self.b)


class CyclicProblem:
    def __init__(self, ass, b):
        self.ass = ass
        self.b = b

    def check_solution(self, solution):
        return self.check_sum_solution(solution)

    def check_sum_solution(self, solution):
        sum = np.zeros(self.t)
        for e in solution:
            sum = sum + e
        for i in range(len(sum)):
            if sum[i] < self.b[i]:
                return False
        return True

    def print_problem(self):
        print('assignment = %s' % self.ass)
        print('b = %s' % self.b)

if __name__ == "__main__":
    g = Generator()
    p = g.generate_general()
    s = p.get_random_solution(time())
    s.print_solution()
    # print str(s.get_cost())
    # p = GeneralProblem(1, 4, 1, 2, 7, [3,3,3,3,3,2,2])
    # print str(p.checkDays([0,0,0,0,0,0,0]))
    # r = p.get_random_solution()
    # sol = Solution([[1,1,1,1,0,1,0],[0,1,1,1,1,0,1], [1,1,1,0,1,1,0], [1,0,1,1,1,0,1]])
    # sol = Solution([[1,1,1,1,1,1,1],[0,1,1,1,1,0,1], [1,1,1,0,1,1,0], [1,0,1,1,1,0,1]])
    # print p.check_solution(sol)
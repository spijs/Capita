__author__ = 'spijs'

from random import randint
import numpy as np
from time import *
from Solution import *
from CyclicSolution import *
import Logger
class Generator:

    def __init__(self):
        self.weekend_mean = 5
        self.weekend_dev = 2
        self.week_mean = 8
        self.week_dev  = 3
        self.t = 7
    '''
    Generate a random, solvable GDODOSP problem. Try to generate a random solution. If the random solution fails
    to be generated, try to generate a different problem.
    '''
    def generate_general(self):
        print("GENERATING NEW PROBLEM")
        d_min = self.t
        o_min = self.t
        while d_min + o_min > self.t:
            d_min = randint(1,self.t)
            o_min = randint(1,self.t)
        d_max = randint(d_min,self.t)
        o_max = randint(o_min,self.t)
        b = self.generate_array(self.t)
        g = GeneralProblem(d_min, d_max, o_min, o_max, self.t, b, None)
        solution = g.get_random_solution(time())
        if len(solution.employees) == 0:
            print "Creating New Problem"
            return self.generate_general()
        else:
            g.initial_solution = solution
            print g.to_string()
            return g

    '''
    Generate a random CDODOSP
    '''
    def generate_cyclic(self):
        b = self.generate_array(self.t)
        ass = np.random.randint(2, size=self.t)
        c = CyclicProblem(ass, b, None)
        sol = c.get_random_solution()
        c.initial_solution = sol
        print c.to_string()
        print sol.to_string()
        return c

    '''
    Generate an array of length t, with the first five days sampled from the "week" normal distribution,
    and the rest of the days from the "weekend" normal distribution
    '''
    def generate_array(self,t):
        b = []
        for i in range(5):
            b.append(max(0,round(np.asarray(np.random.normal(self.week_mean,self.week_dev,1))[0],0)))
        for i in range(t-5):
            b.append(max(0,round(np.asarray(np.random.normal(self.weekend_mean,self.weekend_dev,1))[0],0)))
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

    '''
    Return the initial solution stored in the problem
    '''
    def get_initial_solution(self):
        return self.initial_solution

    '''
    Check whether the solution is a correct one for this problem. The checks are based on the length of each
    of the employees, whether or not the on/off lenght constraints are satisfied, and whether or not the demand
    vector is satisfied
    '''
    def check_solution(self, solution):
        for e in solution.employees:
            if not (len(e) == self.t):
                Logger.write("length is not ok")
                return False
            if not self.checkDays(e):
                return False
        return self.check_sum_solution(solution)

    '''
    Return true if and only if the given solution satisfies the demand vector of this problem
    '''
    def check_sum_solution(self, solution):
        sum = np.zeros(self.t)
        for e in solution.employees:
            sum = sum + e
        for i in range(len(sum)):
            if sum[i] < self.b[i]:
                return False
        return True

    '''
    Returns true if the assignment for the given employee is valid, according to the on/off length constraints
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

    '''
    Return a random solution for this problem. If the solution cannot be generated in time, we assume the
    problem is not solvable and an empty solution is returned.
    '''
    def get_random_solution(self, starttime = time()):
        Logger.write("trying random solution")
        e = self.generateRandomEmployee(time())
        if len(e) == 0:
            Logger.write("NO solution")
            return Solution([])
        else:
            first = [e]
            solution = first
            while not self.check_solution(Solution(solution)):
                if time() - starttime > 2:
                    return Solution([])
                else:
                    solution = np.append(solution, [self.generateRandomEmployee(time())], axis = 0)
        return Solution(solution)

    '''
    Return a random employee, satisfying the on/off length constraints. If no valid employee is created in time,
    return an empty list.
    '''
    def generateRandomEmployee(self, start):
        candidate = np.random.randint(2, size=self.t)
        while not self.checkDays(candidate):
            if time() - start > 2:
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


    def to_string(self):
        return 't = %s, d^- = %s, d^+ = %s, o^- = %s, o^+ = %s\nb = %s\n' % (str(self.t), str(self.d_min), str(self.d_max), str(self.o_min),str(self.o_max),str(self.b))


class CyclicProblem:
    def __init__(self, ass, b, initial_solution):
        self.ass = ass
        self.b = b
        self.initial_solution = initial_solution
    '''
    Check whether or not the given solution is valid for this problem
    '''
    def check_solution(self, solution):
        return self.check_sum_solution(solution)

    '''
    Generate a random solution for this problem. This is done by adding random permutations of the assignment vector
    until the demand vector is satisfied.
    '''

    def get_random_solution(self, starttime = time()):
        first = self.randomShift()
        solution = [first]
        while not self.check_solution(CyclicSolution(solution)):
            solution = np.append(solution, [self.randomShift()], axis = 0)
        return CyclicSolution(solution)

    '''
    Return the initial solution stored in the problem
    '''
    def get_initial_solution(self):
        return self.initial_solution

    '''
    Return a random shift of the assignment vector
    '''
    def randomShift(self):
        i = randint(0,1+len(self.b))
        return np.roll(self.ass, i)

    '''
    Return true is the sum of all employees satisfies the demand vector
    '''
    def check_sum_solution(self, solution):
        sum = np.zeros(len(self.b))
        for e in solution.employees:
            sum = sum + e
        for i in range(len(sum)):
            if sum[i] < self.b[i]:
                return False
        return True

    def to_string(self):
        return ('assignment = %s\nb = %s' % (self.ass,self.b))


if __name__ == "__main__":
    g = Generator()
    p = g.generate_cyclic()
    s = p.get_random_solution(time())
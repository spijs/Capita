import numpy as np
from time import *

class CyclicSolution:
    def __init__(self,employees):
        self.employees = employees

    def to_string(self):
        return str(self.employees)

    '''
    Return the relative cost of this solution. This is based on the optimal cost
    '''
    def get_cost(self,type,problem):
        if type=='number_employees':
            max = np.max(problem.b)
            value = len(self.employees)
        else:
            max = sum(problem.b)
            value = np.sum(self.employees)
        return (value-max)/max

    '''
    Return a new solution based on this one. This can be done by removing an employee, or by shifting a number
    of employees
    '''
    def step(self, problem, percentage, nb_shifts, start = time()):
        new_solution = 0
        i = 0
        while new_solution == 0:
            if time() - start > 20:
                return False
            i+=1
            #print 'Step currently at try: %i' %i
            copy_emp = np.copy(self.employees)
            which_step = np.random.rand()*100
            if which_step < percentage: # verwijder random employee
                which_emp = np.random.randint(len(copy_emp))
                new_emp = np.delete(copy_emp, which_emp, axis = 0)
                if problem.check_solution(CyclicSolution(new_emp)):
                    #print 'removed employee after %i steps' % i
                    new_solution = CyclicSolution(new_emp)
            else: # shift x employees over een random lengte
                if not nb_shifts:
                    nb_shifts = np.random.randint(len(self.employees))+1
                for i in range(nb_shifts):
                    which_emp = np.random.randint(len(self.employees))
                    copy_emp = np.copy(self.employees)
                    emp = copy_emp[which_emp]
                    new_emps = copy_emp
                    amount = np.random.randint(0,len(emp)+1)
                    new_emps = self.shift_one_emp(new_emps, which_emp, amount)
                if problem.check_solution(CyclicSolution(new_emps)):
                    #print 'Shifted one employee after %i steps' % i
                    new_solution = CyclicSolution(new_emps)
        return new_solution

    '''
    Given a list of employees, an index and an amount, return the list with the correct employee
    shifted over the given amount
    '''
    def shift_one_emp(self, employees, which_emp, amount):
        emp = employees[which_emp]
        emp = np.roll(emp, amount)
        employees[which_emp] = emp
        return employees

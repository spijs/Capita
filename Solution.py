import numpy as np
import Logger
class Solution:
    def __init__(self, employees):
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
    Return a new solution based on this one. This can be done by removing an employee, or by performing a number of
    swaps in a certain employee
    '''
    def step(self, problem, percentage, nbChanges):
        new_solution = 0
        i = 0
        while new_solution == 0:
            i+=1
            #print 'Step currently at try: %i' %i
            copy_emp = np.copy(self.employees)
            which_step = np.random.rand()*100
            if which_step < percentage:
                which_emp = np.random.randint(len(copy_emp))
                new_emp = np.delete(copy_emp, which_emp, axis = 0)
                if problem.check_solution(Solution(new_emp)):
                    #print 'removed employee after %i steps' % i
                    new_solution = Solution(new_emp)
            else:
                which_emp = np.random.randint(len(self.employees))
                copy_emp = np.copy(self.employees)
                emp = copy_emp[which_emp]
                new_emps = copy_emp
                if not nbChanges:
                    nbChanges = np.random.randint(len(emp))
                for _ in range(nbChanges):
                    which_day = np.random.randint(len(emp))
                    emp[which_day] = (emp[which_day] + 1) % 2
                    new_emps[which_emp] = emp
                if problem.check_solution(Solution(new_emps)):
                    #print 'switched value after %i steps' % i
                    new_solution = Solution(new_emps)
        return new_solution
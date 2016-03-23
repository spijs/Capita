import numpy as np

class Solution:
    def __init__(self, employees):
        self.employees = employees

    def print_solution(self):
        print str(self.employees)

    def get_cost(self):
        #return len(self.employees)
        return np.sum(self.employees)
    def step(self, problem, percentage):
        new_solution = 0
        i = 0
        while new_solution == 0:
            i+=1
            #print 'Step currently at try: %i' %i
            copy_emp = np.copy(self.employees)
            which_step = np.random.rand()*100
            if which_step < percentage: # verwijder random employee
                which_emp = np.random.randint(len(copy_emp))
                new_emp = np.delete(copy_emp, which_emp, axis = 0)
                if problem.check_solution(Solution(new_emp)):
                    print 'removed employee after %i steps' % i
                    new_solution = Solution(new_emp)
            else: # switch een random aantal 0 of 1 binnen 1 employee
                which_emp = np.random.randint(len(self.employees))
                copy_emp = np.copy(self.employees)
                emp = copy_emp[which_emp]
                new_emps = copy_emp
                nb_days = np.random.randint(len(emp))
                for _ in range(nb_days):
                    which_day = np.random.randint(len(emp))
                    emp[which_day] = (emp[which_day] + 1) % 2
                    new_emps[which_emp] = emp
                if problem.check_solution(Solution(new_emps)):
                    print 'switched value after %i steps' % i
                    new_solution = Solution(new_emps)
        return new_solution
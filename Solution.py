import numpy as np

class Solution:
    def __init__(self, employees):
        self.employees = employees

    def print_solution(self):
        print str(self.employees)

    def get_cost(self):
        return len(self.employees)

    def step(self, problem, percentage):
        new_solution = 0
        while new_solution == 0:
            which_step = np.random.rand()*100
            if which_step < percentage: # verwijder random employee
                which_emp = np.random.randint(len(self.employees))
                new_emp = np.delete(self.employees, which_emp)
                if problem.check_solution(Solution(new_emp)):
                    new_solution = Solution(new_emp)
            else: # switch een random 0 of 1
                which_emp = np.random.randint(len(self.employees))
                emp = self.employees[which_emp]
                new_emps = self.employees
                which_day = np.random.randint(len(emp))
                emp[which_day] = (emp[which_day] + 1) % 2
                new_emps[which_emp] = emp
                if problem.check_solution(Solution(new_emps)):
                    new_solution = Solution(new_emps)
        return new_solution
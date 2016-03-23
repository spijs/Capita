import numpy as np

class CyclicSolution:
    def __init__(self,employees):
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
                if problem.check_solution(CyclicSolution(new_emp)):
                    print 'removed employee after %i steps' % i
                    new_solution = CyclicSolution(new_emp)
            else: # shift x employees over een random  lengte
                nb_shifts = np.random.randint(len(self.employees))+1
                for i in range(nb_shifts):
                    which_emp = np.random.randint(len(self.employees))
                    copy_emp = np.copy(self.employees)
                    emp = copy_emp[which_emp]
                    new_emps = copy_emp
                    amount = np.random.randint(0,len(emp)+1)
                    new_emps = self.shift_one_emp(new_emps, which_emp, amount)
                if problem.check_solution(CyclicSolution(new_emps)):
                    print 'Shifted one employee after %i steps' % i
                    new_solution = CyclicSolution(new_emps)
        return new_solution

    def shift_one_emp(self, employees, which_emp, amount):
        emp = employees[which_emp]
        emp = np.roll(emp, amount)
        employees[which_emp] = emp
        return employees

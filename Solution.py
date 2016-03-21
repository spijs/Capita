class Solution:
    def __init__(self, employees):
        self.employees = employees

    def print_solution(self):
        print str(self.employees)

    def get_cost(self):
        return len(self.employees)
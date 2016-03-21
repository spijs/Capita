__author__ = 'spijs'

from Generator import Generator
import argparse

def main(Lfa,it,percentage):
    g = Generator()
    p = g.generate_general()
    global max_it
    max_it = it
    LAHC_algorithm(p,Lfa,percentage)

def LAHC_algorithm(problem,Lfa,percentage):
    s  = problem.get_random_solution()
    print ('got initial solution')
    c = s.get_cost()
    print 'inital cost :%s' % str(c)
    f = [None]*Lfa
    I = 0
    for k in range(0,Lfa):
        f[k]= c
    while not stop_condition(I):
        print 'Current step: %s' % str(I)
        s_new = s.step(problem,percentage)
        c_new = s_new.get_cost()
        v = I % Lfa
        if c_new <= f[v] or c_new<c:            # True =  Accept
            print ('added new solution: %s') % str(s.print_solution)
            s = s_new
            c = c_new
        f[v] = c
        I += 1
    print ("Found solution: ")
    s.print_solution()


def stop_condition(nbIt):
    if nbIt > max_it:
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #settings
    parser.add_argument('-n', '--number', dest='Lfa', default=1, help='Lfa value (number of memorized results)')
    parser.add_argument('-i', '--iterations', dest='it', default=1000, help='number of iterations as stopping condition')
    parser.add_argument('-p', '--percentage', dest='p', default=50, help='Percentage of type 1 steps')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params['Lfa'],params['it'],params['p'])

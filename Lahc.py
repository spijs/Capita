__author__ = 'spijs'

from Generator import Generator
import argparse

def main(Lfa):
    g = Generator()
    p = g.generate_general(Lfa)
    LAHC_algorithm(p,Lfa)

def LAHC_algorithm(problem,Lfa):
    s  = problem.get_random_solution()
    c = s.get_cost()
    f = []
    I = 0
    for k in range(0,Lfa):
        f[k]= c
    while not stop_condition():
        s_new = s.step()
        c_new = s_new.get_cost()
        v = I % Lfa
        if c_new <= f[v] or c_new<c:            # True =  Accept
            s = s_new
            c = c_new
        f[v] = c
        I += 1
    print ("Found solution: ")
    s.print_solution()


def stop_condition():
    return False #TODO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #settings
    parser.add_argument('-n', '--number', dest='Lfa', default=1, help='Lfa value (number of memorized results)')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params['Lfa'])
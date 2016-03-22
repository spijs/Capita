__author__ = 'spijs'

from Generator import Generator
import argparse
import numpy as np
import pickle

try:
    instances = pickle.load(open('Instances/instances'))
except:
    instances ={}

def main(Lfa,it,percentage,instance):
    if(instance):
        p = get_instance(instance)
    else:
        g = Generator()
        p = g.generate_general()
    global max_it
    max_it = it
    LAHC_algorithm(p,Lfa,percentage)

def get_instance(instance):
    return instances[instance]

def create_instances(number_of_instances):
    instances = {}
    for i in range(number_of_instances):
        g = Generator()
        p = g.generate_general()
        instances[i]=p
    f = open('Instances/instances','w')
    pickle.dump(instances,f)


def LAHC_algorithm(problem,Lfa,percentage):
    s  = problem.get_initial_solution()
    print 'got initial solution:'
    s.print_solution()
    c = s.get_cost()
    print 'inital cost :%s' % str(c)
    f = [None]*Lfa
    I = 0
    for k in range(0,Lfa):
        f[k]= c
    last_change = 0
    while not stop_condition(I):
        print 'Current I: %s' % str(I)
        s_new = s.step(problem,percentage)
        c_new = s_new.get_cost()
        v = I % Lfa
        if c_new <= f[v] or c_new<c:            # True =  Accept
            print ('added new solution with cost %i:' % c_new)
            last_change = I
            s_new.print_solution()
            s = s_new
            c = c_new
        f[v] = c
        I += 1
    print ("Found solution: ")
    s.print_solution()
    print("for problem: ")
    problem.print_problem()
    print("with cost: %i and number of hours necessary: %i" %(c,sum(problem.b)))
    print("last change in iteration %i" % last_change)


def stop_condition(nbIt):
    if nbIt > max_it:
        return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #settings
    parser.add_argument('-n', '--number', dest='Lfa',type=int, default=1, help='Lfa value (number of memorized results)')
    parser.add_argument('-it', '--iterations', dest='it',type=int, default=1000, help='number of iterations as stopping condition')
    parser.add_argument('-p', '--percentage', dest='p',type=int, default=0, help='Percentage of type 1 steps')
    parser.add_argument('-i', '--instance', dest='inst',type=int,default=1, help = 'Instance number to be evaluated')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    main(params['Lfa'],params['it'],params['p'],params['i'])

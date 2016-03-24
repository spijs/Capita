__author__ = 'spijs'

from Generator import Generator
import argparse
import numpy as np
import pickle
import re
import Logger
import sys

try:
    instances = pickle.load(open('Instances/instances'))
except:
    instances = pickle.load(open('../Instances/instances'))

def main(Lfa,it,percentage,instance,cost):
    if(instance):
        p = get_instance(instance)
    else:
        g = Generator()
        p = g.generate_general()
    global max_it
    max_it = it
    LAHC_algorithm(p,Lfa,percentage,cost)

def get_instance(instance):
    return instances[instance]

def create_instances(number_of_instances):
    instances = {}
    name_file = open('training_instance_list-01.txt','w')
    for i in range(number_of_instances):
        g = Generator()
        p = g.generate_general()
        instances[i]=p
        name_file.write(str(i)+'\n')
    f = open('Instances/instances','w')
    pickle.dump(instances,f)

def LAHC_algorithm(problem,Lfa,percentage,cost):
    s  = problem.get_random_solution()
    Logger.write('got initial solution:',1)
    Logger.write(s.to_string(),1)
    c = s.get_cost(cost,problem)
    Logger.write('inital cost :%s' % str(c))
    f = [None]*Lfa
    I = 0
    for k in range(0,Lfa):
        f[k]= c
    last_change = 0
    best = None
    best_cost = 2
    while not stop_condition(I,best_cost):
        if I % 10 == 0:
            update_progress(100*I/max_it)
        #print 'Current I: %s' % str(I)
        s_new = s.step(problem,percentage)
        c_new = s_new.get_cost(cost,problem)
        v = I % Lfa
        if c_new <= f[v] or c_new<c:            # True =  Accept
            Logger.write('added new solution with cost %i:\n' % c_new,1)
            last_change = I
            #s_new.print_solution()
            s = s_new
            c = c_new
        if not best or c_new < best_cost:
            best_cost = c
            best = s
        f[v] = c
        I += 1
    Logger.write("\nFound solution: \n")
    Logger.write(best.to_string())
    Logger.write("\nfor problem: \n")
    Logger.write(problem.to_string())
    Logger.write("\nwith cost: %i and number of hours necessary: %i\n" %(np.sum(best.employees),sum(problem.b)))
    Logger.write("last change in iteration %i\n" % last_change)
    sys.stdout.write('Best %f' % (best_cost))

def update_progress(progress):
    spaces = (100-progress)/2 * ' '
    Logger.write('\r[{0}{2}]  {1}%'.format('#'*(progress/2), progress,spaces))
    Logger.flush()

def stop_condition(nbIt,best_cost):
    print best_cost
    if nbIt > max_it or best_cost == 0:
        return True
    return False

def parse_path(input):
    return int(re.findall('\d+', input)[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #settings
    parser.add_argument('-n', '--number', dest='Lfa',type=int, default=1, help='Lfa value (number of memorized results)')
    parser.add_argument('-it', '--iterations', dest='it',type=int, default=1000, help='number of iterations as stopping condition')
    parser.add_argument('-p', '--percentage', dest='p',type=int, default=50, help='Percentage of type 1 steps')
    parser.add_argument('-i', '--instance', dest='inst',type=int,default=0, help = 'Instance number to be evaluated')
    parser.add_argument('-v','--verbosity',dest='verbosity',type=int,default=0,help='verbosity: 0 only cost returned, 1 more prints,2 all prints')
    parser.add_argument('-is','--instance_string',dest='string',type=str, help='string containing the instance number')
    parser.add_argument('-c','--cost', dest='cost',type=str,default='total_work',help='cost function to be used: total_work(default) or number_employees')
    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    Logger.init_logger(params['verbosity'])
    if not params['inst']:
        params['inst'] = parse_path(params['string'])
    main(params['Lfa'],params['it'],params['p'],params['inst'],params['cost'])

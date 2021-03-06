#!/usr/bin/env python

MZNSOLUTIONBASENAME = "minizinc.out"

import sys
import os
import argparse
import time as ttime
import glob
import pickle
import json

cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd,'scripts'))
sys.path.append(os.path.join(cwd,'data'))
from prices_data import *


def instance2arr(instance):
    tasks = []
    for t in instance.day.tasks:
        tasks.append( {'taskid':t.taskid, 'machid':t.machineid, 'start':t.start} )
    return tasks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run and check a MZN model in ICON challenge data")
    parser.add_argument("file_mzn")
    parser.add_argument("--out", help="file to write the JSON output to", default="out.json")
    parser.add_argument("--mzn-solver", help="the mzn solver to use (mzn-g12mip or mzn-gecode for example)", default='mzn-g12mip')
    parser.add_argument("--mzn-dir", help="optionally, if the binaries are not on your PATH, set this to the directory of the MiniZinc IDE", default="")
    parser.add_argument("--tmp", help="temp directory (default = automatically generated)")
    parser.add_argument("-c", "--historic-days", help="How many historic days to learn from", default=30, type=int)
    # debugging options:
    parser.add_argument("-p", "--print-pretty", help="pretty print the machines and tasks", action="store_true")
    parser.add_argument("-v", help="verbosity (0,1,2 or 3)", type=int, default=0)
    parser.add_argument("--print-output", help="print the output of minizinc", action="store_true")
    parser.add_argument("--tmp-keep", help="keep created temp subdir", action="store_true")
    parser.add_argument("--network", help="regressor to be used")
    parser.add_argument("--factor", help="whether or not to scale prediction based on last sechduling", type=bool, default=False)
    parser.add_argument("--testload", help = "if True, use only load1, for testing", type=bool, default=False)
    args = parser.parse_args()

    dir_load = '../'
    datafile = '../data/cleanData.csv';
    dat = load_prices(datafile)


    if args.testload: # used for testing purposes
        benchmarks = {'load1': ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01'],
                 }
    else:
        benchmarks = {'load1': ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01'],
                      'load8': ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01'],
                 }

    cwd=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(cwd,'..'))

    mymethod = __import__('runitall')

    res = dict()
    curr = 0

    network = pickle.load(open(args.network, 'rb'))
    networkpred, networkcorrect = network.test('test')
    classifications = pickle.load(open('../results/classification', 'r'))
    classifications = np.array(classifications)
    classifications = np.split(classifications, 4)

    for load, startdays in benchmarks.iteritems():
        total = 0
        res[load] = dict()
        globpatt = os.path.join(dir_load, load, 'day*.txt')
        f_instances = sorted(glob.glob(globpatt))

        for day_str in startdays:
            preds = networkpred[curr]  # per day an array containing a prediction for each PeriodOfDay
            actuals = networkcorrect[curr]
            classes = classifications[curr]
            preds = np.split(preds, 14)
            actuals = np.split(actuals, 14)
            classes = np.split(classes, 14)


            res[load][day_str] = dict()
            day = datetime.strptime(day_str, '%Y-%m-%d').date()

            # do predictions and get schedule instances in triples like:
            # [('load1/day01.txt', '2012-02-01', InstanceObject), ...]
            time_start = ttime.time()
            run_triples = mymethod.run(f_instances, day, dat, preds, actuals, classes, args)
            runtime = (ttime.time() - time_start)

            # add to res
            for (f_inst, day, instance) in run_triples:
                f_name = os.path.basename(f_inst)
                res[load][day_str][f_name] = instance2arr(instance)


            # compute total actual cost (and time)
            tot_act = 0
            for (f_inst, dayx, instance) in run_triples:
                instance.compute_costs()
                tot_act += instance.day.cj_act
            total = total + tot_act

            # write results to disk
            resultfile = open('../results/' + load + day_str + args.network.split('/')[-1] + '.txt', 'w+')
            resultfile.write(str(tot_act))
            resultfile.close()
            print "%s from %s, linear: total actual cost: %.1f (runtime: %.2f)"%(load, day_str, tot_act, runtime)

            curr = (curr + 1) % 4

        # write total cost for this load to disk
        totalresfile = open('../results/' + load+'_total' + args.network.split('/')[-1]+ '.txt', 'w+')
        totalresfile.write(str(total))
        totalresfile.close()
    with open(args.out, 'w') as f_out:
        json.dump(res, f_out)

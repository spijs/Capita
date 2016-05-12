#!/usr/bin/env python

MZNSOLUTIONBASENAME = "minizinc.out"

import sys
import os
import shutil
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
    args = parser.parse_args()

    dir_load = '../'
    datafile = '../data/cleanData.csv';
    dat = load_prices(datafile)


    # benchmark setting, you can choose one of load1/load8 instead of both too (but always all start days)
    benchmarks = {'load1': ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01'],
#                  'load8': ['2013-02-01', '2013-05-01', '2013-08-01', '2013-11-01'],
                 }

    cwd=os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.join(cwd,'..'))
    # replace 'mzn-prototype' by your method
    mymethod = __import__('runitall')

    # {'load1': {'2013-12-01': {'day01.txt': [{'taskid': 1, 'machid': 1, 'start': 1},
    #                                         {'taskid': ... }],
    #                           'day02.txt': [...],
    #                           ...
    #                          },
    #            '2013-05-01': {'day01.txt'...}
    #           },
    #  'load8': ...
    # }
    res = dict()
    curr = 0

    network = pickle.load(open(args.network, 'rb'))
    networkpred, networkcorrect = network.test('test')
    # print "pred shape ", networkpred.shape


    for load, startdays in benchmarks.iteritems():
        total = 0
        print "LOAD: ", load
        res[load] = dict()
        globpatt = os.path.join(dir_load, load, 'day*.txt')
        f_instances = sorted(glob.glob(globpatt))
        # print "F instances loaded: ", f_instances

        for day_str in startdays:
            preds = networkpred[curr]  # per day an array containing a prediction for each PeriodOfDay
            actuals = networkcorrect[curr]
            preds = np.split(preds, 14)
            actuals = np.split(actuals, 14)

            # print "shape preds ", np.array(preds).shape
            resultfile = open('../results/' + load + day_str+args.network.split('/')[-1]+'.txt', 'w+')
            res[load][day_str] = dict()
            day = datetime.strptime(day_str, '%Y-%m-%d').date()

            # do predictions and get schedule instances in triples like:
            # [('load1/day01.txt', '2012-02-01', InstanceObject), ...]
            time_start = ttime.time()
            run_triples = mymethod.run(f_instances, day, dat, preds, actuals, args=args)
            runtime = (ttime.time() - time_start)

            # add to res
            for (f_inst, day, instance) in run_triples:
                # TODO: check order of f_inst and subsequent days
                f_name = os.path.basename(f_inst)
                res[load][day_str][f_name] = instance2arr(instance)


            # compute total actual cost (and time)
            tot_act = 0
            for (f_inst, dayx, instance) in run_triples:
                instance.compute_costs()
                tot_act += instance.day.cj_act
            total = total + tot_act
            resultfile.write(str(tot_act))
            resultfile.close()
            print "%s from %s, linear: total actual cost: %.1f (runtime: %.2f)"%(load, day_str, tot_act, runtime)

            curr = (curr + 1) % 4
        totalresfile = open('../results/' + load+'_total' + args.network.split('/')[-1]+ '.txt', 'w+')
        print "~~~~~~~~~Total cost for this load ", str(total)
        totalresfile.write(str(total))
        totalresfile.close()
    with open(args.out, 'w') as f_out:
        json.dump(res, f_out)

#!/usr/bin/env python

MZNSOLUTIONBASENAME = "minizinc.out"

import sys
import os
import shutil
import argparse
import random
import subprocess
import tempfile
import time
import glob
import datetime
import pickle
from scripts.network import *

runcheck = __import__('mzn-runcheck')
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd,'scripts'))
from scripts.checker import *
import scripts.instance2dzn as i2dzn
import scripts.forecast2dzn as f2dzn
import scripts.checker_mzn as chkmzn
from scripts.prices_data import *
from scripts.createdatasubsets import *
from scripts.prices_regress import *
import numpy as np
from sklearn import linear_model


def _qflatten(L,a,I):
    for x in L:
        if isinstance(x,I): _qflatten(x,a,I)
        else: a(x)
def qflatten(L):
    R = []
    _qflatten(L,R.append,(list,tuple,np.ndarray))
    return np.array(R)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run and check a MZN model in ICON challenge data")
    parser.add_argument("file_mzn")
    parser.add_argument("file_instance", help="(can also be a directory to run everything matching 'day*.txt' in the directory)")
    parser.add_argument("--mzn-solver", help="the mzn solver to use (mzn-g12mip or mzn-gecode for example)", default='mzn-g12mip')
    parser.add_argument("--mzn-dir", help="optionally, if the binaries are not on your PATH, set this to the directory of the MiniZinc IDE", default="")
    parser.add_argument("--tmp", help="temp directory (default = automatically generated)")
    parser.add_argument("-d", "--day", help="Day to start from (in YYYY-MM-DD format)")
    parser.add_argument("-c", "--historic-days", help="How many historic days to learn from", default=30, type=int)
    # debugging options:
    parser.add_argument("-p", "--print-pretty", help="pretty print the machines and tasks", action="store_true")
    parser.add_argument("-v", help="verbosity (0,1,2 or 3)", type=int, default=1)
    parser.add_argument("--print-output", help="print the output of minizinc", action="store_true")
    parser.add_argument("--tmp-keep", help="keep created temp subdir", action="store_true")
    parser.add_argument("--testinstance", help="which test instance to use", type = int, default = 0)
    args = parser.parse_args()

    # if you want to hardcode the MiniZincIDE path for the binaries, here is a resonable place to do that
    #args.mzn_dir = "/home/tias/local/src/MiniZincIDE-2.0.13-bundle-linux-x86_64"

    datafile = 'data/cleanData.csv'
    dat = load_prices(datafile)

    day = None
    if args.day:
        day = datetime.strptime(args.day, '%Y-%m-%d').date()
    else:
        day = get_random_day(dat, args.historic_days)
    if args.v >= 1:
        print "First day:", day

    tmpdir = ""
    if args.tmp:
        tmpdir = args.tmp
        os.mkdir(args.tmp)
    else:
        tmpdir = tempfile.mkdtemp()

    # single or multiple instances
    f_instances = [args.file_instance]
    if os.path.isdir(args.file_instance):
        globpatt = os.path.join(args.file_instance, 'day*.txt')
        f_instances = sorted(glob.glob(globpatt))

    ##### data stuff
    os.chdir("./data")
    testset, testresults = getData('test')

    print "shape testset ", testset.shape
    print "shape test results", testresults.shape
    test_inst = args.testinstance
    os.chdir("..")
    # network prediction
    network = pickle.load(open("scripts/learned_network.p", 'rb'))
    os.chdir("./scripts")
    networkpred = network.test(testset)
    print "pred shape ", networkpred.shape
    os.chdir("..")

    preds = []  # per day an array containing a prediction for each PeriodOfDay
    preds = networkpred[test_inst]
    preds = np.split(preds, 14)
    print "shape preds ", np.array(preds).shape
    actuals = testresults[48*test_inst:(48*test_inst+14)] # also per day
    print "actuals shape ", np.array(actuals).shape


    # the scheduling
    tot_act = 0
    tot_time = 0
    for (i,f) in enumerate(f_instances):
        data_forecasts = preds[i].tolist()
        data_actual = actuals[i].tolist()
        #print "data actual ", data_actual
        #print "data actual shapa ", np.array(data_actual).shape
        (timing, out) = runcheck.mzn_run(args.file_mzn, f, data_forecasts,
                                tmpdir, mzn_dir=args.mzn_dir,
                                print_output=args.print_output,
                                verbose=args.v-1)
        instance = runcheck.mzn_toInstance(f, out, data_forecasts,
                                  data_actual=data_actual,
                                  pretty_print=args.print_pretty,
                                  verbose=args.v-1)
        if args.v >= 1:
            # csv print:
            if i == 0:
                # an ugly hack, print more suited header here
                print "scheduling_scenario; date; cost_forecast; cost_actual; runtime"
            today = day + timedelta(i)
            chkmzn.print_instance_csv(f, today.__str__(), instance, timing=timing, header=False)
        instance.compute_costs()
        tot_act += instance.day.cj_act
        tot_time += timing

    print "%s, linear: Total time: %.2f -- total actual cost: %.1f"%(args.file_instance, tot_time, tot_act)


    if not args.tmp_keep:
        shutil.rmtree(tmpdir)

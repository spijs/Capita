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

runcheck = __import__('mzn-runcheck')
cwd=os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd,'scripts'))
from checker import *
import instance2dzn as i2dzn
import forecast2dzn as f2dzn
import checker_mzn as chkmzn
from prices_data import *
from prices_regress import *
import numpy as np
from sklearn import linear_model

# from http://code.activestate.com/recipes/577932-flatten-arraytuple/
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
    args = parser.parse_args()
    
    # if you want to hardcode the MiniZincIDE path for the binaries, here is a resonable place to do that
    #args.mzn_dir = "/home/tias/local/src/MiniZincIDE-2.0.13-bundle-linux-x86_64"

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
    # load train/test data
    datafile = 'data/prices2013.dat';
    dat = load_prices(datafile)

    column_features = [ 'HolidayFlag', 'DayOfWeek', 'PeriodOfDay', 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA' ]; # within the same day you can use all except: ActualWindProduction, SystemLoadEP2, SMPEP2
          # I ommitted ORKTemperature and ORKWindspeed because it contains 'NaN' missing values (deal with it if you want to use those features), also CO2Intensity sometimes
    column_predict = 'SMPEP2'
    historic_days = 30

    day = None
    if args.day:
        day = datetime.strptime(args.day, '%Y-%m-%d').date()
    else:
        day = get_random_day(dat, args.historic_days)
    if args.v >= 1:
        print "First day:",day

    preds = [] # [(model_name, predictions)]

    # features, learning and predictions
    rows_prev = get_data_prevdays(dat, day, timedelta(args.historic_days))
    X_train = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_prev]
    y_train = [ eval(row[column_predict]) for row in rows_prev ]

    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)

    preds = [] # per day an array containing a prediction for each PeriodOfDay
    actuals = [] # also per day
    for (i,f) in enumerate(f_instances):
        today = day + timedelta(i)
        rows_tod = get_data_day(dat, today)
        X_test = [ [eval(v) for (k,v) in row.iteritems() if k in column_features] for row in rows_tod]
        y_test = [ eval(row[column_predict]) for row in rows_tod ]
        preds.append( clf.predict(X_test) )
        actuals.append( y_test )
    if args.v >= 1:
        #print preds, actuals
        print "Plotting actuals vs predictions..."
        plot_preds( [('me',qflatten(preds))], qflatten(actuals) )

    print "shape actuals: ", np.array(actuals).shape
    # the scheduling
    tot_act = 0
    tot_time = 0
    for (i,f) in enumerate(f_instances):
        data_forecasts = preds[i]
        data_actual = actuals[i]
        print "data actual ", data_actual
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

    print "%s from %s, linear: Total time: %.2f -- total actual cost: %.1f"%(args.file_instance, day, tot_time, tot_act)


    if not args.tmp_keep:
        shutil.rmtree(tmpdir)

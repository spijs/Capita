__author__ = 'spijs'

import sys
verbosity = 0


def write(string,level=0):
    global verbosity
    if verbosity and not level:
        sys.stdout.write(string)
    elif verbosity==2 and level==1:
        sys.stdout.write(string)

def flush():
    if verbosity:
        sys.stdout.flush()

def init_logger(v):
    global verbosity
    verbosity = v
import random
import csv
import sys
from datetime import *



def load_prices(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ', quotechar='"', skipinitialspace=True)
        for row in reader:
            row['datetime'] = datetime.strptime(row['#DateTime'], '%a %d/%m/%Y %H:%M')
            data.append(row)
    return data

def removeNan(dataset, column):
    for i in range(len(dataset)):
        if dataset[i][column] == 'NaN':
            interpolate(dataset, column, i)
        if dataset[i][column] == "":
            dataset[i][column] ="no value"
    return dataset

def interpolate(dataset, column, startposition):
    leftbound = startposition-1
    rightbound = startposition+1
    while dataset[rightbound][column] == 'NaN':
        rightbound += 1
    steps = rightbound - leftbound
    leftvalue = float(dataset[leftbound][column])
    rightvalue = float(dataset[rightbound][column])
    step = (rightvalue - leftvalue)/ (steps*1.0)
    for i in range(steps-1):
        dataset[startposition+i][column] = leftvalue + (i + 1)*step
    return dataset


if __name__ == '__main__':
    datafile = '../data/prices2013.dat'
    dat = load_prices(datafile)
    for column in dat[0].keys():
        dat = removeNan(dat, column)

    with open('../data/cleanData.csv', 'wb+') as csvfile:
        print "fieldnames ", dat[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=dat[0].keys(), delimiter = " ", quotechar = '"')
        writer.writeheader()
        for row in dat:
            writer.writerow(row)
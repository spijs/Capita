__author__ = "Wout & Thijs"

import csv
from datetime import *
import math



def load_prices(filename):
    '''
    Load a given csv file into memory
    :param filename: file to be loaded
    :return: list containing the data from the file
    '''
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ', quotechar='"', skipinitialspace=True)
        for row in reader:
            row['datetime'] = datetime.strptime(row['#DateTime'], '%a %d/%m/%Y %H:%M')
            data.append(row)
    return data

def removeNan(dataset, column):
    '''
    Remove the NaN values from a column in a dataset by interpolating. Empty fields are replaced with
    a "no value" string
    :param dataset: dataset to use
    :param column: column to remove Nan from
    :return: the dataset without NaN values in the given column
    '''
    for i in range(len(dataset)):
        if dataset[i][column] == 'NaN':
            interpolate(dataset, column, i)
        if dataset[i][column] == "":
            dataset[i][column] ="no value"
    return dataset

def interpolate(dataset, column, startposition):
    '''
    Interpolates a sequence of NaN values in a column of a dataset, starting at a given position
    :param dataset: the dataset to modify
    :param column: the column in which to do the interpolation
    :param startposition: index of the first NaN in the sequence to replace
    :return: the dataset with the sequence of NaN values replaced with an interpolation of the adjacent values
    '''
    leftbound = startposition-1
    rightbound = startposition+1
    while dataset[rightbound][column] == 'NaN':
        rightbound += 1
    steps = rightbound - leftbound
    leftvalue = float(dataset[leftbound][column])
    rightvalue = float(dataset[rightbound][column])
    step = (rightvalue - leftvalue)/ (steps*1.0)
    for i in range(steps-1):
        dataset[startposition+i][column] = math.ceil((leftvalue + (i + 1)*step)*100)/100
    return dataset


def preprocess_data():
    '''
    Load a dirty dataset into memory, cleans it by removing all NaN values and adding an extra feature
    and then write it to disk
    '''
    datafile = '../data/prices2013.dat'
    dat = load_prices(datafile)
    for column in dat[0].keys():
        dat = removeNan(dat, column)
    for row in dat:
        smpepval = float(row['SMPEP2'])
        if smpepval > 150.0:
            row['peak'] = 1
        else:
            row['peak'] = 0
    with open('../data/cleanData.csv', 'wb+') as csvfile:
        print "fieldnames ", dat[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=dat[0].keys(), delimiter=" ", quotechar='"')
        writer.writeheader()
        for row in dat:
            writer.writerow(row)


if __name__ == '__main__':
    preprocess_data()
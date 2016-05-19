import numpy.random as rand
import numpy as np
import csv
from datetime import *

def get_two_different_randoms():
    test = rand.randint(7)
    val = rand.randint(7)
    while val == test:
        val = rand.randint(7)
    return test, val

def load_prices(filename):
    data = []
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=' ', quotechar='"', skipinitialspace=True)
        for row in reader:
            row['datetime'] = datetime.strptime(row['#DateTime'], '%a %d/%m/%Y %H:%M')
            data.append(row)
    return data

def updateDataset(data, results, day, dataset):
    resultvector = []
    datavector = []
    first = dataset[day*48]
    for featurename in ['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'Month']:
        datavector.append(float(first[featurename]))
    for i in range(48):
        halfhour = dataset[day*48 + i]
        resultvector.append(halfhour['SMPEP2'])
        for featurename in ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'CO2Intensity', 'ORKTemperature', 'ORKWindspeed']:
            datavector.append(float(halfhour[featurename]))
    for days_back in range(1,8):
        for featurename in ['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'Month']:
            datavector.append(float(halfhour[featurename]))
        for h in range(48):
            halfhour = dataset[(day-days_back)*48 + h]
            for featurename in ['ForecastWindProduction', 'SystemLoadEA','PeriodOfDay' , 'SMPEA', 'CO2Intensity', 'SMPEP2', 'ORKTemperature', 'ActualWindProduction', 'ORKWindspeed', 'SystemLoadEP2']:
                datavector.append(float(halfhour[featurename]))
    data.append(datavector)
    results.append(resultvector)
    return data, results

def writeResults(data, results, split):
    np.set_printoptions(suppress=True)
    with open('../data/data_'+split+'.txt', 'w+') as datafile:
        for vector in data:
            datafile.write(str(vector)+'\n')
    with open('../data/results_' + split + '.txt', 'w+') as resfile:
        for vector in results:
            resfile.write(str(vector) + '\n')

def getData(split):
    data = []
    datafile = open('../data/data_' + split + '.txt')
    dataline = datafile.readline()
    while(dataline != ""):
        splitLine = dataline[1:-2].split(", ")
        for el in range(len(splitLine)):
            splitLine[el] = float(splitLine[el])
        data.append(np.array(splitLine))
        dataline = datafile.readline()
    datafile.close()
    resfile = open('../data/results_' + split + '.txt')
    results = []
    dataline = resfile.readline()
    while (dataline != ""):
        splitLine = dataline[1:-2].split(", ")
        for el in range(len(splitLine)):
            splitLine[el] = float(splitLine[el][1:-1])
        results.append(np.array(splitLine))
        dataline = resfile.readline()
    return np.array(data), np.array(results)


if __name__ == '__main__':
    datafile = '../data/cleanData.csv';
    dataset = load_prices(datafile)
    testdata = []
    testresults = []
    valdata = []
    valresults = []
    traindata = []
    trainresults = []
    print "dataset keys", dataset[0].keys()
    test, val = get_two_different_randoms()
    for i in range(7,792): # loop over all days in dataset
        if ((i-7) % 84) < 28:
            traindata, trainresults = updateDataset(traindata, trainresults, i, dataset)
        elif ((i-7) % 84) < 42:
            testdata, testresults = updateDataset(testdata, testresults, i, dataset)
        elif ((i-7) % 84) < 70:
            traindata, trainresults = updateDataset(traindata, trainresults, i, dataset)
        else:
            valdata, valresults = updateDataset(valdata, valresults, i, dataset)

    writeResults(testdata, testresults, 'test')
    writeResults(valdata, valresults, 'val')
    writeResults(traindata, trainresults, 'train')

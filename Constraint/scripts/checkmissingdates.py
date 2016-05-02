from scripts.prices_data import load_prices
from scripts.check_missing_times import check_missing

if __name__ == '__main__':
    datafile = '../data/prices2013.dat';
    dat = load_prices(datafile)

    for i in range(len(dat)-1):
        date1 = [dat[i]['Day'], dat[i]['Month'], dat[i]['Year'], dat[i]['PeriodOfDay']]
        date2= [dat[i+1]['Day'], dat[i+1]['Month'], dat[i+1]['Year'], dat[i+1]['PeriodOfDay']]
        if not check_missing([date1,date2]):
            print ("fout: %s" % str(date1))
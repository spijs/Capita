__author__ = 'spijs'

def get_months(year):
    '''
    :param year: chosen year
    :return: map from month number to number of days in the month
    '''
    months={
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }
    if year==2012:
        months[2]=29
    return months

def check_missing(times):
    ''' Expects a list of lists [[day,month,year,halfhour],...], returns True if there are no missing time slots.'''
    start = times[0]
    for i in range(1,len(times)):
        next = times[i]
        if not is_next(start,next):
            return False
        start=next
    return True


def is_next(first,next):
    '''
    :param first: First time
    :param next: Second time
    :return: True if the second time is one time slot after the first.
    '''
    day = int(first[0])
    month = int(first[1])
    year = int(first[2])
    halfhour = int(first[3])
    next_day = int(next[0])
    next_month = int(next[1])
    next_year = int(next[2])
    next_halfhour = int(next[3])
    check, end_of_day = is_next_halfhour(halfhour,next_halfhour)
    if not check:
        print ('Halfhour missing')
        return False
    check, end_of_month = is_next_day(day,next_day,month,year,end_of_day)
    if not check:
        print ('Day missing')
        return False
    return is_next_month(month,next_month,end_of_month,year,next_year)

def is_next_halfhour(first,next):
    '''
    :param first: half hour number
    :param next: half hour number
    :return: True if the second half hour is right after the first, True if next is the last half hour in a day.
    '''
    if (first==47 and next==0):
        return True,True
    return next == first +1, False

def is_next_day(day,next,month,year,end):
    '''
    Checks whether the given days in the given month and year are after eachother.
    :param day: first day
    :param next: second day
    :param month:
    :param year:
    :param end: True if the first day has just begun
    :return: True if the second day is one day after the first, True if it is at the end of the month.
    '''
    if not end: # Same day?
        return day==next,False
    days_in_month = get_months(year)[month]
    if days_in_month==day: #Last day of the month
        return next==1, True
    else:
        return next == day + 1, False

def is_next_month(first,next,end,year,next_year):
    '''
        Checks whether the first and next are one month apart.
    '''
    if not end:
        return first==next
    if first==12:
        return next_year == year +1 and next==1
    else:
        return next == first + 1


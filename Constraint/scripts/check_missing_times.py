__author__ = 'spijs'

def get_months(year):
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

''' Expects a list of lists [[day,month,year,halfhour],...]'''
def check_missing(times):
    start = times[0]
    for i in range(1,len(times)):
        next = times[i]
        if not is_next(start,next):
            return False
        start=next
    return True

def is_next(first,next):
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
    if (first==47 and next==0):
        return True,True
    return next == first +1, False

def is_next_day(day,next,month,year,end):
    if not end: # Same day?
        return day==next,False
    days_in_month = get_months(year)[month]
    if days_in_month==day: #Last day of the month
        return next==1, True
    else:
        return next == day + 1, False

def is_next_month(first,next,end,year,next_year):
    if not end:
        return first==next
    if first==12:
        return next_year == year +1 and next==1
    else:
        return next == first + 1


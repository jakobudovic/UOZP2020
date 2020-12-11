import datetime as da

FORMAT = "%Y-%m-%d %H:%M:%S.%f"

def parsedate(x): # convert date to a FORMAT
    if not isinstance(x, da.datetime):
        x = da.datetime.strptime(x, FORMAT) # remove last 3 zeros from date
    return x

def diff_dates(x, y):  # calculate difference in dates in seconds, absolute value
    diff = abs((parsedate(x) - parsedate(y)).total_seconds())
    # if diff > 3600:  # buses over midnight
    #     diff = diff - 3600
    return diff

def add_seconds(x, seconds):  # add seconds to the date
    d = da.timedelta(seconds=seconds)
    nd = parsedate(x) + d
    return nd.strftime(FORMAT)

if __name__ == "__main__":

    d1 = "2012-01-01 23:30:30.000"
    d2 = "2012-01-02 00:31:30.000"
    testd1 = da.datetime.strptime(d1, FORMAT) # example how to put our date to format

    a = diff_dates(d1, d2)
    b = add_seconds(testd1, 1)
    print("a: {}, b {}".format(a,b))
    """    """


import time

def timeCheck(type, stime) :
    if type is 's' : # start type
        stime[0] = time.time()

    elif type is 'e' : # end type
        elapsed = time.time() - stime[0]
        print("%.2f seconds elapsed" % (elapsed))
import datetime
import lifelonglearning34 as l34
import os
import collections, itertools, codecs
import numpy as np
def open(name,type):
    if l34.getConstant('VN')==True:
        return codecs.open(name,mode=type,encoding = 'utf-8')
    else:
        return codecs.open(name,mode=type)
def getScriptName():
    logging.debug('hell no')
    return os.path.basename(__file__).split('.')[0]
def getTimeString():
    return str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
def makeExist(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
def sumListCounters(list_counters):
    return sum(list_counters,collections.Counter())
def TupleFixedEmptyPosition(values_list, position_list):
    '''format new tuple to fit old-style tuple'''
    for x in position_list:
        try:
            y = int(x)
        except:
            return None
    if len(values_list)!=len(position_list):
        return None
    positions = sorted(position_list)
    values = [''] * (positions[-1]+1)
    for i in position_list:
        values[i] = values_list[position_list.index(i)]
    return tuple(values)
def sortCounter(count):
    return collections.Counter(collections.OrderedDict(count.most_common()))
def getFileList(mypath):
    '''for example: crawling/'''
    if os.path.isfile(mypath):
        return [mypath.split('/')[-1]]
    return [f for f in os.listdir(mypath) if os.path.isfile(mypath+f)]
def getFolderList(parent_folder):
    return [f for f in os.listdir(parent_folder) if os.path.isdir(parent_folder +f)]
def combineListofList(list_of_list):
    return list(itertools.chain(*(list_of_list)))
def getTrainTestTuple(segments, pivot):
    '''(train,test), combine list of list done'''
    test = segments[pivot]
    train = combineListofList( segments[pivot+1:] + segments[0:pivot])
    return (train,test)
def getTrimString(txt):
    '''remove space, get 3 first letters from each word'''
    m = txt.split()
    r = ''
    for mm in m:
        if mm.isalnum():
            r = r+mm[:3]
        else:
            for i in range(0,3):
                try:
                    if mm[i].isalnum():
                        r = r + mm[i]
                except:
                    continue
    return r
def getMinutesDifference(datetime_new,datetime_old):
    ''' type datetime.datetime '''
    c = datetime_new - datetime_old
    return divmod(np.absolute( c.days * 86400 + c.seconds), 60)[0]
def delNanCounter(count):
    k_list = count.keys()
    for k in k_list:
        if np.math.isnan(count[k]) == True:
            del count[k]
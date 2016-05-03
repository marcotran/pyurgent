import os
import sys
import logging
import collections
import lifelonglearning34 as l3
#def getVietnameseToneMarks():
#    return []
def getEnglishStopwords():
    try:
        from nltk.corpus import stopwords
        import unidecode
        cachedstopwords = stopwords.words('english')
        str_cachedstopwords = []
        for x in cachedstopwords:
            str_cachedstopwords.append(unicodedata.normalize('NFKD',x).encode('ascii','ignore'))
        return str_cachedstopwords 
    except:
        print(sys.exc_info())
        return []
def getScriptName():
    logging.debug('hell no')
    return os.path.basename(__file__).split('.')[0]
def delElementsFromBlackList(item_list,black_list):
    n = item_list
    for x in black_list:
        while 1==1:
            try:
                n.remove(x)
            except:
                break
    return n
def selectUnigramCounter(count):
    if l3.getConstant('bigram naive bayes') != False:
        return collections.Counter(dict( (k,v) for k,v in count.items() if l3.isTxtBigram( k)))
    else:
        return count
def getVietnameseTone():
    return ['́', # sắc
            '̣', # nặng
            '̉',# hỏi
            '̀' ,# huyền
            '̃'  # ngã
            ]
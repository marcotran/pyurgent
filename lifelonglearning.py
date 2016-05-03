from __future__ import division
def mlog(function_name,content):
    import datetime
def sumaCounter(counter):
    return sum(counter.itervalues())
def cleanCounterBagofWords(bag_counter,removing_word_list):
    bag = bag_counter
    for y in bag_counter.keys():
        if y in removing_word_list:
            del bag_counter[y]
    return bag_counter
def getStrStopwords():
    from nltk.corpus import stopwords
    import unicodedata
    cachedstopwords = stopwords.words('english')
    str_cachedstopwords = []
    for x in cachedstopwords:
        str_cachedstopwords.append(unicodedata.normalize('NFKD',x).encode('ascii','ignore'))
    return str_cachedstopwords
def getBagofWordsfromReviews(reviews):
    import collections
    import re
    bag_of_words = [collections.Counter(re.findall(r'\w+',txt.lower())) for txt in reviews]
    sum_bag = sum(bag_of_words,collections.Counter())
    return sum_bag
def getDomainName(filename):
    spl = filename.split('.')
    return spl[0]
def readDomainToList(filename):
    with open(filename) as myfile:
        domain = myfile.readlines()
    del domain[0]
    return domain
def divideDomain(domain):
    mlog(divideDomain.__name__,"call")
    pos_rev = []
    neg_rev = []
    bias_rev = []
    total_rev = []
    for x in domain:
        spl = x.split('\t')
        if len(spl) == 4:
            if spl[1] == 'POS':
                pos_rev.append(spl[3].strip('\n'))
                bias_rev.append((True,spl[3].strip('\n')))
                total_rev.append(('POS',spl[3].strip('\n')))
            elif spl[1] == 'NEG':
                neg_rev.append(spl[3].strip('\n'))
                bias_rev.append((False,spl[3].strip('\n')))
                total_rev.append(('NEG',spl[3].strip('\n')))
            total_rev.append(('NEU',spl[3].strip('\n')))
    return (pos_rev,neg_rev,bias_rev,total_rev)
def initializeX(data):
    import collections
    X_plus = collections.Counter()
    X_minus = collections.Counter()
    for dm in data:
        if type(dm[2])==collections.Counter and type(dm[3])==collections.Counter:
            X_plus = X_plus + dm[2]
            X_minus = X_minus + dm[3]
        else:
            print 'type error at position = '+str(data.index(dm))
    return (X_plus, X_minus)
def getDomainfromFile(filename):
    domainName = getDomainName(filename)
    domain = readDomainToList(filename)
    return domain
def formatDomainBalancing(domain):
    balanced_rev = []
    pos = 0
    neg = 0
    threshold = 100
    for x in domain:
        spl = x.split('\t')
        if len(spl)==4:
            if spl[1] =='POS':
                if pos<100:
                    pos = pos+1
                    balanced_rev.append(x)
            elif spl[1] =='NEG':
                if neg<100:
                    neg = neg +1
                    balanced_rev.append(x)
    if pos !=100 or neg!=100:
        return False
    else:
        return balanced_rev
def formatDomain(testDomain):
    clean_rev = []
    for x in testDomain:
        spl = x.split('\t')
        if len(spl) == 4:
            if spl[1] != 'NEU':
                clean_rev.append((spl[1],spl[3].strip('\n')))
    return clean_rev
def getDomainData(domain,removing_words):
    import collections
    import re
    pos_rev = []
    neg_rev = []
    bias_rev = []
    total_rev = []
    (pos_rev,neg_rev,bias_rev,total_rev) = divideDomain(domain)
    ratio = len(pos_rev) / len(neg_rev)
    pos_bag = []
    neg_bag = []
    pos_bag = getBagofWordsfromReviews(pos_rev)
    neg_bag = getBagofWordsfromReviews(neg_rev)
    bias_count = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1]))) for txt in bias_rev]
    clean_count = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in bias_rev]
    clean_biascount = []
    for c in clean_count:
        clean_biascount.append((c[0],cleanCounterBagofWords(c[1],removing_words)))
    pos_count = cleanCounterBagofWords(pos_bag,removing_words)
    neg_count = cleanCounterBagofWords(neg_bag,removing_words)
    vocab_list = (pos_count + neg_count).keys()
    vocab_len = len(vocab_list)
    domainName = 'pussy_cat'
    mlog(getDomainData.__name__,"end")
    if (type(pos_count)!=collections.Counter or type(neg_count)!=collections.Counter):
        print getDomainData.__name__ + ' wrong type'
    return (domainName,bias_count,pos_count,neg_count, ratio,vocab_len,clean_biascount,vocab_list)
def getObjectiveFunctionDiff(word,document_counter,vocab,smooth,ratio,X_plus,X_minus, sum_Xplus, sum_Xminus,vocab_T,Nt_plus,Nt_minus):
    from sympy import mpmath
    import numpy as np
    import warnings
    import sys
    mpmath.mp.pretty = True
    warnings.filterwarnings("error")
    #INDEPENDENT word
    fraction_nplus = document_counter[word] / (smooth + X_plus[word])
    fraction_nminus = document_counter[word] / (smooth + X_minus[word])
    di = sum(document_counter.itervalues())
    g = ((vocab * smooth + sum_Xplus) / (vocab * smooth + sum_Xminus)) ** di
    product_pi = 1
    for key in document_counter.keys():
        #WARNING: divided by zero
        try:
            product_pi = product_pi * (((smooth + X_minus[key]) / (smooth + X_plus[key])) ** document_counter[key])
        except Warning:
            print "smooth "+str(smooth)+" key "+str(key)+ " "+str(X_plus[key])+" "+str(X_minus[key])
    g_diffpos = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus - X_plus[word] + x) / (smooth * vocab + sum_Xminus)) ** di,X_plus[word])
    g_diffneg = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus) / (smooth * vocab + sum_Xminus - X_minus[word] + x)) ** di,X_minus[word])
    sum_penalty_T = 0
    #INDEPENDENT word
    for w in vocab_T:
        sum_penalty_T = sum_penalty_T + (X_plus[w]-Nt_plus[w])**2 + (X_minus[w]-Nt_minus[w])**2
    penalty_T_diffpos = mpmath.diff(lambda x:sum_penalty_T-(X_plus[word]-Nt_plus[word])**2+(x-Nt_plus[word])**2,X_plus[word])
    penalty_T_diffneg = mpmath.diff(lambda x:sum_penalty_T-(X_minus[word]-Nt_minus[word])**2+(x-Nt_minus[word])**2,X_minus[word])
    #FUNCTION regularization penalty terms here
    #FUNCTION check vocabulary here    
    Fplus = (fraction_nplus + (ratio ** -1) * product_pi * g_diffpos) / (1 + (ratio ** -1) * product_pi * g) - fraction_nplus + penalty_T_diffpos
    import sys
    try:
        Fminus = (fraction_nminus * g + g_diffneg) / (ratio * (product_pi ** -1) + g) + penalty_T_diffneg
    except ZeroDivisionError:
        mlog(getObjectiveFunctionDiff.__name__,ZeroDivisionError.message + ' word=' + word + ' product_pi = ' + str(product_pi))
        product_pi = 1
        Fminus = (fraction_nminus * g + g_diffneg) / (ratio * (product_pi ** -1) + g)
    return (np.float64(Fplus),np.float64(Fminus))
def updateXstochastic(vocab,X_plus,X_minus,clean_document, ratio,vocab_T,Nt_plus,Nt_minus):
    import collections
    import numpy as np
    new_Xplus = X_plus
    new_Xminus = X_minus
    sum_Xplus = sumaCounter(X_plus)
    sum_Xminus = sumaCounter(X_minus)
    for key in clean_document[1].keys():
        #UPDATE
        dF = getObjectiveFunctionDiff(key,clean_document[1],vocab,1,ratio,X_plus,X_minus,sum_Xplus,sum_Xminus,vocab_T,Nt_plus,Nt_minus)
        new_Xplus[key] = X_plus[key] - dF[0]
        new_Xminus[key] = X_minus[key] - dF[1]
    mlog(updateXstochastic.__name__,"Xplus " + str(new_Xplus.most_common(10)))
    mlog(updateXstochastic.__name__,"Xminus " + str(new_Xminus.most_common(10)))
    return (new_Xplus,new_Xminus)
def calcProbabilityVocab(smooth, vocab_list, X_plus, X_minus, sum_Xplus, sum_Xminus):
    from itertools import islice    
    import numpy as np
    Pword_plus = {}
    Pword_minus = {}
    eword_plus, eword_minus = {}, {}
    V = len(vocab_list)  
    for word in vocab_list:
        Pword_plus[word] = (smooth + X_plus[word]) / (smooth * V + sum_Xplus)
        eword_plus[word] = np.log(Pword_plus[word])
        Pword_minus[word] = (smooth + X_minus[word]) / (smooth * V + sum_Xminus)
        eword_minus[word] = np.log(Pword_minus[word])
    return (eword_plus,eword_minus)
def calcProbabilityDocument(Pword_plus,Pword_minus,document_counter,ratio,removing_words):
    Pdoc_plus = ratio
    Pdoc_minus = ratio ** -1
    import numpy as np
    ep, em = np.log(Pdoc_plus), np.log(Pdoc_minus)
    for word in document_counter.keys():
        if word in removing_words:
            continue
        if word in Pword_plus.keys():
            ewp = Pword_plus[word]
            ep = ep + ewp * document_counter[word]
        else:
            mlog(calcProbabilityDocument.__name__,'word ' + word + ' not in Pword_plus')
        if word in Pword_minus.keys():
            ewm = Pword_minus[word]
            em = em + ewm * document_counter[word]
        else:
            mlog(calcProbabilityDocument.__name__,'word ' + word + ' not in Pword_minus')
    return (ep,em)
def evalObjectiveFunction(clean_biascount,Pword_plus,Pword_minus,ratio,removing_words):
    mlog(evalObjectiveFunction.__name__,"call")
    import numpy as np
    import math
    #Obj = np.log(1)
    Obj = 0
    for doc in clean_biascount:
        Pdoc = calcProbabilityDocument(Pword_plus,Pword_minus,doc[1],ratio,removing_words)
        i = clean_biascount.index(doc) 
        if i == 0:
            mlog(evalObjectiveFunction.__name__,'Pdoc 0 ' + str(Pdoc) + ' type ' + str(type(Pdoc)))
            t00 = np.exp(np.float64(Pdoc[0]))
            t01 = np.exp(np.float64(Pdoc[1]))
            t1 = t00 - t01
            #print str(t1)
            t2 = abs(t1)
            Obj = np.log(t2)
            #print str(Obj)
        if i % 100 == 0:
            mlog(evalObjectiveFunction.__name__,"Pdoc + " + str(i) + "= " + str(Pdoc))
        if doc[0] == True:
            Obj = np.logaddexp(Obj,Pdoc[0])
            if i % 100 == 1:
                mlog(evalObjectiveFunction.__name__,"Obj+ " + str(i) + " after += " + str(Obj))
            Obj = np.log(np.exp(Obj) - np.exp(Pdoc[1]))
        elif doc[0] == False:
            Obj = np.log(np.exp(np.float64(Obj)) + np.exp(np.float64(Pdoc[1])))
            if i % 100 == 2:
                mlog(evalObjectiveFunction.__name__,"Obj- " + str(i) + " after += " + str(Obj))
            Obj = np.log(np.exp(Obj) - np.exp(Pdoc[0]))
        if Obj == 0.0:
            mlog(evalObjectiveFunction.__name__, "Obj=0 fuck " + str(i) + "")
    mlog(evalObjectiveFunction.__name__,"Obj = " + str(np.exp(Obj)))
    if math.isnan(np.exp(Obj))==False:
        print 'J = ' + str(np.exp(Obj))
    return Obj #type np.log
def get_removing_words():
    meaningless = ['lrb','rrb']
    return meaningless + getStrStopwords()
def testing(Pword_plus,Pword_minus,testdata,ratio,removing_w):
    #removing NEUTRAL reviews
    import numpy as np
    import collections
    bias_testdata = []
    pos_no = 0
    neg_no = 0
    for x in testdata:
        if x[0] != 'NEU':
            bias_testdata.append(x)
            if x[0] == 'POS' or x[0] == True:
                pos_no = pos_no + 1
            elif x[0] == 'NEG' or x[0] == False:
                neg_no = neg_no + 1
    if pos_no + neg_no != len(bias_testdata):
        print "false counting bias_testdata"
    compare_chart = []
    for x in bias_testdata:
        Pd = calcProbabilityDocument(Pword_plus,Pword_minus,x[1],ratio,removing_w)
        guess = ''
        if np.exp(Pd[0]) > np.exp(Pd[1]):
            guess = 'POS'
        else:
            guess = 'NEG'
        if guess == x[0]:
            compare_chart.append(True)
        else:
            compare_chart.append(False)
    mlog(testing.__name__,'compare_chart = ' + str(compare_chart))
    count_compare = collections.Counter(compare_chart)
    mlog(testing.__name__,'count True/False = ' + str(count_compare))
    accuracy = count_compare[True] / sumaCounter(count_compare)
    print 'accuracy = ' + str(accuracy)
    mlog(testing.__name__,'accuracy = ' + str(accuracy))
    return (accuracy,compare_chart)
def getDomainFileNames():
    return ['AlarmClock.txt',
            'Baby.txt',
            'Bag.txt',
            'CableModem.txt',
            'Dumbbell.txt',
            'Flashlight.txt',
            'Gloves.txt',
            'GPS.txt',
            'GraphicsCard.txt',
            'Headphone.txt',
            'HomeTheaterSystem.txt',
            'Jewelry.txt',
            'Keyboard.txt',
            'Magazine_Subscriptions.txt',
            'Movies_TV.txt',
            'Projector.txt',
            'RiceCooker.txt',
            'Sandal.txt',
            'Vacuum.txt',
            'Video_Games.txt']
def divideDomainCrossValidation(domain,segment):
    #segment varies 0-4 type: int
    #domain/data or any kind of array
    if segment<0:
        segment = 0
    elif segment>4:
        segment = 4
    n_fold = int(len(domain)/5)
    left_pivot = int(segment) * n_fold
    right_pivot = left_pivot + n_fold
    if right_pivot > len(domain) - 1:
        right_pivot = len(domain) - 1
    testDomain = domain[left_pivot:right_pivot]
    trainDomain = domain[0:left_pivot] + domain[right_pivot:-1]
    return (trainDomain,testDomain)
def getTestReviewCounter(domain, removing_words,segment):
    import collections
    import re
    (trainDomain,testDomain) = divideDomainCrossValidation(domain,segment)
    test_rev = formatDomain(testDomain)
    testrev_counter = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in test_rev]
    clean_testrev_counter = []
    for ct in testrev_counter:
        clean_testrev_counter.append((ct[0],cleanCounterBagofWords(ct[1],removing_words)))
    return clean_testrev_counter
def getVocabTRegularization_NaiveProb(pos_count,neg_count,vocab_list, delta):
    import sys
    total_plus = sumaCounter(pos_count)
    total_minus = sumaCounter(neg_count)
    V = len(vocab_list)
    vocab_T = []
    for word in vocab_list:
        try:
            Pp = (pos_count[word]+1)/(total_plus+V)
            Pn = (neg_count[word]+1)/(total_minus+V)
        except:
            print sys.exc_info()
            Pp = 1/(total_plus+V)
            Pn = 1/(total_minus+V)
        if (Pp/Pn > delta):
            vocab_T.append(word)
        elif (Pn/Pp > delta):
            vocab_T.append(word)
    print "added to vocab_T "+str(len(vocab_T))
    return vocab_T
def demoBalanced19():
    import numpy as np
    import datetime
    print datetime.datetime.now().time()
    domainFiles = getDomainFileNames()
    cached_data = []
    cached_domain_s = []
    for df in domainFiles:
        cached_domain = getDomainfromFile(df)
        cached_balanced_domain = formatDomainBalancing(cached_domain)
        if cached_balanced_domain != False:
            cached_domain_s.append(cached_balanced_domain)
            cached_data.append(getDomainData(cached_balanced_domain,get_removing_words()))
            #FUNCTION add M matrix here
        else:
            cached_domain_s.append(None)
            cached_data.append(None)
    n_bd = len(cached_domain_s)
    past_data = []
    total_accuracy = []
    total_result = []
    print datetime.datetime.now().time()
    for i in range(0,n_bd):
        if cached_domain_s[i]!=None:
            #prepare past data
            for j in range(0,n_bd):
                if j!=i:
                    if cached_data[j]!=None:
                        past_data.append(cached_data[j])
            accuracy = []
            for s in range(0,5):
                clean_testrev_counter = getTestReviewCounter(cached_domain_s[i],get_removing_words(),s)
                (train_domain, test_domain)=divideDomainCrossValidation(cached_domain_s[i],s)
                #get T vocabulary set
                target_domain_data = getDomainData(train_domain,get_removing_words())
                past_data.append(target_domain_data)
                vocab_T = getVocabTRegularization_NaiveProb(target_domain_data[2],target_domain_data[3],target_domain_data[7],6)
                X = []
                X.append(initializeX(past_data))
                (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J) = evalLLLValues(1,past_data,X[0])
                J_s = []
                J_s.append(J)
                # update X stochastic
                for n in range(0,len(past_data[-1][6])):
                    #ERROR
                    X.append(updateXstochastic(past_data[-1][5],X[-1][0],X[-1][1],past_data[-1][6][n],past_data[-1][4],vocab_T,past_data[-1][2],past_data[-1][3]))
                    (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J) = evalLLLValues(1,past_data,X[-1])
                    J_s.append(J)
                    if float(np.exp(J_s[-1]) - np.exp(J_s[-2])) < 0.001: 
                        break
                ratio = past_data[-1][4]
                #test_data
                result = testing(Pword_plus,Pword_minus,clean_testrev_counter,ratio,get_removing_words())
                accuracy.append(result[0])
            avg_accuracy = sum(accuracy)/len(accuracy)
            total_accuracy= total_accuracy+accuracy
            total_result.append((domainFiles[i],avg_accuracy))
            print domainFiles[i]+' '+str(avg_accuracy)
            print datetime.datetime.now().time()
    if len(total_accuracy)>0:
        print 'total average accuracy '+str(sum(total_accuracy)/len(total_accuracy))
    else:
        print 'total accuracy len 0'
    print total_result
def evalLLLValues(smooth,data,X):
    sum_Xplus = sumaCounter(X[0])
    sum_Xminus = sumaCounter(X[1])
    (Pword_plus,Pword_minus) = calcProbabilityVocab(smooth, data[-1][7], X[0], X[1], sum_Xplus, sum_Xminus)
    J=evalObjectiveFunction(data[-1][6],Pword_plus,Pword_minus,data[-1][4],get_removing_words())
    return (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J)             
if __name__ == '__main__':
    import winsound
    demoBalanced19()
    print "with V t regularization"
    winsound.PlaySound("*", winsound.SND_ALIAS)
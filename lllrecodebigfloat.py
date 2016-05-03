from __future__ import division
import logging
#from numba import autojit
def autolog(message):
    "Automatically log the current function details."
    import inspect
    # Get the previous frame in the stack, otherwise it would
    # be this function!!!
    func = inspect.currentframe().f_back.f_code
    # Dump the message + the name of this function to the log.
    logging.debug("%s: %s in %s:%i" % (
        message, 
        func.co_name, 
        func.co_filename, 
        func.co_firstlineno
    ))
def doit(message):
    n = 0
def mlog(function_name,content):
    import datetime
    #with open('log.txt','a') as myfile:
        #print >>myfile, function_name + '\t' + content + '\t' + str(datetime.datetime.now().time())
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
    #mlog(divideDomain.__name__,"call")
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
def formatDomainNatural(domain):
    natural_rev = []
    pos = 0
    neg = 0
    threshold = 100
    for x in domain:
        spl = x.split('\t')
        if len(spl)==4:
            if spl[1] =='POS':
                pos = pos+1
                natural_rev.append(x)
            elif spl[1] =='NEG':
                neg = neg +1
                natural_rev.append(x)
    logging.debug(' pos '+str(pos)+' neg '+str(neg))
    return natural_rev
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
    vocab = len(vocab_list)
    domainName = 'pussy_cat'    
    if (type(pos_count)!=collections.Counter or type(neg_count)!=collections.Counter):
        print getDomainData.__name__ + ' wrong type'
    return (domainName,bias_count,pos_count,neg_count, ratio,vocab,clean_biascount,vocab_list)
 
def getObjectiveFunctionDiff(word,document_counter,vocab,smooth,ratio,X_plus,X_minus, sum_Xplus, sum_Xminus,vocab_T,pos_count,neg_count,di,vocab_S,sum_penalty_Vs,R_Mkb,X_zeroplus,X_zerominus):
    from sympy import mpmath
    import numpy as np
    mpmath.mp.pretty = True
    #np.seterr(all='print')
    fraction_nplus = document_counter[word] / (smooth + X_plus[word])
    fraction_nminus = document_counter[word] / (smooth + X_minus[word])
    #INDEPENDENT
    g = ((vocab * smooth + sum_Xplus) / (vocab * smooth + sum_Xminus)) ** di
    product_pi = 1
    for key in document_counter.keys():
        product_pi = product_pi * (((smooth + X_minus[key]) / (smooth + X_plus[key])) ** document_counter[key])
    g_diffpos = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus - X_plus[word] + x) / (smooth * vocab + sum_Xminus)) ** di,X_plus[word])
    g_diffneg = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus) / (smooth * vocab + sum_Xminus - X_minus[word] + x)) ** di,X_minus[word])
    sum_penalty_Vt = 0
    #regularization term 1
    #INDEPENDENT
    for w in vocab_T:
        sum_penalty_Vt = sum_penalty_Vt + (X_plus[w]-pos_count[w])**2 +(X_minus[w]-neg_count[w])**2
    #FIX vocab limitation
    if word in vocab_T:
        penalty_Vt_plus = mpmath.diff(lambda x: 0.5*0.1*(sum_penalty_Vt -( X_plus[word]-pos_count[word])**2+(x-pos_count[word])**2),X_plus[word])
        penalty_Vt_minus = mpmath.diff(lambda x: 0.5*0.1*(sum_penalty_Vt -( X_minus[word]-neg_count[word])**2+(x-neg_count[word])**2),X_minus[word])
    else:
        penalty_Vt_plus = 0
        penalty_Vt_minus = 0
    #regularization term 2
    if word in vocab_S:
        penalty_Vs_plus = mpmath.diff(lambda x:0.5*0.1*(sum_penalty_Vs-(X_plus[word]-R_Mkb[word]*X_zeroplus[word])**2),X_plus[word])
        penalty_Vs_minus = mpmath.diff(lambda x:0.5*0.1*(sum_penalty_Vs-(X_minus[word]-(1-R_Mkb[word])*X_zerominus[word])**2),X_minus[word])
    else:
        penalty_Vs_plus = 0
        penalty_Vs_minus  = 0
    #dF+, dF-
    Fplus = (fraction_nplus + (ratio ** -1) * product_pi * g_diffpos) / (1 + (ratio ** -1) * product_pi * g) - fraction_nplus + penalty_Vt_plus + penalty_Vs_plus
    Fminus = (fraction_nminus * g + g_diffneg) / (ratio * (product_pi ** -1) + g) + penalty_Vt_minus + penalty_Vs_minus
    return (np.float64(Fplus),np.float64(Fminus)) 
def get_sum_penalty_Vs(X_plus,X_minus,R_Mkb,X_zeroplus,X_zerominus,vocab_S):
    sum_penalty_Vs = 0
    for w in vocab_S:
        sum_penalty_Vs = sum_penalty_Vs + (X_plus[w]-R_Mkb[w]*X_zeroplus[w])**2+(X_minus[w]-(1-R_Mkb[w])*X_zerominus[w])**2
    return sum_penalty_Vs
def get_R_Mkb(Mkb_p,Mkb_n,vocab_S):
    import collections
    R = collections.Counter()
    for w in vocab_S:
        if Mkb_p[w]==0:
            R[w]=0
        else:
            R[w]=Mkb_p[w]/(Mkb_p[w]+Mkb_n[w])
    return R
def updateXstochastic(vocab,X_plus,X_minus,clean_document, ratio,vocab_T,pos_count,neg_count,vocab_S,Mkb_p,Mkb_n,X_zeroplus,X_zerominus):
    import collections
    import numpy as np
    new_Xplus = X_plus
    new_Xminus = X_minus
    sum_Xplus = sumaCounter(X_plus)
    sum_Xminus = sumaCounter(X_minus)
    di = sum(clean_document[1].itervalues())
    R_Mkb = get_R_Mkb(Mkb_p,Mkb_n,vocab_S)
    sum_penalty_Vs = get_sum_penalty_Vs(X_plus,X_minus,R_Mkb,X_zeroplus,X_zerominus,vocab_S)
    for key in clean_document[1].keys():
        dF = getObjectiveFunctionDiff(key,clean_document[1],vocab,1,ratio,X_plus,X_minus,sum_Xplus,sum_Xminus,vocab_T,pos_count,neg_count,di,vocab_S,sum_penalty_Vs,R_Mkb,X_zeroplus,X_zerominus)
        new_Xplus[key] = X_plus[key] - dF[0]
        new_Xminus[key] = X_minus[key] - dF[1]
    logging.debug("Xplus " + str(new_Xplus.most_common(10))+" Xminus " + str(new_Xminus.most_common(10)))
    
    return (new_Xplus,new_Xminus) 
def calcProbabilityVocab(smooth, vocab_list, X_plus, X_minus, sum_Xplus, sum_Xminus):
    from itertools import islice
    #mlog(calcProbabilityVocab.__name__,"call")
    import numpy as np
    Pword_plus = {}
    Pword_minus = {}
    eword_plus, eword_minus = {}, {}
    V = len(vocab_list)  #check vocab_list
    for word in vocab_list:
        Pword_plus[word] = (smooth + X_plus[word]) / (smooth * V + sum_Xplus)
        eword_plus[word] = np.log(Pword_plus[word])
        Pword_minus[word] = (smooth + X_minus[word]) / (smooth * V + sum_Xminus)
        eword_minus[word] = np.log(Pword_minus[word])
    logging.debug('Pw+ ' + str(list(islice(Pword_plus.iteritems(),10)))+' Pw- ' + str(list(islice(Pword_minus.iteritems(),10))))
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
            doit('word ' + word + ' not in Pword_plus')
        if word in Pword_minus.keys():
            ewm = Pword_minus[word]
            em = em + ewm * document_counter[word]
        else:
            doit('word ' + word + ' not in Pword_minus')
    return (ep,em) 
def evalObjectiveFunction(clean_biascount,Pword_plus,Pword_minus,ratio,removing_words):
    #mlog(evalObjectiveFunction.__name__,"call")
    import numpy as np
    import math
    import sys
    np.seterr(all='ignore')
    #Obj = np.log(1)
    Obj = 1
    for doc in clean_biascount:
        Pdoc = calcProbabilityDocument(Pword_plus,Pword_minus,doc[1],ratio,removing_words)
        i = clean_biascount.index(doc) 
        if i == 0:
            Obj = np.log(np.absolute(np.subtract(np.exp(Pdoc[0]),np.exp(Pdoc[1]))))
        if doc[0] == True:
            try:
                Obj = np.logaddexp(Obj,np.float64(Pdoc[0]))
            except:
                print sys.exc_info()
                print "doc "+str(i)+" error "
                print "Obj = "+str(Obj)+" Pdoc[0] = "+str(Pdoc[0])+ " Pdoc[1] = "+str(Pdoc[1])
                return np.nan
            if (np.math.isnan(np.exp(Obj))):
                break
            Obj = np.log(np.subtract(np.exp(Obj) , np.exp(np.float64(Pdoc[1]))))
        elif doc[0] == False:
            #Obj = np.log(np.exp(np.float64(Obj)) + np.exp(np.float64(Pdoc[1])))#np.logaddexp(Obj,Pdoc[1])
            try:
                Obj = np.logaddexp(Obj,np.float64(Pdoc[1]))
            except:
                print sys.exc_info()
                print "doc "+str(i)+" error "
                print "Obj = "+str(Obj)+" Pdoc[0] = "+str(Pdoc[0])+ " Pdoc[1] = "+str(Pdoc[1])
                return np.nan
            if (np.math.isnan(np.exp(Obj))):
                break
            Obj = np.log(np.subtract(np.exp(Obj), np.exp(np.float64(Pdoc[0]))))
        if Obj == 0.0:
            logging.debug( "Obj=0 fuck " + str(i) + "")
    #IMPROVE J nan
    if math.isnan(np.exp(Obj))==False:
        print 'J = ' + str(np.exp(Obj))
        logging.debug("Obj = " + str(np.exp(Obj)))
    else:
        logging.debug("J nan")
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
        #CATCH nan problem
        if np.float64(Pd[0]) > np.float64(Pd[1]):
            guess = 'POS'
        else:
            guess = 'NEG'
        if guess == x[0]:
            compare_chart.append(True)
        else:
            compare_chart.append(False)
    
    count_compare = collections.Counter(compare_chart)
    logging.debug('count True/False = ' + str(count_compare))
    accuracy = count_compare[True] / sumaCounter(count_compare)
    print 'accuracy = ' + str(accuracy)
    logging.debug('accuracy = ' + str(accuracy))
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
def getTestReviewCounter(domain, removing_words,segment,sum_counter_testrev):
    import collections
    import re
    (trainDomain,testDomain) = divideDomainCrossValidation(domain,segment)
    test_rev = formatDomain(testDomain)
    testrev_counter = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in test_rev]
    clean_testrev_counter = []
    total_ccbw = []
    for ct in testrev_counter:
        ccbw = cleanCounterBagofWords(ct[1],removing_words)
        total_ccbw.append(ccbw)
        clean_testrev_counter.append((ct[0],ccbw))
    sum_counter_testrev = sum(total_ccbw,collections.Counter())
    return clean_testrev_counter
def getMkb(past_data):
    import collections
    Mkb_p = collections.Counter()
    Mkb_n = collections.Counter()
    for domain_data in past_data:
        (Pp,Pn) = calcNaiveBayesProb(domain_data[2],domain_data[3],domain_data[7])
        for word in domain_data[7]:
            if (Pp>Pn):
                Mkb_p[word] = Mkb_p[word]+1
            elif (Pn>Pp):
                Mkb_n[word] = Mkb_n[word]+1
    return (Mkb_p,Mkb_n)
def demoBalanced19(natural):
    import numpy as np
    import collections
    import random
    logging.debug("natural distribution = "+str(natural))
    domainFiles = getDomainFileNames()
    cached_data = []
    cached_domain_s = []
    for df in domainFiles:
        cached_domain = getDomainfromFile(df)
        if natural == False:
            cached_balanced_domain = formatDomainBalancing(cached_domain)
        else: 
            cached_balanced_domain = formatDomainNatural(cached_domain)
        if cached_balanced_domain != False:
            cached_domain_s.append(cached_balanced_domain)
            cached_data.append(getDomainData(cached_balanced_domain,get_removing_words()))
        else:
            cached_domain_s.append(None)
            cached_data.append(None)
    n_bd = len(cached_domain_s)
    past_data = []
    total_accuracy = []
    total_result = []
    for i in range(0,n_bd):
        if cached_domain_s[i]!=None:
            #prepare past data
            for j in range(0,n_bd):
                if j!=i:
                    if cached_data[j]!=None:
                        past_data.append(cached_data[j])
            accuracy = []
            M_plus = collections.Counter()
            M_minus = collections.Counter()
            (M_plus, M_minus) = getMkb(past_data)
            vocab_S = getVocabSRegularization(M_plus,M_minus)
            for s in range(0,5):
                sum_counter_testrev = collections.Counter()
                clean_testrev_counter = getTestReviewCounter(cached_domain_s[i],get_removing_words(),s,sum_counter_testrev)
                #reason to all the problem
                (train_domain, test_domain)=divideDomainCrossValidation(cached_domain_s[i],s)
                past_data.append(getDomainData(train_domain,get_removing_words()))
                vocab_T = getVocabTRegularization_NaiveProb(past_data[-1][2],past_data[-1][3],past_data[-1][7],6)
                X = []
                X.append(initializeX(past_data))
                (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J) = evalLLLValues(1,past_data,X[0])
                J_s = []
                J_s.append(J)
                for n in range(0,len(past_data[-1][6])):
                    X.append(updateXstochastic(past_data[-1][5],X[-1][0],X[-1][1],past_data[-1][6][n],past_data[-1][4],vocab_T,past_data[-1][2],past_data[-1][3],vocab_S,M_plus,M_minus,X[0][0],X[0][1]))
                    (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J) = evalLLLValues(1,past_data,X[-1])
                    J_s.append(J) 
                    #ASSUME                   
                    if np.math.isnan(J_s[-1])==False and np.math.isnan(J_s[-2])==False: 
                        print "update n time = "+str(n)
                        logging.debug("update n time = "+str(n))
                        break
                ratio = past_data[-1][4]
                #test_data
                result = testing(Pword_plus,Pword_minus,clean_testrev_counter,ratio,get_removing_words())
                accuracy.append(result[0])
                if result[0]<0.79:
                    logging.debug("J< 0.79 "+domainFiles[i]+" segment = "+str(s)+" X+ " + str(X[-1][0].most_common(5))+" X- "+str(X[-1][1].most_common(5))+" sentence "+str(train_domain[0]))
                    logging.debug('X[0] '+str(X[0][0].most_common(5))+str(X[0][1].most_common(5)))
                    logging.debug("test domain "+str(sum_counter_testrev.most_common(15)))
            avg_accuracy = sum(accuracy)/len(accuracy)
            total_accuracy= total_accuracy+accuracy
            total_result.append((domainFiles[i],avg_accuracy))
            print domainFiles[i]+' '+str(avg_accuracy) + ' avg now '+ str(sum(total_accuracy)/len(total_accuracy))
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
def calcNaiveBayesProb(pos_count,neg_count,vocab_list):
    import sys
    import collections
    total_plus = sumaCounter(pos_count)
    total_minus = sumaCounter(neg_count)
    V = len(vocab_list)
    Pp = collections.Counter()
    Pn = collections.Counter()
    for word in vocab_list:
        try:
            Pp[word] = (pos_count[word]+1)/(total_plus+V)
            Pn[word] = (neg_count[word]+1)/(total_minus+V)
        except:
            print sys.exc_info()
            Pp[word] = 1/(total_plus+V)
            Pn[word] = 1/(total_minus+V)
    return (Pp,Pn)
def getVocabSRegularization(Mkb_p,Mkb_n):
    import collections
    tau = 6
    vocab_S = []
    if (type(Mkb_p)!=type(collections.Counter()) or type(Mkb_n)!=type(collections.Counter())):
        print "Mkb wrong type"
    vocab_list = (Mkb_p + Mkb_n).keys()
    for word in vocab_list:
        if Mkb_p[word]>6 or Mkb_n[word]>6:
            vocab_S.append(word)
    print "added to Vocab S "+str(len(vocab_S))
    return vocab_S
def getVocabTRegularization_NaiveProb(pos_count,neg_count,vocab_list, delta):
    (Pp, Pn) = calcNaiveBayesProb(pos_count,neg_count,vocab_list)
    vocab_T = []
    for word in vocab_list:
        if (Pp[word]/Pn[word] > delta):
            vocab_T.append(word)
        elif (Pn[word]/Pp[word] > delta):
            vocab_T.append(word)
    print "added to vocab_T "+str(len(vocab_T))
    return vocab_T 
def resetLogging():
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers)>0:
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])
    filename = 'log'+str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))+'.txt'
    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(logging.DEBUG)
    #formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
    formatter = logging.Formatter("%(asctime)s %(funcName)s : %(message)s %(lineno)d, %(filename)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)   
    logging.debug(sys.prefix)
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()      
if __name__ == '__main__':
    import datetime
    import sys
    resetLogging()
    f = open('console'+str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))+'.txt', 'a')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    origin_time = datetime.datetime.now().time()
    print origin_time
    demoBalanced19(False)
    f.close()
    print str(datetime.datetime.now().time()) + " origin "+ str(origin_time)
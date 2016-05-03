from __future__ import division
from __future__ import print_function
import logging
import datetime
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
    doit('dua nhau di tron')
    #with open('log.txt','a') as myfile:
        #print >>myfile, function_name + '\t' + content + '\t' + str(datetime.datetime.now().time())
def sumaCounter(counter):
    return sum(counter.itervalues())
    #return sum(counter.values())
def cleanCounterBagofWords(bag_counter,removing_word_list,nanwordlist):
    import numpy as np
    bag = bag_counter
    nanwordlist = []
    for y in bag_counter.keys():
        if y in removing_word_list:
            del bag_counter[y]
        elif np.math.isnan(bag_counter[y]):
            del bag_counter[y]
            nanwordlist.append(y)
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
    #ASSUME: all words are lowered and treated the same way
    #LANGUAGE #UNIGRAM
    if getConstant('VN') == True:
        #IMPROVE isalnum()
        #bag_of_words = [collections.Counter([x for x in txt.lower().split() if x.isalnum()]) for txt in reviews]
        bag_of_words = [collections.Counter(splitVietnamese(txt,True)) for txt in reviews]
        print('passed')
    else:
        
        if getConstant('bigram') ==True:
            bag_of_words = [collections.Counter(getBigramsFromListofUnigrams(re.findall(r'\w+',txt.lower()))) for txt in reviews]
        else:
            bag_of_words = [collections.Counter(re.findall(r'\w+',txt.lower())) for txt in reviews]
        #bag_of_words = [collections.Counter([x for x in re.findall(r'\w+',txt.lower()) if x not in get_removing_words()]) for txt in reviews]
    sum_bag = sum(bag_of_words,collections.Counter())
    return sum_bag
def getDomainName(filename):
    if getConstant('VN')==False:
        spl = filename.split('.')
        return spl[0] 
    else:
        spl = filename.split('/')
        return spl[-2].split('.')[0]
def readDomainToList(filename):
    import codecs
    if getConstant('VN')==False:
        with open(filename) as myfile:
            domain = myfile.readlines()
        del domain[0]
        return domain
    else:
        with codecs.open(filename,mode='r',encoding='utf-8') as myfile:
            temp_domain = myfile.readlines()
            domain = []
            pos, neu, neg = 0,0,0
            for d in temp_domain:
                to_append = True
                d_s = d.split('\t')
                if len(d_s) <8:
                    try:
                        print((filename+' '+str(temp_domain.index(d))+' '+str(d_s)+' wrong split <8').encode('utf-8'))
                        continue
                    except:
                        continue
                if len(d_s) > 8:
                    try:
                        print ((filename+' '+str(temp_domain.index(d))+' '+str(d_s)+' wrong split >8').encode('utf-8'))
                    except:
                        print('EXCEPT '+str(sys.exc_info()))
                    rv = ''.join(d_s[7:])
                else:
                    rv = d_s[7]
                if d_s[2] == '100' or d_s[2] == '80':
                    polarity = 'POS'
                    if pos >= 100:
                        to_append = False
                    else: 
                        pos = pos +1
                elif d_s[2] == '60':
                    polarity = 'NEU'
                    if neu>=50:
                        to_append = False
                    else:
                        neu = neu +1
                elif d_s[2] == '20' or d_s[2]== '40':
                    polarity =  'NEG'
                    if neg >=100:
                        to_append = False
                    else:
                        neg = neg +1
                else:
                    print(' wrong label '+filename+' ' +str(temp_domain.index(d))+' '+str(d_s))
                str_append = d_s[0]+'\t'+polarity+'\t'+d_s[2]+'\t'+rv
                if to_append == True:
                    domain.append(str_append)
            return domain
def divideDomain(domain):
    
    pos_rev = []
    neg_rev = []
    bias_rev = []
    total_rev = []
    temp = domain[0].split('\t')
    try:
        domain_name = temp[0]
    except:
        print('DIVIDE DOMAIN domain_name = temp[0]');   print(temp)
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
    return (pos_rev,neg_rev,bias_rev,total_rev,domain_name)
 
def initializeX(data):
    import collections
    X_plus = collections.Counter()
    X_minus = collections.Counter()
    for dm in data:
        if type(dm[2])==collections.Counter and type(dm[3])==collections.Counter:
            X_plus = X_plus + dm[2]
            X_minus = X_minus + dm[3]
        else:
            print('type error at position = '+str(data.index(dm)))
    #FIXED the domain name problem
    logging.debug("initialize X "+str(data[-1][0])+" PLUS "+str(X_plus.most_common(10))+" MINUS "+str(X_minus.most_common(10)))
    return (X_plus, X_minus)
def getDomainfromFile(filename):
    domainName = getDomainName(filename)
    domain = readDomainToList(filename)
    return domain
def formatDomainNatural(domain):
    natural_rev = []
    pos = 0
    neg = 0; neu =0
    print(domain[0].encode("utf-8")); print('format Domain Natural')
    #raw_input()
    for x in domain:        
        spl = x.split('\t')        
        if len(spl)==4:
            if spl[1] =='POS':
                pos = pos+1
                natural_rev.append(x)
            elif spl[1] =='NEG':
                neg = neg +1
                natural_rev.append(x)
            elif spl[1] == 'NEU':
                neu = neu+1
    logging.debug(' pos '+str(pos)+' neg '+str(neg)+' neu '+str(neu) )
    return natural_rev
def formatDomainBalancing(domain):
    balanced_rev = []
    pos = 0
    neg = 0
    #DOMAIN DEPENDENT
    threshold = 100
    for x in domain:
        spl = x.split('\t')        
        if len(spl)==4:
            if spl[1] =='POS':
                #DOMAIN DEPENDENT
                if pos<100:
                    pos = pos+1
                    balanced_rev.append(x)
            elif spl[1] =='NEG':
                if neg<100:
                    neg = neg +1
                    balanced_rev.append(x)
    if pos !=threshold or neg!=threshold:
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
def formatDomainUnchanged(testDomain):
    clean_rev = []
    for x in testDomain:
        spl = x.split('\t')        
        if len(spl) == 4:
            clean_rev.append((spl[1],spl[3].strip('\n')))
    return clean_rev
 
def getDomainData(domain,removing_words):
    import collections
    import re
    pos_rev = []
    neg_rev = []
    bias_rev = []
    total_rev = []
    (pos_rev,neg_rev,bias_rev,total_rev,domainName) = divideDomain(domain)
    l_n_r = len(neg_rev)
    if l_n_r == 0:
        l_n_r = 0.01
    ratio = len(pos_rev) / l_n_r
    pos_bag = []
    neg_bag = []
    #UNIGRAM
    pos_bag = getBagofWordsfromReviews(pos_rev)
    neg_bag = getBagofWordsfromReviews(neg_rev)
    #LANGUAGE
    if getConstant('VN')==True:
        #bias_count = [(txt[0],collections.Counter(txt[1].split())) for txt in bias_rev]
        #clean_count = [(txt[0],collections.Counter(txt[1].lower().split())) for txt in bias_rev]
        bias_count = [(txt[0],collections.Counter(splitVietnamese(txt[1],False))) for txt in bias_rev]
        clean_count = [(txt[0],collections.Counter(splitVietnamese(txt[1],True))) for txt in bias_rev]
    else:
        bias_count = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1]))) for txt in bias_rev]
        if getConstant('bigram')!=True:
            clean_count = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in bias_rev]
        else:
            clean_count = [(txt[0],collections.Counter(getBigramsFromListofUnigrams( re.findall(r'\w+',txt[1].lower())))) for txt in bias_rev]
    if getConstant('lower') == True:
        bias_count = clean_count
    clean_biascount = []
    nanwordlist = []
    for c in clean_count:
        clean_biascount.append((c[0],cleanCounterBagofWords(c[1],removing_words,nanwordlist)))
    #UNIGRAM
    pos_count = cleanCounterBagofWords(pos_bag,removing_words,nanwordlist)
    if len(nanwordlist)>0:
        msg = "domain NAN pos "+str(domainName); print(msg)
        msg = "nan word list - POS " + str(nanwordlist); logging.debug(msg)
    neg_count = cleanCounterBagofWords(neg_bag,removing_words,nanwordlist)
    if len(nanwordlist)>0:
        msg = "domain NAN neg "+str(domainName); print(msg)
        msg = "nan word list - NEG " + str(nanwordlist); logging.debug(msg)
    vocab_list = (pos_count + neg_count).keys()
    vocab = len(vocab_list)
    if (type(pos_count)!=collections.Counter or type(neg_count)!=collections.Counter):
        print(getDomainData.__name__ + ' wrong type')
    return (domainName,bias_count,pos_count,neg_count, ratio,vocab,clean_biascount,vocab_list)
 
def getObjectiveFunctionDiff(word,document_counter,vocab,smooth,ratio,X_plus,X_minus, sum_Xplus, sum_Xminus,vocab_T,pos_count,neg_count,di,vocab_S,sum_penalty_Vs,R_Mkb,X_zeroplus,X_zerominus):
    from sympy import mpmath
    import numpy as np
    import sys
    mpmath.mp.pretty = True
    isnan = False
    #np.seterr(all='print')
    fraction_nplus = document_counter[word] / (smooth + X_plus[word])
    if np.math.isnan(fraction_nplus):
        try:
            print("NAN fraction_nplus "+str(word))
        except:
            logging.debug(str(sys.exc_info()))
    fraction_nminus = document_counter[word] / (smooth + X_minus[word])
    if np.math.isnan(fraction_nminus):
        try:
            print("NAN fraction_nminus "+str(word))
        except:
            logging.debug(str(sys.exc_info()))
    #INDEPENDENT
    try:
        l_g = (np.log(vocab * smooth + sum_Xplus) - np.log(vocab * smooth + sum_Xminus)) * di
        g = np.exp(l_g)
    except OverflowError as e:
        print("over flow g vocab "+str(vocab)+" sumXplus "+str(sum_Xplus)+" sumXminus "+str(sum_Xminus)+" di "+str(di))
        return (np.float64(0),np.float64(0))
    if np.math.isnan(g):
        try:
            print("NAN g "+str(word))
        except:
            logging.debug('EXCEPT '+str(sys.exc_info()))
    l_product_pi = np.log(1)
    for key in document_counter.keys():
        #DOUBTS
        l_product_pi = l_product_pi + (np.log(smooth+X_minus[key])-np.log(smooth+X_plus[key]))*document_counter[key]
    product_pi = np.exp(l_product_pi)
    if np.math.isnan(product_pi):
        try:
            print("NAN product pi "+str(word))
        except:
            logging.debug(str(sys.exc_info()))
    g_diffpos = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus - X_plus[word] + x) / (smooth * vocab + sum_Xminus)) ** di,X_plus[word])
    g_diffneg = mpmath.diff(lambda x: ((smooth * vocab + sum_Xplus) / (smooth * vocab + sum_Xminus - X_minus[word] + x)) ** di,X_minus[word])
    if np.math.isnan(g_diffpos):
        try:
            print("NAN g diff pos "+str(word))
        except:
            print('EXCEPT '+str(sys.exc_info()))
    if np.math.isnan(g_diffneg):
        try:
            print("NAN  g_diff_neg"+str(word))
        except:
            print('EXCEPT '+str(sys.exc_info()))
    sum_penalty_Vt = 0
    #regularization term 1
    #INDEPENDENT
    for w in vocab_T:
        sum_penalty_Vt = sum_penalty_Vt + (X_plus[w]-pos_count[w])**2 +(X_minus[w]-neg_count[w])**2
    #FIX vocab limitation
    if word in vocab_T:
        penalty_Vt_plus = mpmath.diff(lambda x: 0.5*0.1*(sum_penalty_Vt -( X_plus[word]-pos_count[word])**2+(x-pos_count[word])**2),X_plus[word])
        penalty_Vt_minus = mpmath.diff(lambda x: 0.5*0.1*(sum_penalty_Vt -( X_minus[word]-neg_count[word])**2+(x-neg_count[word])**2),X_minus[word])
        if np.math.isnan(penalty_Vt_minus) or np.math.isnan(penalty_Vt_plus):
            penalty_Vt_plus, penalty_Vt_minus = 0,0
            msg = 'penalty Vs nan (+) '+str(penalty_Vt_plus) + ' (-) '+str(penalty_Vt_minus); logging.debug(msg)
    else:
        penalty_Vt_plus = 0
        penalty_Vt_minus = 0
    #regularization term 2
    if word in vocab_S:
        penalty_Vs_plus = mpmath.diff(lambda x:0.5*0.1*(sum_penalty_Vs-(X_plus[word]-R_Mkb[word]*X_zeroplus[word])**2),X_plus[word])
        penalty_Vs_minus = mpmath.diff(lambda x:0.5*0.1*(sum_penalty_Vs-(X_minus[word]-(1-R_Mkb[word])*X_zerominus[word])**2),X_minus[word])
        if np.math.isnan(penalty_Vs_minus) or np.math.isnan(penalty_Vs_plus):
            penalty_Vs_plus, penalty_Vs_minus = 0,0
            msg = 'penalty Vs nan (+) '+str(penalty_Vs_plus) + ' (-) '+str(penalty_Vs_minus); logging.debug(msg)
    else:
        penalty_Vs_plus = 0
        penalty_Vs_minus  = 0
    normal_nan_ignore = True
    if fraction_nplus > 0 and fraction_nminus > 0 and g_diffpos>0 and g_diffneg>0 :
        lcntr_dFp = logaritSum(np.log(fraction_nplus),np.log(ratio**-1)+l_product_pi + np.log(g_diffpos)) - np.log(1+np.exp(np.log(ratio**-1)+l_product_pi+np.log(g)))
        lcntr_dFm = logaritSum(np.log(fraction_nminus)+np.log(g),np.log(g_diffneg)) - logaritSum(np.log(ratio) + l_product_pi*-1,np.log(g))
        if penalty_Vt_plus != 0 and penalty_Vs_plus != 0 and lcntr_dFp>np.log(fraction_nplus):
            normal_nan_ignore = False
            l_dFp_full = logaritSubtract(lcntr_dFp,np.log(fraction_nplus))
            l_pen_p = np.log(penalty_Vt_plus + penalty_Vs_plus)
            l_pen_m = np.log(penalty_Vt_minus + penalty_Vs_minus)
            dFplus = np.exp(logaritSum(l_dFp_full,l_pen_p))
            dFminus = np.exp(logaritSum(lcntr_dFm,l_pen_p))
    if normal_nan_ignore == True or normal_nan_ignore!=False:
        dFplus = (fraction_nplus + (ratio ** -1) * product_pi * g_diffpos) / (1 + (ratio ** -1) * product_pi * g) - fraction_nplus + penalty_Vt_plus + penalty_Vs_plus
        dFminus = (fraction_nminus * g + g_diffneg) / (ratio * (product_pi ** -1) + g) + penalty_Vt_minus + penalty_Vs_minus
    if np.math.isnan(dFplus):
        dFplus = 0
    if np.math.isnan(dFminus):
        dFminus = 0
    return (np.float64(dFplus),np.float64(dFminus)) 
def get_sum_penalty_Vs(X_plus,X_minus,R_Mkb,X_zeroplus,X_zerominus,vocab_S):
    sum_penalty_Vs = 0
    for w in vocab_S:
        sum_penalty_Vs = sum_penalty_Vs + (X_plus[w]-R_Mkb[w]*X_zeroplus[w])**2+(X_minus[w]-(1-R_Mkb[w])*X_zerominus[w])**2
    return sum_penalty_Vs
def get_R_Mkb(Mkb_p,Mkb_n,vocab_S):
    import collections
    R = collections.Counter()
    #UNIGRAM
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
    #IMPROVE catch the nan problem
    for key in clean_document[1].keys():
        if getConstant('update X hard')==True and key not in (X_plus + X_minus).keys():
            dF = (0,0)
            try:
                print(("skipped "+str(key)+" - updateXhard mode").encode('utf-8'))
            except:
                print(str(sys.exc_info()))
        else:
            dF = getObjectiveFunctionDiff(key,clean_document[1],vocab,1,ratio,X_plus,X_minus,sum_Xplus,sum_Xminus,vocab_T,pos_count,neg_count,di,vocab_S,sum_penalty_Vs,R_Mkb,X_zeroplus,X_zerominus)
        new_Xplus[key] = X_plus[key] - dF[0]
        new_Xminus[key] = X_minus[key] - dF[1]
    #FUN FACT: nan problem fixed
    for key in new_Xminus.keys():
        if np.math.isnan(new_Xminus[key]):
            new_Xminus[key] = X_minus[key]
            msg = 'X- key is nan ' +str(key); logging.debug(msg)
    for key in new_Xplus.keys():
        if np.math.isnan(new_Xplus[key]):
            new_Xplus[key] = X_plus[key]
            msg = 'X+ key is nan ' +str(key); logging.debug(msg)
    return (new_Xplus,new_Xminus) 
def calcProbabilityVocab(smooth, vocab_list, X_plus, X_minus, sum_Xplus, sum_Xminus):
    from itertools import islice
    import numpy as np
    Pword_plus = {}
    Pword_minus = {}
    eword_plus, eword_minus = {}, {}
    nanwordlist = []
    message = ''
    V = len(vocab_list)  #check vocab_list
    for word in vocab_list:
        #IMPROVE predict the nan cases and stop
        Pword_plus[word] = (smooth + X_plus[word]) / (smooth * V + sum_Xplus)
        eword_plus[word] = np.log(Pword_plus[word])
        Pword_minus[word] = (smooth + X_minus[word]) / (smooth * V + sum_Xminus)
        eword_minus[word] = np.log(Pword_minus[word])
        if np.math.isnan(eword_plus[word]) == True or np.math.isnan(eword_minus[word]) == False:
            nanwordlist.append(word)
    if len(nanwordlist) >0:
        #logging.debug('Pw+ ' + str(list(islice(Pword_plus.iteritems(),10)))+' Pw- ' + str(list(islice(Pword_minus.iteritems(),10))))
        msg = 'nan word list l='+str(len(nanwordlist))+str(nanwordlist)
    return (eword_plus,eword_minus,nanwordlist) 
def calcProbabilityDocument(Pword_plus,Pword_minus,document_counter,ratio,removing_words):
    Pdoc_plus = ratio
    Pdoc_minus = ratio ** -1
    import numpy as np
    from itertools import islice
    ep, em = np.log(Pdoc_plus), np.log(Pdoc_minus)
    ep0, em0 = ep,em
    skipped = [False,False]
    for word in document_counter.keys():
        if word in removing_words:
            continue
        #ASSUME: =>no P word is nan
        if word in Pword_plus.keys():
            if np.math.isnan(Pword_plus[word])==True:
                skipped[0] = True
                continue
            ewp = Pword_plus[word]
            ep = ep + ewp * document_counter[word]
        else:
            doit('word ' + word + ' not in Pword_plus')
        if word in Pword_minus.keys():
            if np.math.isnan(Pword_minus[word])==True:
                skipped[1] = True
                continue
            ewm = Pword_minus[word]
            em = em + ewm * document_counter[word]
        else:
            doit('word ' + word + ' not in Pword_minus')  
    if skipped[0] == True or skipped[1] ==True:
        msg = 'Pword skipped' + str(skipped) ; logging.debug(msg)
    return (ep,em) 
def evalObjectiveFunction(clean_biascount,Pword_plus,Pword_minus,ratio,removing_words):
    import numpy as np
    import math
    import sys
    np.seterr(all='ignore')
    #Obj = np.log(1)
    Obj = 0
    wrong_pdoc_biased_count = 0
    for doc in clean_biascount:
        Pdoc = calcProbabilityDocument(Pword_plus,Pword_minus,doc[1],ratio,removing_words)
        i = clean_biascount.index(doc) 
        #ASSUME
        if np.math.isnan(Pdoc[0])==True or np.math.isnan(Pdoc[1])==True:
            logging.debug("Pdoc nan")
            Obj = np.nan
            break
            return Obj
        (bigger, smaller) = (Pdoc[0],Pdoc[1])
        wrong_pdoc_biased = False
        #ASSUME
        absolute_mode = getConstant('absolute F')
        signObj = 1
        #IMPROVE wrong probabilty document
        if doc[0]==True and Pdoc[0]>Pdoc[1]:
            (bigger, smaller) = (Pdoc[0],Pdoc[1])
        elif doc[0]==False and Pdoc[0]<Pdoc[1]:
            (bigger, smaller) = (Pdoc[1],Pdoc[0])
        else:
            #msg =  "wrong Pdoc biased "+str(i) ;    logging.debug(msg)
            wrong_pdoc_biased = True
            wrong_pdoc_biased_count = wrong_pdoc_biased_count +1
            if Pdoc[1]>Pdoc[0]:
                (bigger,smaller) = (Pdoc[1],Pdoc[0])
        if absolute_mode == True:
            if i == 0:
                Obj = logaritSubtract(bigger,smaller)
            else:
                temp = logaritSubtract(bigger,smaller)
                Obj = logaritSum(Obj,temp)
        else:
            signtemp = 1
            if wrong_pdoc_biased == False:
                signtemp = -1
            temp = logaritSubtract(bigger,smaller)
            if i == 0:
                (Obj,signObj) = (temp,signtemp)
            else:
                (Obj,signObj) = logaritSum_sign(Obj,signObj,temp,signtemp)
            if np.absolute(signObj) !=1:
                logging.debug("wrong sign sign temp "+str(signtemp)+" sign Obj "+str(signObj))
                signObj = fixSign(signObj)
        if Obj == 0.0:
            logging.debug( "Obj=0 fuck " + str(i) + "")
    #IMPROVE J nan
    if math.isnan(Obj)==False:
        msg = "^Obj = " + str(Obj)+" J= "+str(signObj)+"*"+str(np.exp(Obj))
        print(msg); logging.debug(msg)
    else:
        logging.debug("J nan")
    logging.debug("wrong biased times = "+str(wrong_pdoc_biased_count))
    return (Obj,signObj)
def logaritSum(log_a,log_b):
    import numpy as np
    r =  log_a + np.log(1+np.exp(log_b - log_a))
    if np.math.isnan(r):
        msg = "log sum nan"; logging.debug(msg)
    return r
def fixSign(sign):
    if sign>=0:
        nsign = 1
    else:
        nsign = -1
    return nsign
def logaritSum_sign(log_a,sign_a,log_b,sign_b):
    import numpy as np
    r = 0
    signr = 1
    #FIX wrong sign error
    if np.absolute(sign_a)!=1 or np.absolute(sign_b)!=1:
        msg= "wrong sign"
        print(msg); logging.debug(msg)
    if sign_a * sign_b >0:
        r = logaritSum(log_a,log_b)
        signr = sign_a
    else:
        r = logaritSubtract(log_a,log_b)
        if log_a>log_b:
            signr = sign_a
        else:
            signr = sign_b
    return (r,signr)
def logaritSubtract(log_a,log_b):
    import numpy as np
    if log_a == log_b:
        return 0
    if log_a > log_b:
        a = log_a; b = log_b
    elif log_a < log_b:
        a = log_b; b = log_a
        #msg = "reverse" ; print msg
    r =  a + np.log(1-np.exp(b-a))
    if np.math.isnan(r):
        msg = 'log sub nan'; logging.debug(msg)
    return r
def get_removing_words():
    meaningless = ['lrb','rrb','would']
    return meaningless + getStrStopwords()
def testingF1(Pword_plus,Pword_minus,testdata,ratio,removing_w):
    import numpy as np
    import collections
    bias_testdata = []
    pos_no = 0
    neg_no = 0
    neu_no = 0
    for x in testdata:
        if x[0] == 'NEU':
            neu_no = neu_no +1
        elif x[0] == 'POS':
            pos_no = pos_no + 1
        elif x[0] == 'NEG':
            neg_no = neg_no + 1
    logging.debug('NEU '+str(neu_no)+ ' POS '+str(pos_no)+' NEG '+str(neg_no))
    if len(testdata)!= neu_no+pos_no+neg_no:
        print("Wrong count testing F1 ")
    compare_chart = []
    full_guessing_chart = []
    correct_negative = 0
    correct = 0
    guess_negative = 0
    total_reviews = len(testdata)
    for x in testdata:
        if getConstant('neutral') == False:
            if x[0] == 'NEU':
                continue
        Pd = calcProbabilityDocument(Pword_plus,Pword_minus,x[1],ratio,removing_w)
        guess = ''
        #FIX nan problem        
        if np.float64(Pd[0]) > np.float64(Pd[1]):
            guess = 'POS'
        else:
            guess = 'NEG'
            guess_negative = guess_negative +1
        full_guessing_chart.append((x[0],guess))   
        if x[0] == 'NEG':
            if guess == x[0]:     
                correct_negative = correct_negative +1
        if x[0] == guess:
            correct = correct +1        
    precision_1 = correct_negative/total_reviews
    precision_2 = correct_negative/(pos_no+neg_no)
    precision_3 = correct / (pos_no + neg_no)
    accuracy = correct/len(testdata)
    #ASSUMPTION: Zero Division
    try:
        precision_0 = correct_negative / guess_negative
    except ZeroDivisionError as e:
        print('correct negative '+str(correct_negative)+' guess_negative '+str(guess_negative))
        if correct_negative == 0:
            precision_0 = 1
        else: 
            precision_0 = 0
    if neg_no == 0:
        if correct_negative == 0:
            recall = 1
        else:
            recall = 0
    else:
        recall = correct_negative/neg_no
    #f1_correct = (2*precision_1 * recall)/(precision_1 + recall)
    #f1_unsure = (2*precision_2*recall)/(precision_2+recall)
    #f1_accuracy = (2*precision_3*recall)/(precision_3 + recall)
    if precision_0 + recall == 0:
        f1_drminh = 0
    else:
        f1_drminh = (2*precision_0*recall)/(precision_0 + recall)
    #msg =  'accuray = '+precision_3+' F1 score = ' + str(f1_correct) + ' F1 assumption '+str(f1_unsure) + ' F1 accuracy '+str(f1_accuracy)
    if getConstant('tab chart')==True:
        if getConstant('micro average') == True:
            result = (precision_0,f1_drminh,recall,correct_negative,guess_negative,neg_no,accuracy)
        else:
            result = (precision_0,f1_drminh,recall)
    else:
        result = (precision_0,f1_drminh)
    msg = " all F1 score = "+str(result)+ ' precision '+str(precision_0)+ ' recall ' + str(recall)
    print(msg); logging.debug(msg)
    return result
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
        print("false counting bias_testdata")
    compare_chart = []
    full_guessing_chart = []
    for x in bias_testdata:
        Pd = calcProbabilityDocument(Pword_plus,Pword_minus,x[1],ratio,removing_w)
        guess = ''; guess_analysis = ()
        if getConstant('one domain') == True:
            Pp_relevant = {key:val for key, val in Pword_plus.items() if key not in x[1].keys()}
            Pn_relevant = {key:val for key, val in Pword_minus.items() if key not in x[1].keys()}
            guess_analysis = (Pp_relevant,Pn_relevant)
        #FIX nan problem
        if np.float64(Pd[0]) > np.float64(Pd[1]):
            guess = 'POS'
        else:
            guess = 'NEG'
        r = True
        if guess == x[0]:
            compare_chart.append(True)
        else:
            compare_chart.append(False); r = False
        if getConstant('one domain') == True:
            full_guessing_chart.append((r,Pd,x,guess_analysis))    
    count_compare = collections.Counter(compare_chart)
    logging.debug('count True/False = ' + str(count_compare))
    accuracy = count_compare[True] / sumaCounter(count_compare)
    if getConstant('one domain') == True:
        if accuracy< 0.8334:
            printArrayLineNo('lowacc_guessing'+str(accuracy),PublicValues.origin_time_string,full_guessing_chart)
    print('accuracy = ' + str(accuracy))
    logging.debug('accuracy = ' + str(accuracy))
    return (accuracy,compare_chart)
def getDomainFileNames():
    from os import listdir
    from os.path import isfile,join    
    if getConstant('VN') == False:
        return [
                'CableModem.txt',
                'AlarmClock.txt',
                'Baby.txt',
                'Bag.txt',            
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
                'RiceCooker.txt',
                'Sandal.txt',
                'Vacuum.txt',
                'Video_Games.txt',
                'Projector.txt'
                ]
    else: 
        if getConstant('segmentation')==True:
            mypath = 'crawling/1a3combined14MarPM190828/'
            if getConstant('17')==True:
                mypath = 'crawling/1a4/'
        else:
            mypath = 'crawling/1a3/'
            if getConstant('17')==True:
                mypath = 'crawling/1a5/'
        onlyfiles = [f for f in listdir(mypath) if isfile(mypath+f)]
        return [mypath+f for f in onlyfiles]
def divideDomainCrossValidation_KeepRatio(domain):
    import collections
    import re
    import sys
    segments = []
    pos_domain = []; neg_domain = []
    pos, neg = 0,0
    try:
        print(domain[0]); print('dividedomaincrossvalidation_keepratio')
    except:
        print('EXCEPT dividedomaincrossvalidation_keepratio ' +str(sys.exc_info()))
    for d in domain:
        d_s  = d.split('\t')
        if len(d_s) == 4:
            label = d_s[1]
            if label == 'POS':
                pos = pos +1
                pos_domain.append(d)
            elif label == 'NEG':
                neg = neg + 1
                neg_domain.append(d)
        else:
            logging.debug('wrong split ' +str(d_s))
    pos_seg, pos_left = int(pos/5), pos%5
    neg_seg, neg_left = int(neg/5), neg%5
    start_p, start_n = 0,0
    for i in range(0,5):
        temp = []
        for j in range(start_p, (i+1)*pos_seg-1):
            temp.append(pos_domain[j])
        start_p = (i+1)*pos_seg
        for k in range(start_n, (i+1)*neg_seg-1):
            temp.append(neg_domain[k])
        start_n = (i+1)*neg_seg
        segments.append(temp)
    for i in range(0,pos_left):
        segments[i].append(pos_domain[start_p+i])
    for i in range(0,neg_left):
        segments[i].append(neg_domain[start_n+i])
    print('len segment '+str([len(x) for x in segments]))
    return segments
def getReviewCounterFromSegment(segment, removing_words):
    #segment doesn't have NEUTRAL
    import re
    import collections
    t_rev = formatDomainUnchanged(segment)
    #LANGUAGE #UNIGRAM
    if getConstant('VN') ==True:
        #rev_counter = [(txt[0],collections.Counter(txt[1].lower().split())) for txt in t_rev]
        rev_counter = [(txt[0],collections.Counter(splitVietnamese(txt[1],True))) for txt in t_rev]
    else:
        if getConstant('bigram')==True:
            rev_counter = [(txt[0],collections.Counter(getBigramsFromListofUnigrams( re.findall(r'\w+',txt[1].lower())))) for txt in t_rev]
        else:
            rev_counter = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in t_rev]
    clean_testrev_counter = []
    total_ccbw = []
    neu_no = 0
    nan_list = []
    for ct in rev_counter:
        ccbw = cleanCounterBagofWords(ct[1],removing_words,nan_list)
        total_ccbw.append(ccbw)        
        clean_testrev_counter.append((ct[0],ccbw))
        if ct[0]=='NEU':
            neu_no =neu_no+1
    return clean_testrev_counter
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
    if getConstant('VN')==True:
        msg = "left "+str(left_pivot)+ " right "+str(right_pivot) + " n fold "+str(n_fold)+" len "+str(len(domain)); print(msg)
    testDomain = domain[left_pivot:right_pivot]
    trainDomain = domain[0:left_pivot] + domain[right_pivot:len(domain)]      
    return (trainDomain,testDomain)
def getFullTestReviewCounter(domain, removing_words,segment,sum_counter_testrev):
    import collections
    import re
    (trainDomain,testDomain) = divideDomainCrossValidation(domain,segment)
    test_rev = formatDomainUnchanged(testDomain)
    #LANGUAGE #UNIGRAM
    if getConstant('VN')==True:
        #testrev_counter = [(txt[0],collections.Counter(txt[1].lower().split())) for txt in test_rev]
        testrev_counter = [(txt[0],collections.Counter(splitVietnamese(txt[1],True))) for txt in test_rev]
    else:
        if getConstant('bigram')==True:
            testrev_counter = [(txt[0],collections.Counter(getBigramsFromListofUnigrams( re.findall(r'\w+',txt[1].lower())))) for txt in test_rev]
        else:
            testrev_counter = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in test_rev]
    clean_testrev_counter = []
    total_ccbw = []
    neu_no = 0
    nan_list = []
    for ct in testrev_counter:
        ccbw = cleanCounterBagofWords(ct[1],removing_words,nan_list)
        total_ccbw.append(ccbw)        
        clean_testrev_counter.append((ct[0],ccbw))
        if ct[0]=='NEU':
            neu_no =neu_no+1
    sum_counter_testrev = sum(total_ccbw,collections.Counter())
    print("neutral of domain " + str(neu_no))
    return clean_testrev_counter
def getBigramsFromListofUnigrams(txt):
    bigram_s = []
    for i in range(0,len(txt)-1):
        bigram_s.append(' '.join(txt[i:i+2]))
    return bigram_s
def splitVietnamese(txt, lower):
    if lower == True:
        t = txt.lower()
    else: 
        t = txt
    s_t = t.split()
    for i in range(0,len(s_t)):
        new_str = ''
        for x in s_t[i]:
            if x.isalnum() or x == '_' or x in getVietnameseTone():
                new_str = new_str +x
        s_t[i] = new_str
    if getConstant('no empty string') == True:
        while 1==1:
            try:
                s_t.remove('')
            except:
                break
        if getConstant('bigram')==True:
            #bigram_s = []
            #for i in range(0,len(s_t)-1):
            #    bigram_s.append(' '.join(s_t[i:i+2]))
            #return bigram_s
            return getBigramsFromListofUnigrams(s_t)
    return s_t
def getTestReviewCounter(domain, removing_words,segment,sum_counter_testrev):
    import collections
    import re
    (trainDomain,testDomain) = divideDomainCrossValidation(domain,segment)
    test_rev = formatDomain(testDomain)
    #LANGUAGE #UNIGRAM
    if getConstant('VN') == True:
        #testrev_counter = [(txt[0],collections.Counter(txt[1].lower().split())) for txt in test_rev]
        testrev_counter = [(txt[0],collections.Counter(splitVietnamese(txt[1],True))) for txt in test_rev]
    else:
        if getConstant('bigram')==True:
            testrev_counter = [(txt[0],collections.Counter(getBigramsFromListofUnigrams(re.findall(r'\w+',txt[1].lower())))) for txt in test_rev]
        else:
            testrev_counter = [(txt[0],collections.Counter(re.findall(r'\w+',txt[1].lower()))) for txt in test_rev]
    clean_testrev_counter = []
    total_ccbw = []
    neu_no = 0
    nan_list = []
    for ct in testrev_counter:
        ccbw = cleanCounterBagofWords(ct[1],removing_words,nan_list)
        total_ccbw.append(ccbw)
        clean_testrev_counter.append((ct[0],ccbw))
        if ct[0]=='NEU':
            neu_no =neu_no+1
    sum_counter_testrev = sum(total_ccbw,collections.Counter())
    #print "neutral of domain " + str(neu_no)
    return (trainDomain,clean_testrev_counter)
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
def printArrayLineNo(filename_prefix,origin_time_string,array_to_print):
    with open('more'+filename_prefix+str(origin_time_string)+'.txt','a') as myfile:
        if len(array_to_print)>0:
            for i in range(0,len(array_to_print)):
                print >>myfile, str(i)+ '\t' + str(array_to_print[i])
                print >>myfile, '///'
        else:
            print('array to print <=0 element')
def averageListTuple(listTuple):
    avg = tuple([sum(y) / len(y) for y in zip(*listTuple)])
    return avg
def getVietnameseTone():
    return ['́', #sắc
            '̣', #nặng
            '̉',#hỏi
            '̀' ,#huyền
            '̃' #ngã
            ]
def demoBalanced19(origin_time_string):
    import numpy as np
    import collections
    import random
    import codecs
    import os
    import sys
    natural = getConstant("natural distribution")
    logging.debug("natural distribution = "+str(natural))
    domainFiles = getDomainFileNames()
    if getConstant('shuffle')==True and getConstant('write shuffle')==False and natural==False:
        for df in domainFiles:
            df = 'shuffle_balanced/' + df #DOMAIN DEPENDENT
    cached_data = []
    cached_domain_s = []
    pure_cached_domain_s = []
    for df in domainFiles:
        #GET FROM SHUFFLED DOMAIN
        cached_domain = getDomainfromFile(df)
        logging.debug(df + ' formatting')
        if getConstant('shuffle')==True and getConstant('write shuffle')==True:
            random.shuffle(cached_domain)
        #UNIGRAM        
        if natural == False:
            cached_balanced_domain = formatDomainBalancing(cached_domain)
        else: 
            cached_balanced_domain = formatDomainNatural(cached_domain)
        #   WRITE SHUFFLE DOMAIN
        if getConstant('shuffle')==True and getConstant('write shuffle')==True and natural==False:
            current_directory = os.path.dirname(os.path.realpath('__file__'))

            shuffle_file_name = current_directory + '\\shuffle'+origin_time_string+'\\'+df
            if not os.path.exists(os.path.dirname(shuffle_file_name)):
                os.makedirs(os.path.dirname(shuffle_file_name))
            with open(shuffle_file_name,'a') as shuffle_file:
                for cbd in cached_balanced_domain:
                    shuffle_file.write(cbd)
        if cached_balanced_domain != False:
            cached_domain_s.append(cached_balanced_domain)
            cached_data.append(getDomainData(cached_balanced_domain,get_removing_words()))
            pure_cached_domain_s.append(cached_domain)
        else:
            if getConstant('VN') == False:
                cached_domain_s.append(None)
                cached_data.append(None)
                pure_cached_domain_s.append(cached_domain)
            else:
                continue                
    n_bd = len(cached_domain_s)
    if n_bd < 20:
        print('one domain missing ' + str(domainFiles))
    past_data = []
    total_accuracy = []
    total_result = []
    total_avg_result = []
    for i in range(0,n_bd):
        if getConstant('one domain') == True:
            if domainFiles[i] != 'CableModem.txt':
                print(str(domainFiles[i]) + ' skipped')
                continue        
            else:
                print('domain found '+ str(cached_data[i][0]))
        if cached_domain_s[i]!=None:
            #prepare past data
            for j in range(0,n_bd):
                if j!=i:
                    if cached_data[j]!=None:
                        past_data.append(cached_data[j])
            accuracy = []
            result_s = []
            M_plus = collections.Counter()
            M_minus = collections.Counter()
            (M_plus, M_minus) = getMkb(past_data)
            vocab_S = getVocabSRegularization(M_plus,M_minus)
            if getConstant('consistent distribution')==True:
                segments = divideDomainCrossValidation_KeepRatio(cached_domain_s[i])
            for s in range(0,5):
                sum_counter_testrev = collections.Counter() #for logging only
                '''format new cached domain: divide then format'''
                if getConstant('consistent distribution')==True:
                    clean_testrev_counter = getReviewCounterFromSegment(segments[s],get_removing_words())
                    train_domain = []
                    for t in range(0,5):
                        if t!=s:
                            train_domain  = train_domain+segments[t]
                    
                    print('train domain '+str(len(train_domain)))
                    try:
                        print((' position 0 '+str(train_domain[0])).encode('utf-8'))
                    except:
                        print('EXCEPT '+str(sys.exc_info()))
                else:
                    if natural == False:
                        (train_domain,clean_testrev_counter) = getTestReviewCounter(cached_domain_s[i],get_removing_words(),s,sum_counter_testrev)
                        #(train_domain, test_domain)=divideDomainCrossValidation(cached_domain_s[i],s) #test domain unused
                    else:
                        #SUSPICIOUS
                        (train_domain,clean_testrev_counter) = getFullTestReviewCounter(pure_cached_domain_s[i],get_removing_words(),s,sum_counter_testrev)
                         #(train_domain, test_domain)=divideDomainCrossValidation(cached_domain_s[i],s) #test domain unused
                
                if getConstant('one domain') == True:
                    printArrayLineNo('testdomain'+str(s),origin_time_string,test_domain)
                past_data.append(getDomainData(train_domain,get_removing_words()))
                vocab_T = getVocabTRegularization_NaiveProb(past_data[-1][2],past_data[-1][3],past_data[-1][7],6)
                X = []
                X.append(initializeX(past_data))
                (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J,signObj) = evalLLLValues(1,past_data,X[0])
                J_s = []
                J_s.append((J,signObj))
                len_target_training_part = len(past_data[-1][6])                
                for n in range(0,len_target_training_part):
                    X.append(updateXstochastic(past_data[-1][5],X[-1][0],X[-1][1],past_data[-1][6][n],past_data[-1][4],vocab_T,past_data[-1][2],past_data[-1][3],vocab_S,M_plus,M_minus,X[0][0],X[0][1]))
                    (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J,signObj) = evalLLLValues(1,past_data,X[-1])
                    J_s.append((J,signObj))
                    #ASSUME                   
                    if np.math.isnan(J_s[-1][0])==False and np.math.isnan(J_s[-2][0])==False and np.absolute(np.exp(J_s[-2][0])*J_s[-2][1]-np.exp(J_s[-1][0])*J_s[-1][1])<0.001:
                        if n>0:
                            msg = 'X[0]= plus '+str(X[0][0].most_common(10))+' MINUS '+str(X[0][1].most_common(10)); logging.debug(msg)
                            msg = 'X[-1]= plus '+str(X[-1][0].most_common(10))+' MINUS '+str(X[-1][1].most_common(10)); logging.debug(msg)
                            msg = 'sum X[0] PLUS '+str(sumaCounter(X[0][0]))+' MINUS '+str(sumaCounter(X[0][1])); logging.debug(msg)
                            print(msg)
                            msg = 'sum X[0] PLUS '+str(sumaCounter(X[-1][0]))+' MINUS '+str(sumaCounter(X[-1][1])); logging.debug(msg)
                            print(msg)
                        break
                ratio = past_data[-1][4]
                if natural == False:
                    result = testing(Pword_plus,Pword_minus,clean_testrev_counter,ratio,get_removing_words())
                else:
                    result = testingF1(Pword_plus,Pword_minus,clean_testrev_counter,ratio,get_removing_words())
                accuracy.append(result[0]) 
                #IMPROVE if accuracy less than 0.8334 print P each doc, 
                result_s.append(result)
                if getConstant('one domain') == True:
                    printArrayLineNo('lowacc'+str(s)+'train',origin_time_string,train_domain)
                    printArrayLineNo('lowacc'+str(s)+'countertest',origin_time_string,clean_testrev_counter)
             #OR F1 SCORE
            if natural ==True:
                avg_result = averageListTuple(result_s)
                total_avg_result = total_avg_result + result_s
            total_accuracy= total_accuracy+accuracy #list adding
            if natural == False:
                avg_accuracy = sum(accuracy)/len(accuracy)
                total_result.append((domainFiles[i],avg_accuracy))
            else:
                total_result.append((domainFiles[i],avg_result))          
            if natural == False:
                print(domainFiles[i]+' '+str(avg_accuracy) + ' avg now '+ str(sum(total_accuracy)/len(total_accuracy)))
            else: #IMPROVE print all the test cases
                print(domainFiles[i]+' '+str(avg_result) + ' F1 score now '+ str(averageListTuple(total_avg_result)))
    if len(total_accuracy)>0:
        if natural==False:
            print('total average accuracy '+str(sum(total_accuracy)/len(total_accuracy)))
        else: 
            print('total average F1 score of negative class '+str(sum(total_accuracy)/len(total_accuracy)))
    else:
        print('total accuracy len 0')
    print(total_result)
    if getConstant('tab chart')==True and natural==True:
        with open('tabchart'+origin_time_string+'.txt','a') as tab_file:
            print('domain\tprecision\tf1\trecall',end='\n',file = tab_file)
            for t_r in total_result:
                print(t_r[0],end = '\t',file=tab_file)
                print('\t'.join(map(str,t_r[1])),end = '\n',file=tab_file)
            if getConstant('micro average')==True:
                micro_correct_negative = sum([mic[1][3] for mic in total_result])
                micro_guess_negative = sum([mic[1][4] for mic in total_result])
                micro_neg_no = sum([mic[1][5] for mic in total_result])
                try:
                    micro_precision = micro_correct_negative/micro_guess_negative
                except:
                    print('no negative guess ' +str(sys.exc_info()))
                    if micro_correct_negative ==0:
                        micro_precision = 1
                    else:
                        micro_precision = 0
                try:
                    micro__recall = micro_correct_negative/micro_neg_no
                except:
                    print('no negative in segment ' +str(sys.exc_info()))
                    if micro_correct_negative == 0:
                        micro__recall = 1
                    else:
                        micro__recall = 0
                try:
                    micro_f1 = (2*micro_precision*micro__recall)/(micro_precision+micro__recall)
                except:
                    micro_f1 = 0
                    print('precision 0 recall 0 '+str(sys.exc_info()))
                print('MICRO AVERAGE F1 '+str(micro_f1),end='',file = tab_file)
            else:
                print('MACRO AVERAGE',end='\n',file = tab_file)
                print(averageListTuple(total_result[1]))

def getFullListVocabulary(data):
    v_list = []
    for dt in data:
        vl = dt[7]
        v_list = vl+list(set(vl)-set(v_list))
    return v_list
def evalLLLValues(smooth,data,X):
    sum_Xplus = sumaCounter(X[0])
    sum_Xminus = sumaCounter(X[1])
    #FIXEDASSUME: only use the vocabulary list from target domain to calculate the P for each word
    v_list = getFullListVocabulary(data)
    (Pword_plus,Pword_minus,nanwordlist) = calcProbabilityVocab(smooth, v_list, X[0], X[1], sum_Xplus, sum_Xminus)
    (J,signObj)=evalObjectiveFunction(data[-1][6],Pword_plus,Pword_minus,data[-1][4],get_removing_words())
    return (sum_Xplus,sum_Xminus,Pword_plus,Pword_minus,J,signObj)  
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
            print('EXCEPT '+str(sys.exc_info()))
            Pp[word] = 1/(total_plus+V)
            Pn[word] = 1/(total_minus+V)
    return (Pp,Pn)
def getVocabSRegularization(Mkb_p,Mkb_n):
    import collections
    tau = 6
    vocab_S = []
    if (type(Mkb_p)!=type(collections.Counter()) or type(Mkb_n)!=type(collections.Counter())):
        print("Mkb wrong type")
    vocab_list = (Mkb_p + Mkb_n).keys()
    doubtful_words = getConstant('doubtful words')
    doubtful_match = []
    for word in vocab_list:
        if Mkb_p[word]>6 or Mkb_n[word]>6:
            vocab_S.append(word)
            if word in doubtful_words:
                doubtful_match.append(word)
    print("added to Vocab S "+str(len(vocab_S))+' matched '+str(doubtful_match))
    return vocab_S
def getVocabTRegularization_NaiveProb(pos_count,neg_count,vocab_list, delta):
    (Pp, Pn) = calcNaiveBayesProb(pos_count,neg_count,vocab_list)
    vocab_T = []
    doubtful_words = getConstant('doubtful words')
    doubtful_match = []
    for word in vocab_list:
        if (Pp[word]/Pn[word] > delta):
            vocab_T.append(word)
            if word in doubtful_words:
                doubtful_match.append(word)
        elif (Pn[word]/Pp[word] > delta):
            vocab_T.append(word)
            if word in doubtful_words:
                doubtful_match.append(word)
    print("added to vocab_T "+str(len(vocab_T))+' matched '+str(doubtful_match))
    return vocab_T 
def resetLogging():
    import sys
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers)>0:
        logger.handlers[0].stream.close()
        logger.removeHandler(logger.handlers[0])
    filename = 'log'+str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))+'.txt'
    file_handler = logging.FileHandler(filename,encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(funcName)s : %(message)s %(lineno)d, %(filename)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)   
    logging.debug(sys.prefix)
def getConstant(cmd):        
    cmd_r = { 'absolute F': False,
             'shuffle' : False,
             'natural distribution': True,
             'one domain': False,
             'update X hard': True,
             'write shuffle':False,
             'VN':True,
             'segmentation':False,
             'no empty string':True,
             'bigram':True,
             #'fixed-size':True,
             '17':True,
             'neutral':False,
             'consistent distribution':True,
             'lower':True,
             'tab chart':True,
             'micro average':True,
             'doubtful words': []#'get','make','use']
             }
    if cmd == 'print':
        print(cmd_r); logging.debug(cmd_r)
        return
    elif cmd in cmd_r.keys():
        r = cmd_r[cmd]
    else:
        r = None
    return r
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
def backupCode(origin_time_string):
    import inspect
    cls = inspect.getmembers(sys.modules[__name__], inspect.isfunction)
    with open('code'+origin_time_string+'.txt','a') as myfile:
        for c in cls:
            try:
                code = inspect.getsourcelines(c[1])
                #print(code[0])
                for line in code[0]:
                    print(str(line),end='',file = myfile)
            except:
                e = sys.exc_info()[0]
                print("EXCEPT back up fails " + str(c)+" " +str(e))
def resetLoggingConsole(origin_time_string):
    f = open('console'+origin_time_string+'.txt', 'a')
    original = sys.stdout
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr,f)
class PublicValues:
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
def resetDefaultEncoding():
    import sys
    if getConstant('VN')==True:
        reload(sys)
        sys.setdefaultencoding('UTF8')
if __name__ == '__main__':    
    import sys
    #Preparation
    resetLogging() 
    resetDefaultEncoding()   
    origin_time = datetime.datetime.now().time()
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    PublicValues.origin_time_string = origin_time_string
    resetLoggingConsole(origin_time_string)
    backupCode(origin_time_string)
    getConstant('print')
    print(origin_time)
    #Execution
    demoBalanced19(origin_time_string)
    #End    
    print(str(datetime.datetime.now().time()) + " origin "+ str(origin_time_string))
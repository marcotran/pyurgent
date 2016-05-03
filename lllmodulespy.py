from __future__ import division
import lifelonglearning34 as l3
import collections, re, os, logging,codecs, itertools, copy, shutil, datetime
import Qglobal as qg
import baseline as bs
import numpy as np
def getScriptName():
    logging.debug('hell no')
    return os.path.basename(__file__).split('.')[0]
def getDictCountFromSplit(split_rev):
    dictcount_rev = [(x[0], dict(qg.sortCounter( collections.Counter(x[1])))) for x in split_rev]
    return dictcount_rev

def getBiasSplit(bias_rev):    
    '''meaningless words removed '''
    if l3.getConstant('VN') != True:
        return [(txt[0], l3.splitEnglish(txt[1])) for txt in bias_rev]
    else:
        return [(txt[0], l3.splitVietnamese(txt[1],True)) for txt in bias_rev]
def getDomainData(dcount_rev):
    #return ratio(number of pos,neg), pos/neg sum counter{}, vocab list[]
    full_count = [(x[0],collections.Counter(x[1])) for x in dcount_rev]
    label_c = collections.Counter([x[0] for x in full_count])
    pos_c = [ x[1] for x in full_count if x[0]==True]
    neg_c = [ x[1] for x in full_count if x[0]==False]
    sum_pos_c = sum(pos_c,collections.Counter())
    sum_neg_c = sum(neg_c,collections.Counter())
    vocab_list = (sum_pos_c+sum_neg_c).keys()
    return (label_c[True],label_c[False],dict(qg.sortCounter( sum_pos_c)), dict(qg.sortCounter( sum_neg_c)),list(vocab_list))
def divide5ValidationKeepRatioFromBiasRev(output_file_path,bias_rev,to_print):
    '''applicable to [(label,count)] or any [(label,...)]'''
    polar_list = [x[0] for x in bias_rev]
    polarity = collections.Counter(polar_list) 
    pos = polarity[True];    neg = polarity[False]
    pos_domain = [x for x in bias_rev if x[0] == True]
    neg_domain = [x for x in bias_rev if x[0] == False]
    pos_seg, pos_left = int(pos/5), pos%5
    neg_seg, neg_left = int(neg/5), neg%5
    start_p, start_n = 0,0
    segments = []
    for i in range(0,5):
        temp = []
        for j in range(start_p, (i+1)*pos_seg):
            temp.append(pos_domain[j])
        start_p = (i+1)*pos_seg
        for k in range(start_n, (i+1)*neg_seg):
            temp.append(neg_domain[k])
        start_n = (i+1)*neg_seg
        segments.append(temp)
    for i in range(0,pos_left):
        segments[i].append(pos_domain[start_p+i])
    for i in range(0,neg_left):
        segments[i].append(neg_domain[start_n+i])
    print('len segment '+str([len(x) for x in segments]))
    if to_print == True:
        qg.makeExist(output_file_path)
        with qg.open(output_file_path, 'w') as wrtf:
            print(segments,end='',file=wrtf)
    else:
        return segments
def fold_count(past_folder_count,output_folder):
    '''28MarPM174908/count/'''
    domain_f_names = qg.getFileList(past_folder_count)
    for dfn in domain_f_names:
        with qg.open(past_folder_count+dfn, 'r') as cfile:
            dcount = eval(cfile.read())
            segments = divide5ValidationKeepRatioFromBiasRev('',dcount,False)
            for i in range(0,5):
                (train,test) = qg.getTrainTestTuple(segments,i) #RERUN
                tst_fname = output_folder+'test/'+dfn+'_'+str(i)+'.txt'; qg.makeExist(tst_fname)
                trn_fname = output_folder+'train/'+dfn+'_'+str(i)+'.txt'; qg.makeExist(trn_fname)
                with qg.open(tst_fname, 'w') as tst_file:
                    print(test,end = '', file = tst_file)
                with qg.open(trn_fname, 'w') as trn_file:
                    print(train,end = '',file = trn_file)
def getMkb(past_folder_data, past_list):
    '''past_folder + '/data/'''
    print(past_list)
    past_domain = []
    for x in past_list:
        with qg.open(past_folder_data+x, 'r') as dfile:
            data = eval(dfile.read())
            pos_count = collections.Counter(data[2])
            neg_count = collections.Counter(data[3])
            vocab_list = data[4]
            fdata = qg.TupleFixedEmptyPosition([pos_count,neg_count,vocab_list],[2,3,7])
            past_domain.append(fdata)
    (Mplus,Mminus) = l3.getMkb(past_domain) #ERROR wrong bigram P eval 
    vocab_S = l3.getVocabSRegularization(Mplus, Mminus)
    return (dict(Mplus),dict(Mminus),vocab_S)
def getVocabSnDataTargetDomain(segments):
    '''[(VT,data)] from train part of target domain '''
    if len(segments)!= 5:
        return None
    full_5_data_vt = []
    for i in range(0,5):
        train_split = qg.getTrainTestTuple(segments,i)[0] #list(itertools.chain(*(qg.getTrainTestTuple(segments,i)[0])))
        dc = getDictCountFromSplit(train_split)
        d = getDomainData(dc)
        vt = l3.getVocabTRegularization_NaiveProb(collections.Counter(d[2]),collections.Counter(d[3]),d[4],6)
        full_5_data_vt.append((vt,d))
    return full_5_data_vt
def format(output_folder,past_folder_data,past_folder_vt, domain_names):
    '''format to (domainName,clean_count,pos_count,neg_count, ratio,vocab,clean_biascount,vocab_list, lst_NBsentence_words)
    clean_count same as clean_biascount; cc-> ratio, vl -> vcb; size = 8 (not 9)
    from  (label_c[True],label_c[False],dict(sum_pos_c), dict(sum_neg_c),list(vocab_list))'''
    for df in domain_names:
        past_list = copy.deepcopy(domain_names); past_list.remove(df)
        domain_data = []
        for x in past_list:
            with qg.open(past_folder_data+x, 'r') as dfile:
                data = eval(dfile.read())
                fdata = qg.TupleFixedEmptyPosition([x,data[2],data[3],data[0]/data[1],data[4]],[0,2,3,4,7])
                domain_data.append(fdata)
            del fdata
        with qg.open(past_folder_vt+df,'r') as vfile:
            five_data_vt = eval(vfile.read())
            if len(five_data_vt)!=5:
                return None
            for i in range(0,5):
                data = five_data_vt[i][1] 
                fdata = qg.TupleFixedEmptyPosition([df,data[2],data[3],data[0]/data[1],data[4]],[0,2,3,4,7])
                f_domain_data = domain_data + [fdata]
                with qg.open(output_folder+df+'_'+str(i)+'.txt','w') as wrtf:
                    print(f_domain_data,end='',file=wrtf)
def initX0(output_folder,format_folder):
    f_list = qg.getFileList(format_folder)
    print(f_list)
    for f in f_list:
        with qg.open(format_folder+f, 'r') as d_file:
            full_data = eval(d_file.read())
            print(f+' read')
            full_list_data = []
            for x in full_data:
                y = list(x)
                y[2] = collections.Counter(x[2])
                y[3] = collections.Counter(x[3])
                full_list_data.append(y)
            X=l3.initializeX(full_list_data)
            with qg.open(output_folder+f, 'w') as wrtf:
                print((dict(qg.sortCounter(X[0])),dict(qg.sortCounter(X[1]))),end='',file = wrtf)
def initP0(output_folder, past_folder_X0): #, past_folder_format):
    smooth = 1
    X_files = qg.getFileList(past_folder_X0)
    for xf in X_files:
        print(xf)
        with qg.open(past_folder_X0+xf,'r') as readXf:
            X_tupdict = eval(readXf.read())
            Xp = collections.Counter(X_tupdict[0]); Xn = collections.Counter(X_tupdict[1])
            sum_Xp = l3.sumaCounter(Xp); sum_Xn = l3.sumaCounter(Xn)
            vocab_list_full = list(Xp.keys()) +list(Xn.keys())
            P = l3.calcProbabilityVocab(smooth,vocab_list_full,Xp,Xn,sum_Xp,sum_Xn)            
            with qg.open(output_folder+xf, 'w') as wrtf:
                print((dict(P[0]),dict(P[1])),end='',file =wrtf)
def refineBigramSplit4Doc(domain_set):
    '''type [(label,[list of unigrams and bigrams])] '''
    refined_list = []
    if l3.getConstant('bigram') == True:
        for rev in domain_set:
            unigrams = [x for x in rev[1] if l3.isTxtBigram(x) == False]
            bigrams = [x for x in rev[1] if l3.isTxtBigram(x) == True]
            try:
                nb_grams = [unigrams[0] ]+bigrams
            except:
                nb_grams = bigrams
            refined_list.append((rev[0],nb_grams))
    else:
        refined_list = copy.deepcopy(domain_set)
    return refined_list
def getDoc4Prob(output_folder,past_folder_fold):
    '''now for unigram only - switch to bigram based on past_folder_fold
    print test-train list'''
    fold_files = qg.getFileList(past_folder_fold)
    for ff in fold_files:
        print(ff)
        with qg.open(past_folder_fold+ff, 'r') as readFF:
            segments = eval(readFF.read())
            for i in range(0,5):
                (train,test) = qg.getTrainTestTuple(segments,i)
                r_train = refineBigramSplit4Doc(train)
                r_test = refineBigramSplit4Doc(test)
                tst_fname = output_folder+'test/'+ff+'_'+str(i)+'.txt'; qg.makeExist(tst_fname)
                trn_fname = output_folder+'train/'+ff+'_'+str(i)+'.txt'; qg.makeExist(trn_fname)
                with qg.open(tst_fname, 'w') as tst_file:
                    print(r_test,end = '', file = tst_file)
                with qg.open(trn_fname, 'w') as trn_file:
                    print(r_train,end = '',file = trn_file)
def calcEmptyProbWord(word,X_plus, X_minus,is_positive):
    '''not work for bigram due to sum a counter function and not suitable formula'''
    if is_positive == True:
        sum_N = l3.sumaCounter(X_plus)
    else:
        sum_N = l3.sumaCounter(X_minus)
    vocab_list =list( X_minus.keys()) + list(X_plus.keys())
    smooth = l3.PublicValues.smooth
    if l3.isTxtBigram(word)!=True or l3.getConstant('bigram naive bayes')!=True:
        P = smooth/(smooth*getLenVocabList(vocab_list) + sum_N)
    else:
        if is_positive == True:
            P = smooth/(smooth*getLenVocabList(vocab_list) + X_plus[l3.getFirstWordOfBigram(word)])
        else:
            P = smooth/(smooth*getLenVocabList(vocab_list) + X_minus[l3.getFirstWordOfBigram(word)])
    return np.log(P)
def calcProbDoc(doc, ratio, Pwp, Pwn,X_plus,X_minus):
    '''doc in [list of words], P in type counter value = log
    now only with unigram(bigram not solve missing case)'''
    Pdoc_pos = np.log(ratio); Pdoc_neg = np.log(ratio**-1)
    for word in doc:
        if Pwp[word]!=np.nan and X_plus[word] != 0:
                Pdoc_pos = Pdoc_pos + Pwp[word]
        else:
            msg = 'nan word '+word;print(msg);logging.debug(msg)
            Pwp[word] = calcEmptyProbWord(word,X_plus,X_minus,True)
            Pdoc_pos = Pdoc_pos + Pwp[word]
        if Pwn[word] != np.nan and X_minus[word] != 0:
                Pdoc_neg = Pdoc_neg + Pwn[word]
        else:
            msg = 'nan word '+word;print(msg);logging.debug(msg)
            Pwn[word] = calcEmptyProbWord(word,X_plus,X_minus,False)
            Pdoc_neg = Pdoc_neg + Pwn[word]
    return (Pdoc_pos,Pdoc_neg)
def calcObjFunc(doc_s,P_p, P_n,X_p,X_n,ratio):    
    '''now for unigram only (use l3.calcDocProb)
    P type counter value = log, X type counter'''
    Obj = 0
    wrong_pdoc_biased_count = 0
    for doc in doc_s:
        #NAIVE add list of word in order
        Pdoc = calcProbDoc(doc[1],ratio,P_p,P_n,X_p,X_n) # BACK LATER
        i = doc_s.index(doc) 
        #ASSUME
        if np.math.isnan(Pdoc[0])==True or np.math.isnan(Pdoc[1])==True:
            logging.debug("Pdoc nan")
            Obj = np.nan
            break
            return Obj
        (bigger, smaller) = (Pdoc[0],Pdoc[1])
        wrong_pdoc_biased = False
        signObj = 1
        #IMPROVE wrong probabilty document
        if doc[0]==True and Pdoc[0]>Pdoc[1]:
            (bigger, smaller) = (Pdoc[0],Pdoc[1])
        elif doc[0]==False and Pdoc[0]<Pdoc[1]:
            (bigger, smaller) = (Pdoc[1],Pdoc[0])
        else:
            wrong_pdoc_biased = True
            wrong_pdoc_biased_count = wrong_pdoc_biased_count +1
            if Pdoc[1]>Pdoc[0]:
                (bigger,smaller) = (Pdoc[1],Pdoc[0])
        signtemp = 1
        if wrong_pdoc_biased == False:
            signtemp = -1
        temp = l3.logaritSubtract(bigger,smaller)
        if i == 0:
            (Obj,signObj) = (temp,signtemp)
        else:
            (Obj,signObj) = l3.logaritSum_sign(Obj,signObj,temp,signtemp)
        if np.absolute(signObj) !=1:
            logging.debug("wrong sign sign temp "+str(signtemp)+" sign Obj "+str(signObj))
            signObj = l3.fixSign(signObj)
        if Obj == 0.0:
            logging.debug( "Obj=0 fuck " + str(i) + "")
    #IMPROVE J nan
    if np.math.isnan(Obj)==False:
        msg = "^Obj = " + str(Obj)+" J= "+str(signObj)+"*"+str(np.exp(Obj))
        print(msg); logging.debug(msg)
    else:
        logging.debug("J nan")
    logging.debug("wrong biased times = "+str(wrong_pdoc_biased_count))
    return (Obj,signObj)
def initObjFunc(output_folder,past_folder_doc,past_folder_P,past_folder_X,past_folder_format):
    if past_folder_doc.endswith('train/') ==False and past_folder_doc.endswith('.txt') == False:
        dir_doc_train = past_folder_doc + 'train/'
        doc_files = qg.getFileList(dir_doc_train)
    else:
        dir_doc_train = '/'.join( past_folder_doc.split('/')[:-1])+'/'
        doc_files = qg.getFileList(past_folder_doc)
    print(dir_doc_train)    
    print('file list '+str(doc_files))
    for dcfl in doc_files:
        with qg.open(dir_doc_train + dcfl, 'r') as dc_file:
            doc_s = eval(dc_file.read())
        with qg.open(past_folder_P + dcfl, 'r') as p_file:
            dict_P = eval(p_file.read())
            (Pp,Pn) = (collections.Counter(dict_P[0]),collections.Counter(dict_P[1]))
        with qg.open(past_folder_X + dcfl,  'r') as x_file:
            dict_X = eval(x_file.read())
            (Xp,Xn) = (collections.Counter(dict_X[0]), collections.Counter(dict_X[1]))
        with qg.open(past_folder_format +  dcfl, 'r') as dt_file:
            past_data = eval(dt_file.read())
        ratio = past_data[-1][4]
        J = calcObjFunc(doc_s,Pp,Pn,Xp,Xn,ratio)
        print(dcfl+' '+ str(J))
        obj_filename = output_folder + dcfl
        with qg.open(obj_filename,  'w') as wrtf:
            print(J,end = '',file = wrtf)
        if len(doc_files)==1:
            return J
def stopSGD(last_Obj, J, org_time):
    '''type (float to exp, sign) '''
    print(last_Obj); print(J)
    if np.math.isnan(J[0]) == False and np.math.isnan(last_Obj[0]) == False:
        x = np.absolute( np.exp(J[0])*J[1] - np.exp(last_Obj[0]) * last_Obj[1])
        if x < l3.PublicValues.learning_stop:
            return True
    if qg.getMinutesDifference(datetime.datetime.now(),org_time) > 7 and l3.getConstant('time limit')==True:
        return True
    return False
def getLenVocabList(vocab_list):
    if l3.getConstant('bigram naive bayes')==True:
        return len([x for x in vocab_list if l3.isTxtBigram(x) == False])
    else:
        return len(vocab_list)
def optimize(output_folder, past_folder_doc_train, past_folder_X0,past_folder_format,past_folder_Mkb, past_folder_VT, past_folder_Obj ):
    doc_file_names = qg.getFileList(past_folder_doc_train)
    org_time = datetime.datetime.now()
    for dfn in doc_file_names:
        with qg.open(past_folder_doc_train + dfn, 'r') as read_doc:
            doc_s = eval(read_doc.read())
        with qg.open(past_folder_format + dfn, 'r') as read_data:
            d_data = eval(read_data.read())
            ratio = d_data[-1][4]
            target_p_count =collections.Counter( d_data[-1][2])
            target_n_count = collections.Counter( d_data[-1][3])
        with qg.open(past_folder_X0 + dfn,  'r') as read_X0:
            d_X0 = eval(read_X0.read())
            X0 = (collections.Counter(d_X0[0]),collections.Counter(d_X0[1]))
        with qg.open(past_folder_VT + dfn[:-6],  'r') as read_VT:
            pivot = int(dfn[-5])
            Vocab_T = eval(read_VT.read())[pivot][0]
        with qg.open(past_folder_Mkb + dfn[:-6], 'r') as read_VS:
            d_M = eval(read_VS.read())
            Vocab_S = d_M[2]
            (Mkb_p, Mkb_n) = (collections.Counter(d_M[0]),collections.Counter(d_M[1]))
            R_Mkb =  l3.get_R_Mkb(Mkb_p, Mkb_n,Vocab_S)
        with qg.open(past_folder_Obj + dfn, 'r') as read_Obj:
            try:
                Obj_0 = eval(read_Obj.read())
            except: 
                Obj_0 = (0.0,1)
                print(str(read_Obj.read())+' '+past_folder_Obj+' '+dfn)
        #STOCHASTIC GRADIENT DESCENT
        limit = len(doc_s)
        for i in range(0,limit): #starting updating X values
            doc = doc_s[-1-i]
            print(str(i)+'\t'+str(doc) )
            di = len(doc[1])
            cc_doc = collections.Counter(doc[1])
            if i == 0:
                last_X = X0
                last_Obj = Obj_0
            else:
                with qg.open(output_folder+'X_'+str(i)+'/'+dfn, 'r') as lX_file:
                    d_lX = eval(lX_file.read())
                    last_X = (collections.Counter(d_lX[0]),collections.Counter(d_lX[1]))
                with qg.open(output_folder+'Obj_'+str(i)+'/'+dfn, 'r') as lObj_file:
                    try:
                        last_Obj = eval(lObj_file.read())
                    except:
                        last_Obj = (0.0,1) #ASSUME
                        print(str(lObj_file.read())+' '+output_folder+'Obj_'+str(i)+'/'+dfn)
            vocab_list = list(last_X[0].keys()) + list(last_X[1].keys())
            sum_Xp = l3.sumaCounter(last_X[0])
            sum_Xn = l3.sumaCounter(last_X[1])
            sum_penalty_Vs = l3.get_sum_penalty_Vs(last_X[0],last_X[1],R_Mkb,X0[0],X0[1],Vocab_S)
            next_X = copy.deepcopy(last_X)
            for k in cc_doc.keys():
                dF = l3.getObjectiveFunctionDiff(k,cc_doc,getLenVocabList(vocab_list),l3.PublicValues.smooth,ratio,last_X[0],last_X[1],sum_Xp,sum_Xn,Vocab_T,target_p_count,target_n_count,di,Vocab_S,sum_penalty_Vs,R_Mkb,X0[0],X0[1])
                next_X[0][k] = last_X[0][k] - dF[0]
                next_X[1][k] = last_X[1][k] - dF[1]
            X_folder = output_folder + 'X_'+str(i+1) + '/'; qg.makeExist(X_folder)
            with qg.open(X_folder+dfn, 'w') as wrtf:
                d_nX = (dict(next_X[0]),dict(next_X[1]))
                print(d_nX,end = '',file = wrtf)
            P_folder = output_folder + 'P_'+str(i+1) +'/';qg.makeExist(P_folder)
            initP0(P_folder,X_folder)
            Obj_folder = output_folder + 'Obj_' + str(i+1) +'/';qg.makeExist(Obj_folder)
            J=initObjFunc(Obj_folder,past_folder_doc_train+dfn,P_folder,X_folder,past_folder_format)
            print(dfn+' '+str(J))
            if stopSGD(last_Obj,J,org_time) == True:
                break
def getPClassifier(output_folder,past_folder_optimize):
    P_folder_list = [x for x in qg.getFolderList(past_folder_optimize) if x.startswith('P_')]
    l = len(P_folder_list)
    print('len P '+str(l))
    match_list = []; path_list = []
    for i in range(0,l):
        cur_dir = past_folder_optimize+'P_'+str(l-i)+'/'
        domain_s = qg.getFileList(cur_dir)
        for d in domain_s:
            if d not in match_list:
                match_list.append(d); path_list.append(cur_dir+d)
                shutil.copyfile(cur_dir+d,output_folder+d)
    print(path_list)
def finalizeX(output_folder,past_folder_optimize):
    X_folder_list = [x for x in qg.getFolderList(past_folder_optimize) if x.startswith('X_')]
    l = len(X_folder_list)
    print('len X '+str(l))
    match_list = []; path_list = []
    for i in range(0,l):
        cur_dir = past_folder_optimize+'X_'+str(l-i)+'/'
        domain_s = qg.getFileList(cur_dir)
        for d in domain_s:
            if d not in match_list:
                match_list.append(d); path_list.append(cur_dir+d)
                shutil.copyfile(cur_dir+d,output_folder+d)
    print(path_list)
def guessLabelDoc(document, P_pos, P_neg, X_pos, X_neg, ratio):
    Pdoc = calcProbDoc(document,ratio,P_pos, P_neg,X_pos, X_neg)
    if Pdoc[0]>Pdoc[1]:
        return True
    else:
        return False
def printTabchart(filename, f1_chart):
    '''f1 chart type [(name str, (x1,x2,x3,...)] '''
    with qg.open(filename,  'a') as wrtf:
        for t_r in f1_chart:
            print(str(t_r[0])+'\t'+'\t'.join(map(str,t_r[1])),file = wrtf)
def printTabchart2(filename, compare_chart):
    with qg.open(filename,  'a') as wrtf:
        for t_r in compare_chart:
            print('\t'.join(map(str,t_r)),file = wrtf)
def testing(output_folder,past_folder_classifier,past_folder_doc_test, past_folder_X,past_folder_format):
    test_list = qg.getFileList(past_folder_doc_test)
    result_f_name = '__result.txt'
    full_F1  = []
    for t in test_list:
        with qg.open(past_folder_classifier+t, 'r') as c_file:
            dP = eval(c_file.read())
            P = ( collections.Counter(dP[0]), collections.Counter(dP[1]))
        with qg.open(past_folder_doc_test + t, 'r') as t_file:
            doc_s = eval(t_file.read())
        with qg.open(past_folder_format + t,  'r') as dt_file:
            ratio = eval(dt_file.read())[-1][4]            
        with qg.open(past_folder_X + t,  'r') as x_file:
            d_X = eval(x_file.read())
            X = (collections.Counter(d_X[0]),collections.Counter(d_X[1])) 
        compare_chart = []
        for doc in doc_s:
            guess = guessLabelDoc(doc[1], P[0],P[1],X[0],X[1],ratio)
            truth = doc[0]
            if truth != guess:
                compare_chart.append((truth,guess,doc))
            else:
                compare_chart.append((truth,guess))
        printTabchart2(output_folder+t,compare_chart)
        F1 = l3.calcF1values(compare_chart)
        full_F1.append((t,F1))
    printTabchart(output_folder+'__fileF1.txt',full_F1)
    if len(full_F1) % l3.PublicValues.fold == 0:
        simplified_F1 = []
        domains_amount =int( len(full_F1)/l3.PublicValues.fold)
        for i in range(0,domains_amount):
            domain_name = full_F1[i*l3.PublicValues.fold][0].split('_')[0]
            for j in range(0,l3.PublicValues.fold):
                index = i*l3.PublicValues.fold + j
                seg_name = full_F1[index][0]
                if seg_name.startswith(domain_name) == False:
                    print('incorrect name match '+ seg_name)
            domain_average = l3.averageListTuple([x[1] for x in full_F1[i*5:i*5+5]])
            simplified_F1.append((domain_name,domain_average))
        printTabchart(output_folder+'__domainF1.txt',simplified_F1)
        final_result = bs.calculateMicroAverage(simplified_F1,'',1,4,5,6,10,11,12)
        with qg.open(output_folder+result_f_name, 'w') as wrtf:
            print(final_result,end ='',file = wrtf); print(final_result)
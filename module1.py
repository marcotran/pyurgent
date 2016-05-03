from __future__ import division
import os
import datetime
import sys
def mergeAfterSegmentation():
    import codecs
    from os import listdir
    from os.path import isfile, join
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    original_folder = 'crawling/1a2'
    ws_folder = 'crawling/process15MarPM151720f1a2_15MarPM152024f1a2/out'
    review_files = [f for f in listdir(original_folder) if isfile(original_folder +'/'+ f)]
    print(review_files)
    ws_files = [f for f in listdir(ws_folder) if isfile(ws_folder +'/'+ f)]
    print(ws_files)
    for r in review_files:
        match_count = 0
        for w in ws_files:
            if w.startswith(r):
                match_count = match_count+1
                r_f = codecs.open(original_folder+'/'+r,mode='r',encoding = 'utf-8')
                w_f = codecs.open(ws_folder+'/'+w, mode = 'r', encoding = 'utf-8')
                lines_w_f = w_f.readlines()
                lines_r_f = r_f.readlines()
                ws_reviews = []; temp = ''
                marks = 0
                for l in lines_w_f:
                    if len(l) <=6:
                        try:
                            ind = int(l.strip())
                            marks = marks+1
                            ws_reviews.append(temp)
                            temp = ''
                            if len(ws_reviews) != ind+1:
                                print('wrong length '+w+' '+l)
                        except:
                            print(w+' '+l+' '+str(sys.exc_info()))
                            continue                        
                    else:
                        temp = temp+' '+l.strip()
                ws_reviews.append(temp)
                del(ws_reviews[0])
                print(ws_reviews[-1])
                if marks != len(ws_reviews):
                    print(w+' len '+str(len(lines_w_f))+' marks '+str(marks)+' ws_reviews '+str(len(ws_reviews)))
                if len(ws_reviews) == len(lines_r_f):
                    my_path = original_folder+ 'combined' + origin_time_string +'/'+r
                    os.makedirs(os.path.dirname(my_path), exist_ok=True)
                    with open(my_path,mode = 'a',encoding = 'utf-8') as mywritefile:
                        for i in range(0,len(ws_reviews)):
                            info = lines_r_f[i].split('\t')[:7]
                            info.append(ws_reviews[i])
                            full_review_info = '\t'.join(info)
                            print(full_review_info.strip(),end= '\n',file=mywritefile)
                        #for l in lines_r_f:
                        #    info = l.split('\t')[:7]
                        #    info.append(ws_reviews[lines_r_f.index(l)])
                        #    full_review_info = '\t'.join(info)
                        #    print(full_review_info,end= '\n',file=mywritefile)
                else:
                    print('different length '+r+' '+w+' '+str(len(ws_reviews)))
        if match_count == 0:
            print('no match found '+r)
        elif match_count >1:
            print('so many match '+r)
    return None
def printBiasedRatio(origin_time_string):
    import codecs
    from os import listdir
    from os.path import isfile,join
    import baseline
    directory = os.path.dirname(os.path.realpath('__file__'))
    mypath = directory+"\\crawling\\1a5"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    tiki_data = []
    for f in onlyfiles:
        if f.endswith('.txt'):
            l_rv = []
            with codecs.open(join(mypath,f),mode='r',encoding='utf-8') as myfile:
                lines = myfile.readlines()
                pos, neg, neu = 0,0,0
                for l in lines:
                    ls = l.split('\t')
                    try:
                        star = int(ls[2])
                    except:
                        print(str(f) + ' '+str(lines.index(l))+' ' + l + ' '+str(sys.exc_info()))
                        continue
                    if len(ls) == 8:
                        rv = ls[7]
                    elif len(ls)>8:
                        rv  = ' '.join(ls[7:])
                    else:
                        print(l+' wrong left split'); rv = ''
                    if rv!= '':
                        l_rv.append(len(baseline.splitSentenceToListWords(rv)))
                    if star == 80 or star == 100:
                        pos = pos +1
                    elif star == 60:
                        neu = neu+1
                    elif star == 40 or star == 20:
                        neg = neg +1
                    elif star <20:
                        print("<20 star, no ideas")
                    elif star > 100:
                        print(">100 star, oh shit")
                    else:
                        print("star wrong "+str(ls))
                if pos+neu+neg !=len(lines):
                    print("wrong counting")
                print(str(f)+ " readlines "+str(len(lines))+" neg rate "+str(neg/len(lines))+" pos "+str(pos)+" neg "+str(neg)+ " neu "+str(neu))
                tiki_data.append((f,pos,neu,neg,sum(l_rv)/len(l_rv),len(l_rv)))
    sorted_tiki_data = sorted(tiki_data,key= lambda tup: tup[3], reverse = True)
    sorted_tiki_data.append(('SUMMARY',sum([x[1] for x in tiki_data]),sum([x[2] for x in tiki_data]),sum([x[3] for x in tiki_data]),'NA',sum([x[5] for x in tiki_data])))
    with codecs.open('tiki_data_summary'+origin_time_string+'1a5.txt',mode='a') as summary_file:
        for s in sorted_tiki_data:
            print('\t'.join([str(x) for x in list(s)]),end = '\n',file=summary_file)
def trimTikiReviewFiles():
    import codecs
    from os import listdir
    from os.path import isfile,join
    import re
    #from nltk import tokenize
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    directory = os.path.dirname(os.path.realpath('__file__'))
    mypath = directory+"\\crawling\\1a2"
    new_path = directory+"\\crawling\\"+origin_time_string+'f1a2'
    
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    for f in onlyfiles:
        if f.endswith('.txt'): #and f.startswith('ButViet'):
            with codecs.open(join(mypath,f),mode='r',encoding='utf-8') as myfile:
                lines = myfile.readlines()
                os.makedirs(os.path.dirname(join(new_path,f)), exist_ok=True)
                with codecs.open(join(new_path,f),mode='a',encoding='utf-8-sig') as mywritefile:
                    for l in lines:
                        ls = l.split('\t') 
                        to_print = ls[-1]
                        if len(ls)!= 8:
                            print('wrong len review'+str(ls))
                            if len(ls)>8:
                                print('len review >8'+str(f))
                                to_print = ' '.join(ls[7:])                                
                        #tkn =tokenize.sent_tokenize(to_print)
                        tkn = get_first_n_sentence(to_print)
                        print(lines.index(l),end='\n',file=mywritefile)
                        for t in tkn: 
                            if t.endswith('.') == False or t.endswith('?') or t.endswith('!'):
                                t = t+' . '     
                            t= t.replace(',',' ,').replace('_',' ').replace('(','').replace(')','').replace('^','').replace(':','')
                            tr = re.findall('[a-zA-Z]', t)                   
                            if tr!=None:
                                if  len(tr) > 1:
                                    t = re.sub(',',' ,',t)
                                    print(t.strip(),end='\n',file=mywritefile)
    print("trimmed")
def post_process():
    import codecs
    from os import listdir
    from os.path import isfile,join
    import re
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    directory = os.path.dirname(os.path.realpath('__file__'))
    t = '15MarPM151720f1a2'
    mypath = directory+"\\crawling\\"+t
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    new_path = directory+"\\crawling\\process"+t+'_'+origin_time_string+'f1a2'
    for f in onlyfiles:
        with codecs.open(join(mypath,f),mode='r',encoding='utf-8') as myfile:
            lines = myfile.readlines()
            os.makedirs(os.path.dirname(join(new_path,f)), exist_ok=True)
            with codecs.open(join(new_path,f),mode='a',encoding='utf-8-sig') as mywritefile:
                for l in lines:
                    l=l.replace('_','')
                    print(l.strip(),file = mywritefile)
    print('processed')
def get_first_n_sentence(text):
    import itertools
    endsentence = ".?!"
    first_n_sentences =[]
    sentences = itertools.groupby(text, lambda x: any(x.endswith(punct) for punct in endsentence))
    for number,(truth, sentence) in enumerate(sentences):
        if truth:
            continue
            #first_n_sentences.append(''.join(sentence).replace('\n',' '))
            #print('enumerate truth I have no idea '+''.join(sentence))
        else:
            first_n_sentences.append(''.join(sentence).replace('\n',' '))
            #print("enumerate not truth "+''.join(sentence).replace('\n',' '))
    #print(first_n_sentences[:10])
    return first_n_sentences
def demo1():
    print('Chevrolet Spark Sexy White')
    directory = os.path.dirname(os.path.realpath('__file__'))
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    full_file_name = directory + '\\domain\\'+origin_time_string + '\\file.txt'
    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)
    dynamic_file_name = '/domain/'+origin_time_string + '/f.txt'
    os.makedirs(os.path.dirname(dynamic_file_name), exist_ok=True)
    with open(full_file_name,'a') as myfile:
        print('bring it on', end="\n", file=myfile)
        myfile.write('bullshit')
        myfile.write('bitch')        
    with open(dynamic_file_name,'a') as f2:
        print('bring it on', end="", file=f2)
        print('Ana Sunamoon', end="", file=f2)
    print('executed successfully')
def print_method_module(method):
    def printer(self):
        name = self.__module__
        if name == '__main__':
            filename = sys.modules[self.__module__].__file__
            name = os.path.splitext(os.path.basename(filename))[0]
        print(name)
        return method(self)
    return printer
if __name__ == '__main__':
    origin_time_string = str(datetime.datetime.now().strftime('%b%d%p%H%M%S'))
    print(origin_time_string)
    printBiasedRatio(origin_time_string)
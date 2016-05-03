import os
import sys
import logging
import lifelonglearning34 as l34
import Qglobal as qg
import lllmodulespy as lm
import baseline as bs
import collections
import numpy
import codecs
import copy
def getScriptName():
    logging.debug('hell no')
    return os.path.basename(__file__).split('.')[0]
def findExistFolder(folder1,folder2,list_of_subdir):
    d_folder = {}
    for sub in list_of_subdir:
        if os.path.exists(os.path.dirname(folder1+'/'+sub+'/')):
            d_folder[sub] = folder1+'/'+sub+'/'
        else:
            d_folder[sub] = folder2+'/'+sub+'/'
    return d_folder
def getLatestFolderNameFromPath(list_of_path):
    return [x.split('/')[-1] for x in list_of_path]
def democsonet2016(new_folder,step_list,step, input_folder_code):
    logging.debug('input = '+str((step,input_folder_code)))
    natural = l34.getConstant('natural distribution')
    domain_file_names = l34.getDomainFileNames()
    org_step = step; step_code = step_list[step]
    if step!=0 and step<len(step_list) and input_folder_code!= '' and os.path.exists(input_folder_code) :
        past_folder = input_folder_code
    else:
        past_folder = new_folder; step = 0
    if step_code == 'cached': #28MarPM150712
        for df in domain_file_names: #ERROR wrong in Vietnamese case
            cached_domain = l34.getDomainfromFile(df)
            cached_balanced_domain = l34.formatDomainBalancing(cached_domain) #ERROR no natural 
            cached_file = new_folder+'/'+step_code+'/'+df.split('/')[-1]
            qg.makeExist(cached_file)
            with qg.open(cached_file,'w') as wtf:
                print(cached_balanced_domain,end='',file=wtf)
            del cached_domain, cached_balanced_domain, cached_file
    domain_file_names = getLatestFolderNameFromPath(domain_file_names)
    for i in range(step,len(step_list)-1):
        step_code = step_list[i]
        if step_code != step_list[org_step]:
            past_folder = new_folder
        d_folder = findExistFolder(new_folder,input_folder_code,step_list)
        if step_code == 'divide': #28MarPM164324
            for df in domain_file_names:
                with qg.open(past_folder+'/cached/'+df,'r') as wtf:
                    divide = l34.divideDomain(eval(wtf.read()))
                divide_file = new_folder+'/divide/'+df; qg.makeExist(divide_file)
                with qg.open(divide_file,  'w') as wrtf:
                    print(divide,end = '',file=wrtf)
                del divide, divide_file

        if step_code == 'split':
            for df in domain_file_names:
                with qg.open(past_folder+'/divide/'+df,'r') as wtf:
                    split_rev = lm.getBiasSplit(eval(wtf.read())[2]) # remove nonsense unigrams
                    print(df)
                split_file = new_folder+'/'+step_code+'/'+df; qg.makeExist(split_file)
                with qg.open(split_file,  'w') as wrtf:
                    print(split_rev,end = '',file = wrtf)
                del split_rev, split_file
        if step_code == 'count':
            for df in domain_file_names:
                with qg.open(past_folder+'/split/'+df,  'r') as wtf:
                    dcount = lm.getDictCountFromSplit(eval(wtf.read()))
                dcount_file = new_folder+'/'+step_code+'/'+df; qg.makeExist(dcount_file)
                with qg.open(dcount_file,  'w') as wrtf:
                    print(dcount,end='',file = wrtf)
                del dcount,dcount_file
        if step_code == 'data':
            for df in domain_file_names:
                with qg.open(past_folder+'/count/'+df,'r') as wtf:
                    data = lm.getDomainData(eval(wtf.read()))
                data_file = new_folder + '/'+step_code+'/'+df; qg.makeExist(data_file)
                with qg.open(data_file,  'w') as wrtf:
                    print(data,end = '',file = wrtf)
                del data, data_file
        if step_code == 'fold': #29MarPM151930
            for df in domain_file_names:
                with qg.open(past_folder+'/split/'+df,'r') as wtf:
                    segments_file_path = new_folder + '/'+step_code+'/'+df
                    lm.divide5ValidationKeepRatioFromBiasRev(segments_file_path, eval(wtf.read()),True)
                del segments_file_path
        if step_code == 'MKB': #29MarPM123359
            for df in domain_file_names:
                past_list =copy.deepcopy( domain_file_names); past_list.remove(df)
                M_2tup = lm.getMkb(past_folder+'/data/',past_list)
                Mkb_file = new_folder + '/'+step_code+'/'+df; qg.makeExist(Mkb_file)
                with qg.open(Mkb_file, 'w') as wrtf:
                    print(M_2tup,end = '',file = wrtf)
                del past_list
        if step_code == 'VT': #29MarPM205805VTformatX_0fold_countcondisnoempstrlowtabchamicaveupdXhar
            for df in domain_file_names:
                
                with qg.open(past_folder+'/fold/'+df,'r') as wtf:
                    five_vt_data = lm.getVocabSnDataTargetDomain(eval(wtf.read()))
                target_file = new_folder + '/'+step_code+'/'+df; qg.makeExist(target_file)
                with qg.open(target_file, 'w') as wrtf:
                    print(five_vt_data,end = '',file = wrtf)
                del five_vt_data, target_file
        if step_code == 'format':
            #d_folder = findExistFolder(input_folder_code,new_folder,['data','VT'])
            format_folder = new_folder + '/'+step_code+'/'; qg.makeExist(format_folder)
            #lm.format(format_folder,input_folder_code+'/data/',input_folder_code+'/VT/',domain_file_names)
            lm.format(format_folder,d_folder['data'],d_folder['VT'],domain_file_names)
        if step_code == 'X_0': #29MarPM161140
            #d_folder = findExistFolder(input_folder_code,new_folder,['format'])
            X_folder = new_folder + '/'+step_code+'/'; qg.makeExist(X_folder)
            #lm.initX0(X_folder,past_folder+'/format/')
            lm.initX0(X_folder,d_folder['format'])
        if step_code == 'fold_count':
            #d_folder = findExistFolder(input_folder_code,new_folder,['count'])
            fc_folder = new_folder + '/'+step_code+'/'; qg.makeExist(fc_folder)
            lm.fold_count(d_folder['count'],fc_folder)
        if step_code == 'P_0':
            #d_folder = findExistFolder(input_folder_code,new_folder,['X_0'])
            p_folder = new_folder+ '/'+step_code+'/'; qg.makeExist(p_folder)
            lm.initP0(p_folder,d_folder['X_0'])
        if step_code == 'doc':
            #d_folder = findExistFolder(input_folder_code,new_folder,['fold'])
            doc_folder = new_folder+'/'+step_code+'/';qg.makeExist(doc_folder)
            lm.getDoc4Prob(doc_folder,d_folder['fold'])
        if step_code == 'Obj_0':
            #d_folder = findExistFolder(input_folder_code,new_folder,['doc','P_0','X_0','format'])
            Obj_folder = new_folder + '/'+step_code+'/'; qg.makeExist(Obj_folder)
            lm.initObjFunc(Obj_folder,d_folder['doc'],d_folder['P_0'],d_folder['X_0'],d_folder['format'])
        if step_code == 'optimize':
            Op_folder = new_folder + '/' + step_code + '/'; qg.makeExist(Op_folder)
            lm.optimize(Op_folder,d_folder['doc']+'train/',d_folder['X_0'],d_folder['format'],d_folder['MKB'],d_folder['VT'],d_folder['Obj_0'])
        if step_code == 'classifier':
            clsf_folder = new_folder +'/'+step_code+'/'; qg.makeExist(clsf_folder)
            lm.getPClassifier(clsf_folder,d_folder['optimize'])
        if step_code == 'finalX':
            Xfinal_folder = new_folder +'/'+step_code+'/'; qg.makeExist(Xfinal_folder)
            lm.finalizeX(Xfinal_folder,d_folder['optimize'])
        if step_code == 'test':
            tst_folder = new_folder+ '/'+step_code+'/'; qg.makeExist(tst_folder)
            lm.testing(tst_folder,d_folder['classifier'],d_folder['doc']+'test/',d_folder['finalX'],d_folder['format'])
    return None
def printCmdR(new_folder, input_folder_code, folder_name_ext):
    print('please type the change log')
    change_log = input()
    fn = new_folder+'/constant.txt'; qg.makeExist(fn)
    with qg.open(fn, 'w') as f:
        print(l34.getConstant('get'),end='',file = f)
    with qg.open(new_folder+'/changelog.txt','w') as fcl:
        print(change_log,end = '',file =fcl)
    with qg.open(new_folder +'/input.txt','w') as f_in:
        print(input_folder_code,end = '',file =f_in)
    with qg.open('changlog'+folder_name_ext+'.txt', 'a') as f_info:
        print(change_log,file = f_info)
        print('VN: '+str(l34.getConstant('VN')),file = f_info)
        print(input_folder_code, file = f_info)
if __name__ == '__main__':
    l34.PublicValues.origin_time_string = qg.getTimeString()
    orgtimestr = l34.PublicValues.origin_time_string
    l34.PublicValues.constant_list =  eval(qg.open('__constant.txt','r').read())
    var_input = eval(qg.open('__input.txt','r').read())
    #LOGGING    
    l34.backupCode(orgtimestr)
    l34.resetLogging(orgtimestr)
    l34.resetLoggingConsole(orgtimestr)    
    #PRINTING PREVIEW INFO    
    step_list = ['cached','divide','split','count','data',
                 'fold','MKB','VT','format','X_0',
                 'fold_count','P_0','doc','Obj_0','optimize',
                 'classifier','finalX','test','']
    step = int(var_input[0]); folder_name_ext = '_'+ bs.getEnvironmentName() +'_'+qg.getTrimString(' '.join(step_list[step:]))+'__'+l34.getConstant('str') 
    new_folder = l34.PublicValues.origin_time_string + folder_name_ext
    input_folder_code =var_input[1]  #'30MarPM222548_cacdivsplcoudatfolMKBVTforX0folP0docObjoptclafintes__CondisPunUpdxharNoempstrLow'#'31MarAM014811_cacdivsplcoudatfolMKBVTforX0folP0docObjoptclafintes__LowCondisPunVnTimlimUpdxharNoempstr'#'30MarPM222548_cacdivsplcoudatfolMKBVTforX0folP0docObjoptclafintes__CondisPunUpdxharNoempstrLow'#'30MarPM231621_cacdivsplcoudatfolMKBVTforX0folP0docObjoptclafintes__LowVnNoempstrCondisUpdxharPun'# '30MarPM141536_forX0folP0docObjoptclafintes__LowCondisBigBignaibayTabchaMicaveNoempstrUpdxhar' #'30MarPM134730_splcoudatfolMKBVTforX0folP0docObjoptclafintes__NoempstrCondisBigBignaibayTabchaMicaveLowUpdxhar')#'29MarAM083655')#'30MarAM102500_cla__CondisTabchaMicaveUpdxharNoempstrLow')#'30MarAM081447_opt__NoempstrTabchaMicaveCondisUpdxharLow')#'30MarAM012501_Obj__LowMicaveNoempstrTabchaCondisUpdxhar')#'30MarAM004307_folP0doc__TabchaLowMicaveCondisUpdxharNoempstr')#'29MarPM151930')#'29MarPM161140')#'29MarPM151930')#'29MarPM123359')#'29MarAM110441')
    if step == 0:
        input_folder_code = ''
    printCmdR(new_folder, input_folder_code, folder_name_ext)
    #EXECUTING
    democsonet2016(new_folder,step_list,step,input_folder_code)
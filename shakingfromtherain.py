from __future__ import division
from bs4 import BeautifulSoup
from urllib.request import urlopen
def getReviewFeature(review,feature):
    err, err_msg = False, ''
    if feature == 'author':
        m = review.find_all('p',{'class':'name','itemprop':'author'})
        if len(m) ==1:
            try:
                return m[0].string.title()
            except:
                print("m[0] "+str(m[0]))
                return m[0].string
        else:
            err, err_msg = True, 'authors missing or ambigous '+str(len(m))
    elif feature == 'star':
        s = review.find_all(lambda tag: tag.name=='span' and len(tag.attrs)==1 and 'style'in tag.attrs)
        if len(s)== 1:
            return s[0].attrs['style'].split(':')[1].split('%')[0]
        else:
            return None
    elif feature == 'headline':
        h = review.find_all(lambda tag: tag.name=='a' and len(tag.attrs)==1 and 'href'in tag.attrs and tag.attrs['href']!='')
        if len(h) ==1:
            return h[0].string
        else:
            err, err_msg = True, 'headlines ambigous '+str(len(m))
    elif feature == 'thanks':
        y = review.find_all('span',{'class':'text-success'})
        if len(y) == 1:
            b= y[0].find_all('b')        
            return int(b[0].string.split(' ')[0])
        else:
            return 0
    elif feature == 'bought':
        b = review.find_all('p',{'class':'buy-already'})
        if len(b) == 0:
            return False
        elif len(b) == 1: 
            return True
    elif feature == 'detail':
        d = review.find_all('span',{'class':'review_detail','itemprop':'reviewBody'})
        if len(d)==1:
            return d[0].string
        else:
            return False
    elif feature == 'days':
        dd = review.find_all('p',{'class':'days'})
        if len(dd) == 1:
            return dd[0].string.strip('(').strip(')')
        else:
            return False
    else: 
        err, err_msg = True, 'feature not found'
def getRawReviewsFromOnePage(page):
    soup = BeautifulSoup(page.read(),'html.parser')
    items = soup.find_all('div',{'class':'item','itemprop':'review'})
    return items
def format_url(urllink,code):
    new_link = ''
    if code=='soup_tiki_product':
        if urllink.endswith('.html') == False:
            new_link = urllink + '.html'
            return new_link
    if code == 'soup_tiki_show_reviews':
        if 'ref' in urllink or '.html?' in urllink:
            new_link = urllink.split('.html')[0]+'/nhan-xet#reviewShowArea'+'.html'
            return new_link
    else: 
        return None
def doSth():
    a = 1
def getMultipleReviewPages(urllink):
    new_link = ''
    if urllink.endswith('/nhan-xet#reviewShowArea') == False and urllink.endswith('/nhan-xet#reviewShowArea.html') == False:
        if 'ref' in urllink or '.html?' in urllink:
            doSth() #print('correct link type')     
        new_link = urllink.split('.html')[0] #+'/nhan-xet#reviewShowArea'+'.html'
    else:
        return False
    link_collection = []
    page = urlopen(format_url(urllink,'soup_tiki_product'))
    soup = BeautifulSoup(page.read(),'html.parser')
    comments_count = getPageFeature(soup,'comments_count')
    pages = int(comments_count / 10)
    if int(comments_count) % 10 != 0:
        pages = pages +1
    #print('pages = '+str(pages))
    for i in range(1,pages+1):
        newer_link = new_link + '/nhan-xet?p='+str(i)+'#reviewShowArea'
        link_collection.append(newer_link)
    return link_collection
def getPageFeature(soup,feature):
    err, err_msg = False, ''
    if feature == 'product_name':
        name = soup.find_all('h1',{'class':'item-name','itemprop':'name'})
        if len(name)==1:
            return name[0].string.title().strip()
    if feature == 'comments_count':
        c = soup.find_all('p',{'class':'comments-count'})
        #if len(c)>1:
        try:
            cc = c[0].find_all('a')
            if len(cc) == 1:
                return int(cc[0].string.strip('(').strip(')').split()[0])
        except:
            return 1000
    else:
        return None
    #category: div class = col-md-12
def getConstant(name):
    if name == 'product_feature_list':
        return [ 'product_name','comments_count']
    elif name == 'review_feature_list':
        return ['author','star','headline','thanks','bought','days','detail']
def getAllProductReviewsFromURL(urllink):
    link_collection =  getMultipleReviewPages(urllink)
    
    product_feature_list = getConstant('product_feature_list')
    review_feature_list = getConstant('review_feature_list')
    product_info, product_review_info, page_org = [] ,[], urlopen(urllink)
    soup_org = BeautifulSoup(page_org,'html.parser')
    negative_count = 0
    for f in product_feature_list:
        product_info.append(getPageFeature(soup_org,f))
    for link in link_collection:
        try:
            page_r = urlopen(link)
        except:
            print(link + ' 404 not found')
            continue
        reviews = getRawReviewsFromOnePage(page_r)
        for r in reviews:
            single_review_info = []
            if getReviewFeature(r,'detail')==False or getReviewFeature(r,'detail') == None:
                continue
            for f in review_feature_list:
                single_review_info.append(getReviewFeature(r,f))
                if f == 'star':
                    bias = int(getReviewFeature(r,f))
                    if bias<60:
                        negative_count = negative_count+1
            product_review_info.append(single_review_info)
    product_info.append(negative_count)
    return (product_info,product_review_info)
def normalizeVietnamese(phrase, without_space):
    from unidecode import unidecode
    m = unidecode(phrase)
    if without_space == False:
        return m
    real_phrase = ''
    for ch in m:
        if ch.isalnum():
            real_phrase = real_phrase +ch
    return real_phrase
def WriteToNewFileOneProduct(product_name_V,product_info,product_review_info,origin_time_string,category,category_mode):    
    import os
    from unidecode import unidecode
    prefix = '\\crawling\\'
    file_name = normalizeVietnamese(product_name_V,True)
    directory = os.path.dirname(os.path.realpath('__file__'))
    if category_mode == True:    
        full_file_name = directory + prefix + origin_time_string + '\\'+category +'.txt'
    else:
        full_file_name = directory + prefix + origin_time_string + '\\'+category+'\\'+file_name+'.txt'
    #IF CATEGORY MODE: file name = category name
    os.makedirs(os.path.dirname(full_file_name), exist_ok=True)
    splitter = '\t'
    count = 0
    import codecs
    with codecs.open(full_file_name,mode="a",encoding='utf-8') as myfile:
        if category_mode != True:
            for i in product_info:
                if product_info.index(i) == len(product_info)-1:
                    print(str(i), end='\n', file=myfile)
                else:
                    print(str(i), end=splitter, file=myfile)
            const_rf = getConstant('review_feature_list')
            for x_rf in const_rf:
                if const_rf.index(x_rf) == len(const_rf)-1:
                    print(str(x_rf), end='\n', file=myfile)
                else:
                    print(str(x_rf), end=splitter, file=myfile)
        #IF CATEGORY MODE: do not print the above
        for ri in product_review_info:
            count = count+1
            #IF CATEGORY MODE: print product name after/in lieu of 
            if category_mode != True:
                myfile.write(str(product_review_info.index(ri))+splitter)
            else:
                myfile.write(file_name+splitter)
            for ri_f in ri:
                if ri.index(ri_f)==len(ri)-1:
                    print(str(ri_f), end='\n', file=myfile)
                else:
                    print(str(ri_f), end=splitter, file=myfile)
    
def CrawlToFileOneProduct(urllink,origin_time_string,category,category_mode):
    (product_info,product_review_info) = getAllProductReviewsFromURL(urllink)
    product_name_V = product_info[0]
    WriteToNewFileOneProduct(product_name_V,product_info,product_review_info,origin_time_string,category,category_mode)
    return (product_info[1],product_info[2])
def resetOutputEncoding():
    import sys
    import codecs
    if sys.stdout.encoding != 'cp850':
        sys.stdout = codecs.getwriter('cp850')(sys.stdout, 'strict')
    if sys.stderr.encoding != 'cp850':
        sys.stderr = codecs.getwriter('cp850')(sys.stderr, 'strict')
def getProductLinksFromMenu(urllink):
    menu_page = urlopen(urllink)
    soup = BeautifulSoup(menu_page.read(),'html.parser')
    category = soup.find_all('div',{'class':'filter-list-box'})
    if len(category) ==1:
        product_type = category[0].find_all('h1')[0].string.strip()
        no_products = int(category[0].find_all('h4')[0].string.strip('(').strip(')'))
    else:
        product_type = 'unknown'; no_products=0
    pages = int(no_products/40)+1
    l = urllink[:-1]
    product_links = []
    for i in range(1,pages+1):
        new_link = l + str(i)
        soup_pl = BeautifulSoup(urlopen(new_link).read(),'html.parser')
        pl = [x for x in soup_pl.find_all(lambda tag: tag.name=='a' and len(tag.attrs)==3 and 'href'in tag.attrs and 'title' in tag.attrs and 'data-id' in tag.attrs) if len(x.find_all('p',{'class':'review'}))==1]
        links = [x.attrs['href'] for x in pl]
        product_links = product_links + links
    return (product_type,no_products,product_links)
def crawl_links():
    import sys
    page = urlopen('http://tiki.vn/')
    soup = BeautifulSoup(page.read(), 'html.parser')
    nav_sub = soup.find_all('div',{'class':'nav-sub-list-box'})
    href_list = []
    for n_s in nav_sub:
        a_link_s = n_s.find_all(lambda tag: tag.name == 'a' and 'href' in tag.attrs)
        for a_link in a_link_s:
            l= a_link.attrs['href']
            if 'bestsellers' not in l and 'new-products' not in l and 'san-pham-giam-gia' not in l and 'tuyen-tap' not in l and 'khuyen-mai' not in l and 'san-pham-noi-bat'not in l and 'sap-phat-hanh' not in l and 'sach-truyen-tieng-viet' not in l and 'sach-tieng-anh' not in l and 'cty-sach' not in l and 'discounted-books' not in l and 'thuong-hieu' not in l:
                if l.count('http') > 1:
                    continue
                if len(l.split('/'))<=4:
                    continue
                if l.endswith('html') == False and l.split('/')[-1].startswith('c')==False:
                    print('wrong '+l)
                    continue
                try:
                    p_l = urlopen(l+'?limit=40&page=1')
                    soup_pl = BeautifulSoup(p_l.read(),'html.parser') 
                    product_link = [x for x in soup_pl.find_all(lambda tag: tag.name=='a' and len(tag.attrs)==3 and 'href'in tag.attrs and 'title' in tag.attrs and 'data-id' in tag.attrs) if len(x.find_all('p',{'class':'review'}))==1] 
                    if len(product_link)>0:
                        print(l,end='\n')
                        href_list.append(l+'?limit=40&page=1')
                except:
                    print('cannot open ' +l+ ' ' +str(sys.exc_info()))
                    continue
            else:
                continue
    
    return href_list
def getProductReviewInfoFromURL(urllink):
    #urllink must have reviews
    try:
        page_r = urlopen(urllink)
    except:
        print(urllink+" 404 not found")
        return (0,0,0)
    soup = BeautifulSoup(page_r,'html.parser')
    chart = soup.find_all('div',{'class':'product-customer-col-2'})
    pos,neg,neu = 0,0,0
    if len(chart)==1:
        rating = chart[0].find_all('div',{'class':'item'})
        if len(rating) ==5:
            for r in rating:
                score = int(r.find_all('span',{'class':'rating-num'})[0].string.split(' ')[0])
                total_rate = int(r.find_all('span',{'class':'rating-num-total'})[0].string.split(' ')[0])
                if score ==5 or score == 4:
                    pos = pos+total_rate
                elif score == 1 or score ==2:
                    neg = neg+total_rate
                elif score == 3:
                    neu = neu + total_rate
                else:
                    print(urllink+" no score detected")
            return (pos+neg+neu,pos,neg)
        else:
            print(urllink+" rating len not 5")
    else:
        print(urllink+" product customer col 2 not 1")
    return (0,0,0)
def getProductTypeSummaryFromMenu(urllink):
    #domain_category = AlnumPhrase(urllink.split('/')[3])
    import sys
    try:
        menu_page = urlopen(urllink)
    except:
        print(urllink +  ' ' + str(sys.exc_info()))
        return None
    soup = BeautifulSoup(menu_page.read(),'html.parser')
    category = soup.find_all('div',{'class':'filter-list-box'})
    if len(category) ==1:
        #product_type = category[0].find_all('h1')[0].string.strip()
        no_products = int(category[0].find_all('h4')[0].string.strip('(').strip(')'))
    else:
        print("len category "+str(len(category))+" "+urllink)
        #product_type = 'unknown'; 
        no_products=0
    pages = int(no_products/40)+1
    l = urllink+'?limit=40&page='
    product_links = []
    no_of_reviews = 0
    total_reviews, total_negative_reviews = 0,0
    for i in range(1,pages+1):
        new_link = l + str(i)
        try:
            soup_pl = BeautifulSoup(urlopen(new_link).read(),'html.parser')
        except:
            print(l+str(sys.exc_info))
            continue
        pl = [x for x in soup_pl.find_all(lambda tag: tag.name=='a' and len(tag.attrs)==3 and 'href'in tag.attrs and 'title' in tag.attrs and 'data-id' in tag.attrs) if len(x.find_all('p',{'class':'review'}))==1]
        links = [x.attrs['href'] for x in pl]
        product_links = product_links + links
    for pl in product_links:
        r_info = getProductReviewInfoFromURL(pl)
        try:
            total_reviews = total_reviews+ r_info[0]
            total_negative_reviews = total_negative_reviews +r_info[2]
        except:
            print(sys.exc_info())
            continue
    result = (urllink,no_products,total_reviews,total_negative_reviews)
    print(result)
    return result
def read_links():
    with open('tiki.txt','r') as myfile:
        links = myfile.readlines()
        fit_links = [l.split('\n')[0] for l in links]
        print(fit_links)
    return fit_links
def sister_crawl():
    import datetime
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    href_list = read_links()
    total_info = []
    for h in href_list:
        info  = getProductTypeSummaryFromMenu(h)
        if info==None:
            continue
        total_info.append(info)
    sorted_info = sorted(total_info,key = lambda tup: tup[-1],reverse=True)
    print(sorted_info[:40])
    return [x[0]+'?limit=40&page=1' for x in sorted_info[:40]]
def getURLlinkcollection(code):    
    collection_1 = [
        'http://tiki.vn/xit-khoang-evoluderm-400ml-7084-p96006.html?ref=c1520.c1582.c1643.c2347.c2767.c2772.c2813.c2856.c3533.c4166.c4662.c2879.c2889.c3196.'
        ,'http://tiki.vn/gel-duong-the-duong-mat-blumei-jeju-aloe-vera-soothing-gel-97-300ml-p132864.html?ref=c3494.c4865.c4927.'
        ,'http://tiki.vn/combo-2-chai-xit-khoang-evoluderm-150ml-p149356.html?ref=c3494.c4865.c4927.'
        ,'http://tiki.vn/gel-lo-hoi-milaganics-150g-p137947.html?ref=c3494.c4865.c4927.'
        ,'http://tiki.vn/but-ke-mat-mira-net-sieu-manh-khong-troi-e233-p98689.html?ref=c4865.c4927.'
        ,'http://tiki.vn/sua-rua-mat-st-ives-tuoi-mat-da-huong-mo-p86284.html?ref=c3494.c4865.c4927.c4957.'
        ,'http://tiki.vn/gel-khang-khuan-tri-mun-tinh-chat-tram-tra-p103948.html?ref=c1520.c1582.c1599.c3426.c4662.'
        ,'http://tiki.vn/sua-rua-mat-st-ives-kiem-soat-mun-huong-mo-p86295.html?ref=c3494.c4865.c4927.c4957.'
        ,'http://tiki.vn/combo-mascara-dau-dua-duong-mi-milaganics-tinh-chat-duong-moi-lip-gloss-milaganics-p145983.html?ref=c3494.c4865.c4927.'
        ,'http://tiki.vn/bot-yen-mach-milaganics-200g-p145292.html?ref=c1520.c1582.c3423.c4662.c4086.c4096.'
        ,'http://tiki.vn/bong-rua-mat-bot-bien-cao-cap-mira-b577-p123074.html?ref=c1520.c1582.c2990.c1599.'
        ,'http://tiki.vn/bot-cam-gao-milaganics-p137949.html?ref=c1520.c1592.c1619.c3423.c4086.c4096.'
        ,'http://tiki.vn/sua-rua-mat-sang-da-tinh-dau-tra-vedette100ml-p82117.html?ref=c1520.c1582.c1583.c1790.c2294.c2879.'
        ,'http://tiki.vn/mascara-dau-dua-duong-mi-milaganics-p145982.html?ref=c1520.c1584.c1586.c4662.c4790.c1641.c4084.c4096.'
        ,'http://tiki.vn/giay-tham-dau-mayan-than-hoat-tinh-p126150.html?ref=c1520.c1582.c1589.c1601.'
        ,'http://tiki.vn/sua-tay-te-bao-chet-duong-sang-da-p82128.html?ref=c1520.c1582.c1603.c1790.c3426.c2879.'
        ,'http://tiki.vn/bookmark-tiki-dau-an-thoi-gian-p161229.html?src=bookmark&ref=c1857.c2368.c4583.c4695.c4724.c1911.c4723.c4725.'
        ,'http://tiki.vn/bookmark-in-the-zoo-p128569.html?src=bookmark&ref=c1857.c2368.c1911.c2045.c4798.'
        ,'http://tiki.vn/bo-bookmark-fairy-corner-p118744.html?src=bookmark&ref=c1857.c2368.c1911.c2045.c2774.c2808.c2816.c2860.c2886.c2923.'
        ,'http://tiki.vn/bo-lau-nha-da-nang-360-do-eco-mop-p127314.html?ref=c1883.c1951.c3130.c4527.c4575.c4654.c4695.c4862.c5092.c1987.c2064.c3132.c3837.c4534.c4630.c4667.c4682.c4789.c4871.c4926.c5047.c5095.c2234.c3137.c3440.'
        ,'http://tiki.vn/den-pin-doc-sach-energizer-booklite-p95677.html?ref=c1883.c2015.c4527.c4576.c4654.c5002.c5092.c5120.c2018.c4047.c4534.c4660.c4681.c4787.c4870.c5059.c5095.c3846.c4718.'
        ,'http://tiki.vn/binh-giu-nhiet-carlmann-bes523-500ml-p109886.html?ref=c1883.c1975.c3183.c4575.c4654.c4695.c5002.c5009.c5092.c5120.c2065.c2362.c3184.c3647.c4630.c4660.c4666.c4667.c4682.c4789.c4926.c5059.c5095.c2686.c3111.c3112.c3137.c3189.c3366.'
        ,'http://tiki.vn/dien-thoai-nokia-105-p67291.html?ref=c1789.c1793.c1796.c2277.c3843.c5055.c5093.c3498.c4709.c4861.c2926.c4214.c4947.'
        ,'http://tiki.vn/mo-hinh-lego-classic-10692-sang-tao-221-manh-ghep-n-p135753.html?ref=c1929.c3772.c4581.c1953.c2049.c2050.c3587.c3771.c3842.c3888.c4048.c4442.c4537.c4665.c4684.c4792.c4915.c4929.c3580.c3582.c3584.c3585.c3591.c3871.c4044.c4720.c3745.'
        ,'http://tiki.vn/mo-hinh-lego-thung-gach-trung-classic-10696-sang-tao-484-manh-ghep-n-p135752.html?ref=c1929.c3772.c1953.c2049.c2050.c3587.c3771.c3888.c4442.c3580.c3582.c3584.c3585.c3591.c3871.c4044.c3745.'
        ,'http://tiki.vn/but-ve-ky-thuat-marvy-4600-p111579.html?ref=c1857.c2365.c2045.c2454.c2774.c2816.c2860.c3525.c3715.c3845.c4797.c4932.c5125.c1858.c2883.c3015.c3031.c3263.c1879.c2923.'
        ,'http://tiki.vn/but-bi-muc-duc-marvy-sb10-p111618.html?ref=c1857.c2365.c2454.c4932.c1858.c2532.c2883.c1869.'
        ,'http://tiki.vn/combo-tron-bo-blueup-ielts-1100-essential-flashcards-for-ielts-p120620.html?ref=c1857.c2368.c3183.c4583.c1910.c2045.c2852.c2860.c2861.c3399.c3525.c4726.c4797.c4932.c2885.c3016.c3199.c3266.c4734.c2947.'
        ,'http://tiki.vn/but-long-kim-nhieu-mau-marvy-4300-p111599.html?ref=c1857.c2365.c2045.c2454.c4932.c1858.c3263.c1873.'
        ,'http://tiki.vn/bang-trang-tri-deco-rush-p74191.html?ref=c1857.c1862.c1866.c2045.c2047.c2221.c2816.c2860.c3525.c3715.c3845.c4797.c4932.c1926.c2883.c3031.c3443.'
        ,'http://tiki.vn/day-nhay-so-p131280.html?ref=c1975.c3183.c4527.c4584.c2978.c3838.c4539.c3137.c3189.c3195.'
        ]
    collection_2 = [
        
        #'http://tiki.vn/kem-duong-da/c1599?limit=40&page=2'
        #,'http://tiki.vn/mat-na-cac-loai/c1601?limit=40&page=2'
        #,'http://tiki.vn/sua-rua-mat/c1583?limit=40&page=2'
        #,'http://tiki.vn/tinh-chat-san-pham-dac-tri/c3423?limit=40&page=2'
        'http://tiki.vn/sua-tam/c1598?limit=40&page=2'
        ,'http://tiki.vn/vpp-but-viet/c1858?limit=40&page=2'
        ,'http://tiki.vn/tap-hoc-sinh-tiki/c3429?limit=40&page=2'
        ,'http://tiki.vn/flashcard-hoc/c1910?limit=40&page=2'
        ,'http://tiki.vn/dien-thoai-di-dong/c1793?limit=40&page=2'
        ,'http://tiki.vn/tai-nghe-nhac/c1804?limit=40&page=2'
        ,'http://tiki.vn/binh-uong-nuoc-the-thao/c2686?limit=40'
        ,'http://tiki.vn/dung-cu-tap-luyen/c2978?limit=40'
        ,'http://tiki.vn/binh-dun-sieu-toc/c1931?limit=40&page=2'
        ]
    collection_3 = [
        
        #'http://tiki.vn/do-dung-nha-bep/c1884?limit=40&page=2',
        'http://tiki.vn/phong-khach/c1959?limit=40&order=price%2Casc&page=2'
        ,'http://tiki.vn/thoi-trang-nam/c915?limit=40&page=2'
        ,'http://tiki.vn/do-choi-go/c2848?limit=40&page=2'
        ,'http://tiki.vn/do-dung-cho-be/c2550?limit=40&page=2'
        ,'http://tiki.vn/loa-nghe-nhac/c1805?limit=40&page=2'
        ,'http://tiki.vn/dung-cu-hoc-sinh/c2365?limit=40&page=2'
        #,'http://tiki.vn/lam-dep-suc-khoe/c1520?order=newest&limit=40&page=2'
        ]
    collection_4 = [
        'http://tiki.vn/trang-diem/c1584',
        'http://tiki.vn/tai-nghe-nhac/c1804',
        'http://tiki.vn/do-dung-nha-bep/c1884',
        'http://tiki.vn/nuoc-hoa/c1595',
        'http://tiki.vn/do-dung-cho-be/c2550',
        'http://tiki.vn/dung-cu-hoc-sinh/c2365',
        'http://tiki.vn/san-pham-ve-giay/c2368',
        'http://tiki.vn/vpp-but-viet/c1858',
        'http://tiki.vn/may-xay-may-ep/c2024',
        'http://tiki.vn/bia-ho-so/c1860',
        'http://tiki.vn/thiet-bi-lam-dep/c2306',
        'http://tiki.vn/dung-cu-an-uong/c2556',
        'http://tiki.vn/dung-cu-ve-sinh/c1987',
        'http://tiki.vn/binh-dun-sieu-toc/c1931',
        'http://tiki.vn/phu-kien-dien-thoai-may-tinh-bang/c1816',
        'http://tiki.vn/tranh-dong-ho/c1981',
        'http://tiki.vn/quat-dien/c2001',
        'http://tiki.vn/dau-goi/c1613',
        'http://tiki.vn/noi-bo-noi/c1983',
        'http://tiki.vn/but-xoa-gom-tay/c1871'
        ]
    if code==1:
        return collection_1
    elif code == 2:
        return collection_2
    elif code == 3:
        return [x+'?limit=40&page=1' for x in collection_4]
    elif code==4:
        return sister_crawl()
    else:
        return collection_1
def crawl1():
    import datetime
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    urllink_collection = getURLlinkcollection(1)
    print('bring it on bitch no. of links = '+str(len(urllink_collection)))
    total_reviews = 0
    for urllink in urllink_collection:
        r = CrawlToFileOneProduct(urllink,origin_time_string,'bestseller')
        total_reviews = total_reviews +r
    print('watch me bitch total reviews = '+ str(total_reviews))
def crawl2():
    import datetime
    import sys
    origin_time_string = str(datetime.datetime.now().strftime('%d%b%p%H%M%S'))
    menulink_collection = getURLlinkcollection(4)
    total_reviews = 0
    category_mode = True
    for menu in menulink_collection:
        product_collection = getProductLinksFromMenu(menu)
        category = normalizeVietnamese( product_collection[0],True)
        print("category  "+category+" product collection " + str(len(product_collection[2])))
        total_category_reviews, total_category_negative_reviews = 0,0
        for urllink in product_collection[2]:
            try:
                r = CrawlToFileOneProduct(urllink,origin_time_string,category,category_mode)
            except:
                print(sys.exc_info())
                continue
            total_category_reviews = total_category_reviews + r[0]
            total_category_negative_reviews = total_category_negative_reviews + r[1]
        print(category + ' total reviews = '+ str(total_category_reviews)+' negative = '+str(total_category_negative_reviews))
        total_reviews = total_reviews + total_category_reviews
    print('total reviews = '+ str(total_reviews))
if __name__ == '__main__':
    crawl2()
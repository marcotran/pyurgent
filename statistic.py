from bs4 import BeautifulSoup
from urllib.request import urlopen
import sys
def read_links():
    with open('tiki.txt','r') as myfile:
        links = myfile.readlines()
        fit_links = [l.split('\n')[0] for l in links]
        print(fit_links)
    return fit_links
def crawl_links():
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
                    continue
                try:
                    p_l = urlopen(l+'?limit=40&page=1')
                    soup_pl = BeautifulSoup(p_l.read(),'html.parser') 
                    product_link = [x for x in soup_pl.find_all(lambda tag: tag.name=='a' and len(tag.attrs)==3 and 'href'in tag.attrs and 'title' in tag.attrs and 'data-id' in tag.attrs) if len(x.find_all('p',{'class':'review'}))==1] 
                    if len(product_link)>0:
                        print(l,end='\n')
                        href_list.append(l+'?limit=40&page=1')
                except:
                    continue
            else:
                continue
    
    return href_list
def AlnumPhrase(phrase):
    real_phrase = ''
    for ch in phrase:
        if ch.isalnum():
            real_phrase = real_phrase +ch
    return real_phrase
def normalizeVietnamese(phrase, without_space):
    from unidecode import unidecode
    m = unidecode(phrase)
    if without_space == False:
        return m
    return AlnumPhrase(m)
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
    domain_category = AlnumPhrase(urllink.split('/')[3])
    try:
        menu_page = urlopen(urllink)
    except:
        print(urllink +  ' ' + str(sys.exc_info()))
        return None
    soup = BeautifulSoup(menu_page.read(),'html.parser')
    category = soup.find_all('div',{'class':'filter-list-box'})
    if len(category) ==1:
        product_type = category[0].find_all('h1')[0].string.strip()
        no_products = int(category[0].find_all('h4')[0].string.strip('(').strip(')'))
    else:
        print("len category "+str(len(category))+" "+urllink)
        product_type = 'unknown'; no_products=0
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
    result = (urllink,domain_category,normalizeVietnamese(product_type,True),no_products,total_reviews,total_negative_reviews)
    print(result)
    return result
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
    with open('tiki_summary_'+origin_time_string,'a') as myfile:
        for i in range(0,40):
            print(sorted_info[i],end = '\n',file=myfile)
    print(sorted_info[:40])
if __name__ == '__main__':
    sister_crawl()
    print('end')
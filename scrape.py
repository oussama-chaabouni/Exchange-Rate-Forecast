import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
from proxy_requests import ProxyRequests
import time
import concurrent.futures
#https://www.mql5.com/en/economic-calendar/united-states/average-hourly-earnings-mm/export
#API URL
'''
def getProxies():
    r = requests.get('https://free-proxy-list.net/')
    soup = BeautifulSoup(r.content, 'html.parser')
    table = soup.find('tbody')
    proxies = []
    for row in table:
        if row.find_all('td')[4].text =='elite proxy':
            proxy = ':'.join([row.find_all('td')[0].text, row.find_all('td')[1].text])
            proxies.append(proxy)
        else:
            pass
    return proxies

def extract(proxy):
    #this was for when we took a list into the function, without conc futures.
    #proxy = random.choice(proxylist)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:80.0) Gecko/20100101 Firefox/80.0'}
    try:
        #change the url to https://httpbin.org/ip that doesnt block anything
        r = requests.get('https://httpbin.org/ip', headers=headers, proxies={'http' : proxy,'https': proxy}, timeout=1)
        print(r.json(), r.status_code)
    except requests.ConnectionError as err:
        print(repr(err))
    return proxy

proxylist = getProxies()
print((proxylist))

#check them all with futures super quick
#with concurrent.futures.ThreadPoolExecutor() as executor:
#        executor.map(extract, proxylist)
        
'''  

indicators_US=['producer-price-index-yy','consumer-price-index-yy','ism-manufacturing-pmi','ism-non-manufacturing-pmi',
'consumer-confidence-index','retail-sales-yy','retail-sales-ex-autos-mm','durable-goods-orders','nonfarm-payrolls',
'industrial-production-mm','trade-balance','housing-starts',
'new-home-sales','building-permit','existing-home-sales','unemployment-rate','adp-nonfarm-employment-change',
'ny-empire-state-manufacturing-index','chicago-pmi','capacity-utilization','factory-orders','durable-goods-orders-ex-transportation',
'business-inventories-mm','pce-price-index-yy','pce-price-index-mm','core-pce-price-index-yy','core-pce-price-index-mm',
'average-hourly-earnings-mm','adp-nonfarm-employment-change','fed-interest-rate-decision US']

indicators_EU=['producer-price-index-yy','consumer-price-index-yy','markit-manufacturing-pmi',
'markit-manufacturing-pmi','markit-services-pmi','consumer-confidence-indicator',
'retail-sales-yy','industrial-production-yy','trade-balance','construction-output',
'unemployment-rate','markit-composite-pmi','ecb-deposit-rate-decision','zew-indicator-of-economic-sentiment','fed-interest-rate-decision US']

indicators_interest = ['snb-interest-rate-decision','boe-interest-rate-decision']


def get_data(pageNumber,indicator):


    url = "https://www.mql5.com/en/economic-calendar/united-states/"+indicator+"/history?page=" + pageNumber
    r=requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    #print(r.status_code)
    content = BeautifulSoup(r.text.replace("\r", "").replace("\t", "").replace("\K", "").replace("\n", ""),'html.parser')
    date=[]
    actual=[]
    forecast=[]
    previous=[]

    col=[]
    col1=[]
    col2=[]
    col3=[]

    for div in content.findAll('div', attrs={'class':'event-table-history__period'}):
        date.append(div.text)
        
    for span in content.findAll('span', attrs={'class':'event-table-history__actual__value ec-tooltip-value'}):
        actual.append(span.text)  

    for div in content.findAll('div', attrs={'class':'event-table-history__forecast'}):
        forecast.append(div.text)
        
    for span in content.findAll('span', attrs={'class':'prev-revised ec-tooltip-value'}):
        previous.append(span.text)
        
        
    col.extend(date)
    col1.extend(actual)
    col2.extend(forecast)
    col3.extend(previous)

    df = pd.DataFrame(data=[col, col1,col2,col3])
    df=df.T   

    df.rename( columns={0 :'Date', 1:'Actual',2:'Forecast',3:'Previous'}, inplace=True ) 

    return df
countries =['switzerland','united-kingdom']

def get_data_interest(pageNumber,indicator,countries):


    url = "https://www.mql5.com/en/economic-calendar/"+ countries + '/' +indicator+"/history?page=" + pageNumber
    r=requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    #print(r.status_code)
    content = BeautifulSoup(r.text.replace("\r", "").replace("\t", "").replace("\K", "").replace("\n", ""),'html.parser')
    date=[]
    actual=[]
    forecast=[]
    previous=[]

    col=[]
    col1=[]
    col2=[]
    col3=[]

    for div in content.findAll('div', attrs={'class':'event-table-history__period'}):
        date.append(div.text)
        
    for span in content.findAll('span', attrs={'class':'event-table-history__actual__value ec-tooltip-value'}):
        actual.append(span.text)  

    for div in content.findAll('div', attrs={'class':'event-table-history__forecast'}):
        forecast.append(div.text)
        
    for span in content.findAll('span', attrs={'class':'prev-revised ec-tooltip-value'}):
        previous.append(span.text)
        
        
    col.extend(date)
    col1.extend(actual)
    col2.extend(forecast)
    col3.extend(previous)

    df = pd.DataFrame(data=[col, col1,col2,col3])
    df=df.T   

    df.rename( columns={0 :'Date', 1:'Actual',2:'Forecast',3:'Previous'}, inplace=True ) 

    return df




#print(len(indicators_US)+len(indicators_EU))


def get_data_country(country,indi):
    d = pd.DataFrame()
    for i in indi:
        for x in ["1","2","3","4"]:
            d=d.append(get_data(x,i))
            time.sleep(10)
        d = d.reset_index(drop=True)
        d.drop(d.columns[0], axis=1)
        d.to_csv('./data'+'/'+country + '.' + i + '.csv', mode='a' ,header = False)
        time.sleep(20)


def get_data_in(indi):
    d = pd.DataFrame()
    for i in indi:
        for c in countries:
            for x in ["1","2","3","4"]:
                d=d.append(get_data_interest(x,i,c))
                time.sleep(10)
            d = d.reset_index(drop=True)
            d.drop(d.columns[0], axis=1)
            d.to_csv('./data'+'/' + c + '.' + i +'.csv', mode='a' ,header = False)
            time.sleep(20)

get_data_in(indicators_interest)
         
#get_data_country('united-states',indicators_US)  
#time.sleep(50)      
#get_data_country('european-union',indicators_EU)


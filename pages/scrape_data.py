from dash import dcc, html, Input, Output, callback
from dash import Dash, dash_table
from dash import dcc ,State,Dash
from dash import  html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import datetime
import requests
from requests.exceptions import ConnectionError
from bs4 import BeautifulSoup
from proxy_requests import ProxyRequests
import time
import concurrent.futures
import pandas_datareader.data as web
import urllib.request
import urllib.error
import bz2
import pickle
from functools import reduce
from datetime import datetime
from statsmodels.tsa.stattools import kpss
from dateutil.relativedelta import relativedelta
import io
import base64

indicators_US=['producer-price-index-yy','consumer-price-index-yy','ism-manufacturing-pmi','ism-non-manufacturing-pmi',
'consumer-confidence-index','retail-sales-yy','retail-sales-ex-autos-mm','durable-goods-orders','nonfarm-payrolls',
'industrial-production-mm','trade-balance','housing-starts',
'new-home-sales','building-permits','existing-home-sales','unemployment-rate','adp-nonfarm-employment-change',
'ny-empire-state-manufacturing-index','chicago-pmi','capacity-utilization','factory-orders','durable-goods-orders-ex-transportation',
'business-inventories-mm','pce-price-index-yy','pce-price-index-mm','core-pce-price-index-yy','core-pce-price-index-mm',
'average-hourly-earnings-mm','fed-interest-rate-decision']

indicators_EU=['producer-price-index-yy','consumer-price-index-yy',
'markit-manufacturing-pmi','markit-services-pmi','consumer-confidence-indicator',
'retail-sales-yy','industrial-production-yy','trade-balance','construction-output-yy',
'unemployment-rate','markit-composite-pmi','ecb-deposit-rate-decision','zew-indicator-of-economic-sentiment','ecb-interest-rate-decision']

indicators_interest = ['snb-interest-rate-decision','boe-interest-rate-decision']

countries =['switzerland','united-kingdom']


df = pd.read_csv("./data/macro_data_final2.csv").set_index('Date')
df = df.iloc[::-1]

data_frame=pd.read_csv("./data/macro_data_final2.csv")
data_frame = data_frame.iloc[::-1]

listoptions = []

listindicators = []
listindicators_eu = []
for i in list(df.columns):
    dicti={}
    dicti['label'] = i
    dicti['value'] = i
    listoptions.append(dicti)

for i in list(indicators_US):
    dic={}
    dic['label'] = i
    dic['value'] = i
    listindicators.append(dic)

for i in list(indicators_EU):
    dic={}
    dic['label'] = i
    dic['value'] = i
    listindicators_eu.append(dic)








def get_data(country,indicator):

    url = "https://www.mql5.com/en/economic-calendar/"+ country + '/' +indicator+"/export"

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, "./data/"+country + '.' + indicator+'.csv')


def get_data_interest(country,indicator):
    url = "https://www.mql5.com/en/economic-calendar/"+ country + '/' +indicator+"/export"


    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, "./data/"+country + '.' + indicator+'.csv')






layout = html.Div([
                                    
                                            html.Div([
                                                
                                                    html.Div([
                                                        html.Div([
                                                            dcc.Dropdown(
                                                                id='my_dropdown2',
                                                                options=indicators_US,
                                                                value=['producer-price-index-yy'],
                                                                multi=True,
                                                                clearable=False,
                                                                style={"width": "100%"}
                                                            ),

                                                            html.Div([
                                                                html.Button("SELECT ALL", id="select-all", n_clicks=0),
                                                                html.Button('Scrape United States Data', id='submit-val', n_clicks=0),

                                                                html.Div(id='container-button-basic',
                                                                        children='Enter a value and press submit'),
                                                            ]),
                                                            ],style={"width":"30%", "margin-right":"20%",'display': 'inline-block'}),
                                                        html.Div([
                                                        html.Div([
                                                            dcc.Dropdown(
                                                                id='my_dropdown3',
                                                                options=indicators_EU,
                                                                value=['producer-price-index-yy'],
                                                                multi=True,
                                                                clearable=False,
                                                                style={"width": "100%"}
                                                            ),

                                                            html.Div([
                                                                html.Button("SELECT ALL", id="select-all2", n_clicks=0),
                                                                html.Button('Scrape European Union Data', id='submit-val2', n_clicks=0),

                                                                html.Div(id='container-button-basic2',
                                                                        children='Enter a value and press submit'),
                                                            ],style={"width": "80%"}),



                                                                
                                                            ],style={}),


                                                                    ],style={"width": "30%",'textalign':"center",'display': 'inline-block'}),


                                                                
                                                                ],style={"margin-bottom":"10%","display": "flex"}),
                                                            

]),



    html.Div([
                                                    html.Div([
                                                        html.Div([
                                                            dcc.Dropdown(
                                                                id='my_dropdown4',
                                                                options=indicators_interest,
                                                                value=['snb-interest-rate-decision'],
                                                                multi=True,
                                                                clearable=False,
                                                                style={"width": "100%",'display': 'inline-block'}
                                                            ),

                                                            html.Div([
                                                                html.Button('Scrape Interest Data', id='submit-val3', n_clicks=0),

                                                                html.Div(id='container-button-basic3',
                                                                        children='Enter a value and press submit'),
                                                            ],style={"width": "80%",'display': 'inline-block'}),



                                                        
                                                        ],style={'display': 'inline-block'}),
                                                    ],style={'display': 'inline-block',"margin-right":"20%"}),









                                                    html.Div([
                                                            html.Button('Update EUR/USD Exchange rate', id='submit-val-eur-usd', n_clicks=0),

                                                            html.Div(id='container-button-basic-eur-usd',
                                                                    children='Enter a value and press submit')
                                                    ],style={"width": "50%",'textalign':"center",'display': 'inline-block'}),


                                                ],style={"display": "flex"})
 ])


@callback(Output("my_dropdown2", "value"), Input("select-all", "n_clicks"),prevent_initial_call=True)
def select_all(n_clicks):
    return [option["value"] for option in listindicators]

@callback(Output("my_dropdown3", "value"), Input("select-all2", "n_clicks"),prevent_initial_call=True)
def select_all(n_clicks):
    return [option["value"] for option in listindicators_eu]


@callback(    
    Output('container-button-basic', 'children'),
    Input('submit-val', 'n_clicks'),
    State('my_dropdown2', 'value'),
    prevent_initial_call=True
)

def get_data_in(n_clicks,value):

    print(value)
    country = 'united-states'
    for i in value:
        get_data(country,i)
        time.sleep(3)   
        #d.drop(d.columns[0], axis=1)
        #d.columns = ['Date','Actual','Forecast','Previous']
    #d.to_csv('./data'+'/'+country + '.' + i + '.csv', mode='a' ,header = None,index=None)
    #d = pd.DataFrame()
    time.sleep(5)
    
    return 'Finished Scraping data'

@callback(    
    Output('container-button-basic2', 'children'),
    Input('submit-val2', 'n_clicks'),
    State('my_dropdown3', 'value'),
    prevent_initial_call=True
)

def get_data_in(n_clicks,value):

    print(value)
    country = 'european-union'
    for i in value:
        get_data(country,i)
        time.sleep(3)   
    time.sleep(5)
    
    return 'Finished Scraping data'


    

@callback(    
    Output('container-button-basic3', 'children'),
    Input('submit-val3', 'n_clicks'),
    State('my_dropdown4', 'value'),
    prevent_initial_call=True
)



def get_data_interest_(n_clicks,value):

    print(value)
    for i in value:
        if i=='snb-interest-rate-decision':
            country='switzerland'
        else:
            country='united-kingdom'
        
        get_data_interest(country,i)
        time.sleep(3)   
        
    
    return 'Finished Scraping data'

                
@callback(    
    Output('container-button-basic-eur-usd', 'children'),
    Input('submit-val-eur-usd', 'n_clicks'),
    prevent_initial_call=True
)

def update_exchange_rate(n_clicks):
    

    import yfinance as yf    
    from datetime import datetime
        
    #Load Stock price
    dfs = yf.download("EURUSD=X", start= datetime(2007,5,1),interval='1mo')
    dfs.index =  pd.to_datetime(dfs.index, format='%Y-%m').strftime("%Y-%m")
    dfs = dfs[~dfs.index.duplicated(keep='last')]
    dfs = dfs.iloc[::1]

    dfs = dfs.rename(columns={'Close': 'EUR/USD'})
    dfs=dfs.sort_values(by="Date", ascending=[0])
    
    
    with open('./data/EURUSD=X.pkl', 'wb') as f:
        pickle.dump(dfs, f)


    #dfs.to_csv("./data/EURUSD=X.csv",index=True,mode='a')

    return 'Finished Updating data'





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
import dash_bootstrap_components as dbc

indicators_United_States=['producer-price-index-yy','consumer-price-index-yy','ism-manufacturing-pmi','ism-non-manufacturing-pmi',
'consumer-confidence-index','retail-sales-yy','retail-sales-ex-autos-mm','durable-goods-orders','nonfarm-payrolls',
'industrial-production-mm','trade-balance','housing-starts',
'new-home-sales','building-permits','existing-home-sales','unemployment-rate','adp-nonfarm-employment-change',
'ny-empire-state-manufacturing-index','chicago-pmi','capacity-utilization','factory-orders','durable-goods-orders-ex-transportation',
'business-inventories-mm','pce-price-index-yy','pce-price-index-mm','core-pce-price-index-yy','core-pce-price-index-mm',
'average-hourly-earnings-mm','fed-interest-rate-decision']

indicators_European_Union=['producer-price-index-yy','consumer-price-index-yy',
'markit-manufacturing-pmi','markit-services-pmi','consumer-confidence-indicator',
'retail-sales-yy','industrial-production-yy','trade-balance','construction-output-yy',
'unemployment-rate','markit-composite-pmi','ecb-deposit-rate-decision','zew-indicator-of-economic-sentiment','ecb-interest-rate-decision']


indicators_United_Kingdom=['average-weekly-earnings-regular-pay','average-weekly-earnings-total-pay','claimant-count-change',
'unemployment-rate','cpi-yy','core-cpi-mm','core-cpi-yy','ppi-input-mm',
'ppi-input-yy','ppi-output-mm','ppi-output-yy','rpi-mm','rpi-yy'
]



indicators_interest = ['snb-interest-rate-decision','boe-interest-rate-decision']

countries =['switzerland','united-kingdom']


df = pd.read_csv("./data/macro_data_final2.csv").set_index('Date')
df = df.iloc[::-1]

data_frame=pd.read_csv("./data/macro_data_final2.csv")
data_frame = data_frame.iloc[::-1]

listoptions = []

listindicators = []
listindicators_European_Union = []
listindicators_United_Kingdom = []
listindicators_United_States = []

for i in list(df.columns):
    dicti={}
    dicti['label'] = i
    dicti['value'] = i
    listoptions.append(dicti)

for i in list(indicators_United_States):
    dic={}
    dic['label'] = i
    dic['value'] = i
    listindicators_United_States.append(dic)

for i in list(indicators_European_Union):
    dic={}
    dic['label'] = i
    dic['value'] = i
    listindicators_European_Union.append(dic)


for i in list(indicators_United_Kingdom):
    dic={}
    dic['label'] = i
    dic['value'] = i
    listindicators_United_Kingdom.append(dic)





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



SIDEBAR_STYLE = {
    "position": "fixed",
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

CONTENT_STYLE = {
    "margin-left": "25rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "font-size":"15px",
}
	
	
	
# 	AUD/CAD
# NZD/USD	
# AUD/USD	
# 	CHF/JPY
# 	AUD/JPY
# 	AUD/NZD

data = {'currencies': ['EUR_USD', 'EUR_GBP', 'EUR_AUD', 'EUR_CAD','EUR_JPY','EUR_CHF','USD_JPY','USD_CAD','USD_CHF','GBP_USD','GBP_CHF','GBP_JPY']}
#data = {'currencies': ['EUR/USD', 'EUR/GBP', 'EUR/AUD', 'EUR/CAD','EUR/JPY','EUR/CHF','USD/JPY','USD/CAD','USD/CHF','GBP/USD','GBP/CHF','GBP/JPY']}

df_currencies = pd.DataFrame(data)

def generate_currency_button(currencies):

    return dbc.DropdownMenuItem(
                      str(currencies),
                      className="mr-1",
                      href='/page-1/'+currencies,
                      id=str(currencies))


def generate_currency(currencies,country1,country2):

    return html.Div([
                                                
                                                    html.Div([
                                                        html.Div([
                                                            dcc.Dropdown(
                                                                id='my_dropdown2'+str(currencies),
                                                                options=globals()['indicators_'+ country1],
                                                                value=['producer-price-index-yy'],
                                                                multi=True,
                                                                clearable=False,
                                                                style={"width": "100%"}
                                                            ),

                                                            html.Div([
                                                                html.Button("SELECT ALL", id="select-all"+str(currencies), n_clicks=0),
                                                                html.Button('Scrape ' + country1 +' Data', id='submit-val'+str(currencies), n_clicks=0),

                                                                html.Div(id='container-button-basic'+str(currencies),
                                                                        children='Enter a value and press submit'),
                                                            ]),
                                                            ],style={"width":"30%", "margin-right":"20%",'display': 'inline-block'}),
                                                        html.Div([
                                                        html.Div([
                                                            dcc.Dropdown(
                                                                id='my_dropdown3'+str(currencies),
                                                                options=globals()['indicators_'+ country2],
                                                                value=['producer-price-index-yy'],
                                                                multi=True,
                                                                clearable=False,
                                                                style={"width": "100%"}
                                                            ),

                                                            html.Div([
                                                                html.Button("SELECT ALL", id="select-all2"+str(currencies), n_clicks=0),
                                                                html.Button('Scrape ' + country2 +' Data', id='submit-val2'+str(currencies), n_clicks=0),

                                                                html.Div(id='container-button-basic2'+str(currencies),
                                                                        children='Enter a value and press submit'),
                                                            ],style={"width": "80%"}),



                                                                
                                                            ],style={}),


                                                                    ],style={"width": "30%",'textalign':"center",'display': 'inline-block'}),


                                                                
                                                                ],style={"margin-bottom":"10%","display": "flex"}),
                                                            





            html.Div([
                                                            html.Div([
                                                                html.Div([
                                                                    dcc.Dropdown(
                                                                        id='my_dropdown4'+str(currencies),
                                                                        options=indicators_interest,
                                                                        value=['snb-interest-rate-decision'],
                                                                        multi=True,
                                                                        clearable=False,
                                                                        style={"width": "100%",'display': 'inline-block'}
                                                                    ),

                                                                    html.Div([
                                                                        html.Button('Scrape Interest Data', id='submit-val3'+str(currencies), n_clicks=0),

                                                                        html.Div(id='container-button-basic3'+str(currencies),
                                                                                children='Enter a value and press submit'),
                                                                    ],style={"width": "80%",'display': 'inline-block'}),



                                                                
                                                                ],style={'display': 'inline-block'}),
                                                            ],style={'display': 'inline-block',"margin-right":"20%"}),









                                                            html.Div([
                                                                    html.Button('Update '+ currencies +' Exchange rate', id='submit-val-rate'+str(currencies), n_clicks=0),

                                                                    html.Div(id='container-button-basic-rate'+str(currencies),
                                                                            children='Enter a value and press submit')
                                                            ],style={"width": "50%",'textalign':"center",'display': 'inline-block'}),


                                                        ],style={"display": "flex"})
 ])


sidebar2 = html.Div(
    [
        html.H2("Select Exchange Rate", className="display-4"),
        html.Hr(),
        # html.P(
        #     "A simple sidebar layout with navigation links", className="lead"
        # ),



        
        dbc.Nav(
            [
            dbc.DropdownMenu(
            children=[

                generate_currency_button(i) for i in df_currencies['currencies']
            ],
            
                        nav=True,
            in_navbar=True,
            label="More",
        ),
            ],
            vertical=False,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
content2 = html.Div(id="page-content2", style=CONTENT_STYLE)


layout = html.Div([
    html.Div([dcc.Location(id="url2"), sidebar2, content2])
    
])




data = {'currencies': ['EUR_USD', 'EUR_GBP', 'EUR_AUD', 'EUR_CAD','EUR_JPY','EUR_CHF','USD_JPY','USD_CAD','USD_CHF','GBP_USD','GBP_CHF','GBP_JPY']}



@callback(Output("page-content2", "children"), [Input("url2", "pathname")])
def render_page_content_(pathname="/page-1"):

    for i in df_currencies['currencies']:
        if pathname == "/page-1/" + i: 
            if i=='EUR_USD':
                return generate_currency(i,'European_Union','United_States')
           
            if i=='EUR_GBP':

                return generate_currency(i,'European_Union','United_Kingdom')

            if i=='EUR_AUD':

                return generate_currency(i,'European_Union','Australia')

            if i=='EUR_CAD':

                return generate_currency(i,'European_Union','Canada')

            if i=='EUR_JPY':

                return generate_currency(i,'European_Union','Japan')


            if i=='EUR_CHF':

                return generate_currency(i,'European_Union','Switzerland')

            if i=='USD_JPY':

                return generate_currency(i,'United_States','Japan')

            if i=='USD_CAD':

                return generate_currency(i,'United_States','Canada')

            if i=='USD_CHF':

                return generate_currency(i,'United_States','Switzerland')

            if i=='GBP_USD':

                return generate_currency(i,'United_Kingdom','United States')


            if i=='GBP_CHF':

                return generate_currency(i,'United_Kingdom','Switzerland')


            if i=='GBP_JPY':

                return generate_currency(i,'United_Kingdom','Japan')                






# def select_all(j):
#     def test():
#         print(j)
#         countriess=['European_Union','United_Kingdom','United_States']
#         if i == 'EUR_USD':
#             print(i)
#             print('ok')
#             c1='United_States'
#             c2='European_Union'
#             x=[option["value"] for option in globals()['listindicators_'+ c1]]
        
#         elif i== 'EUR_GBP':
#             c1='European_Union'
#             c2='United_Kingdom'
#             x=[option["value"] for option in globals()['listindicators_'+ c1]]
#         else:
#             x=[1]
        
#         return x

# for i in df_currencies['currencies']:

#     callback(Output("my_dropdown2"+i, "value"), Input("select-all"+i, "n_clicks"),prevent_initial_call=True)(select_all())



def test(a,b):
    if b == '/page-1/EUR_USD':
        c1='European_Union'
        c2='United_States'
        x= [option["value"] for option in globals()['listindicators_'+ c1]]
            
            
    elif b== '/page-1/EUR_GBP':
        c1='European_Union'
        c2='United_Kingdom'
        x= [option["value"] for option in globals()['listindicators_'+ c1]]    
            
    return x    




for i in df_currencies['currencies']:

    callback(Output("my_dropdown2"+i, "value"), Input("select-all"+i, "n_clicks"), State('url2', 'pathname'),prevent_initial_call=True)(test)
    # def select_all(n_clicks):
    #     countriess=['European_Union','United_Kingdom','United_States']
    #     if i == 'EUR_USD':
    #         county1='United_States'
    #         country2='European_Union'
        
    #     elif i== 'EUR_GBP':
    #         county1='United_States'
    #         country2='European_Union'
        
    #     return [option["value"] for option in globals()['listindicators_'+ country1]]



def test2(a,b):

    if b == '/page-1/EUR_USD':
        c1='European_Union'
        c2='United_States'
        x= [option["value"] for option in globals()['listindicators_'+ c2]]
            
            
    elif b== '/page-1/EUR_GBP':
        c1='European_Union'
        c2='United_Kingdom'
    x= [option["value"] for option in globals()['listindicators_'+ c2]]        
    
            
    return x

for i in df_currencies['currencies']:

    callback(Output("my_dropdown3"+i, "value"), Input("select-all2"+i, "n_clicks"), State('url2', 'pathname'),prevent_initial_call=True)(test2)



def get_data_in(n_clicks,value,b):

    if b == '/page-1/EUR_USD':
        country='european-union'
    elif b== '/page-1/EUR_GBP':
        country='european-union'
    for i in value:
        get_data(country,i)
        time.sleep(3)   
        #d.drop(d.columns[0], axis=1)
        #d.columns = ['Date','Actual','Forecast','Previous']
    #d.to_csv('./data'+'/'+country + '.' + i + '.csv', mode='a' ,header = None,index=None)
    #d = pd.DataFrame()
    time.sleep(5)
    
    return 'Finished Scraping data'

    
for i in df_currencies['currencies']:



    callback(    
        Output('container-button-basic'+i, 'children'),
        Input('submit-val'+i, 'n_clicks'),
        State('my_dropdown2'+i, 'value'),
        State('url2', 'pathname'),
        prevent_initial_call=True
    )(get_data_in)




def get_data_in(n_clicks,value,b):

    if b == '/page-1/EUR_USD':
        country = 'united-states'
    elif b== '/page-1/EUR_GBP':
        country = 'united-kingdom'

    for i in value:
        get_data(country,i)
        time.sleep(3)   
    time.sleep(5)
    
    return 'Finished Scraping data'


for i in df_currencies['currencies']:

    callback(    
        Output('container-button-basic2'+i, 'children'),
        Input('submit-val2'+i, 'n_clicks'),
        State('my_dropdown3'+i, 'value'),
        State('url2', 'pathname'),
        prevent_initial_call=True
    )(get_data_in)




    

@callback(    
    Output('container-button-basic3', 'children'),
    Input('submit-val3', 'n_clicks'),
    State('my_dropdown4', 'value'),
    prevent_initial_call=True
)



def get_data_interest_(n_clicks,value):

    for i in value:
        if i=='snb-interest-rate-decision':
            country='switzerland'
        else:
            country='united-kingdom'
        
        get_data_interest(country,i)
        time.sleep(3)   
        
    
    return 'Finished Scraping data'


def update_exchange_rate(n_clicks,path):

    currency = path.split('/page-1/', 1)[-1].replace("_", "")
    import yfinance as yf    
    from datetime import datetime
        
    #Load Stock price
    dfs = yf.download(currency+"=X", start= datetime(2007,5,1),interval='1mo')
    dfs.index =  pd.to_datetime(dfs.index, format='%Y-%m').strftime("%Y-%m")
    dfs = dfs[~dfs.index.duplicated(keep='last')]
    dfs = dfs.iloc[::1]

    dfs = dfs.rename(columns={'Close': currency})
    dfs=dfs.sort_values(by="Date", ascending=[0])
    
    
    with open('./data/'+currency+'=X'+'.pkl', 'wb') as f:
        pickle.dump(dfs, f)


    #dfs.to_csv("./data/EURUSD=X.csv",index=True,mode='a')

    return 'Finished Updating data'



for i in df_currencies['currencies']:
    callback( Output('container-button-basic-rate'+i, 'children'),
    Input('submit-val-rate'+i, 'n_clicks'),
    State('url2', 'pathname'),
    prevent_initial_call=True)(update_exchange_rate)







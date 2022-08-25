from dash import dcc, html, Input, Output, callback
import pandas as pd
import dash_bootstrap_components as dbc

import dash
from dash import Dash, dash_table
from dash import dcc ,State,Dash
from dash import  html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

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
from datetime import date

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
#data_frame = data_frame.iloc[::-1]
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

PAGE_SIZE = 10


columns = list(df.columns)




def update_data(end_date):
    file_name = './data/united-states.{}.csv'
    df_list = []
    for i in indicators_US:   
        df_list.append(pd.read_csv(file_name.format(i), sep='\t',header=0,names=['Date', i +' US', 'ForecastValue','PreviousValue']))

    for j in  range(0,len(indicators_US)):
        df_list[j]['Date'] = pd.to_datetime(df_list[j]['Date'], format='%Y-%m').dt.strftime("%Y-%m")
        df_list[j] = df_list[j].set_index('Date').sort_values(by="Date", ascending=[0])
        df_list[j] = df_list[j][~df_list[j].index.duplicated(keep='first')]    
        

    data_frames=[[]for i in range (len(indicators_US))]
    for k in  range(0,len(indicators_US)):
        data_frames[k]=df_list[k][[df_list[k].columns[0]]] 

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on='Date',
                                                how='outer'), data_frames)
    df_merged = df_merged.sort_values(by="Date", ascending=[0])
    df_merged [~df_merged .index.duplicated(keep='first')]
    df_merged['fed-interest-rate-decision US'] = df_merged['fed-interest-rate-decision US'].bfill().ffill()
    


    
    mask = (df_merged.index >= '2007-05') & (df_merged.index <= end_date)
    df_merged = df_merged.loc[mask]

    df_merged.loc[:, df_merged.columns != 'fed-interest-rate-decision US'] = df_merged.loc[:, df_merged.columns != 'fed-interest-rate-decision US'].ffill()

###########################################################

    file_name_euro = './data/european-union.{}.csv'
    df_list_euro = []
    for i in indicators_EU:   
        df_list_euro.append(pd.read_csv(file_name_euro.format(i), sep='\t',header=0,names=['Date', i +' EuroZone', 'ForecastValue','PreviousValue']))

    for j in  range(0,len(indicators_EU)):
        df_list_euro[j]['Date'] = pd.to_datetime(df_list_euro[j]['Date'], format='%Y-%m').dt.strftime("%Y-%m")
        df_list_euro[j] = df_list_euro[j].set_index('Date').sort_values(by="Date", ascending=[0])
        df_list_euro[j] = df_list_euro[j][~df_list_euro[j].index.duplicated(keep='first')]    
        

    data_frames_EuroZone=[[]for i in range (len(indicators_EU))]
    for k in  range(0,len(indicators_EU)):
        data_frames_EuroZone[k]=df_list_euro[k][[df_list_euro[k].columns[0]]]  



    df_merged_EuroZone = reduce(lambda  left,right: pd.merge(left,right,on='Date',
                                                how='outer'), data_frames_EuroZone)
    df_merged_EuroZone = df_merged_EuroZone.sort_values(by="Date", ascending=[0])
    df_merged_EuroZone [~df_merged_EuroZone .index.duplicated(keep='first')]
    df_merged_EuroZone['ecb-interest-rate-decision EuroZone'] = df_merged_EuroZone['ecb-interest-rate-decision EuroZone'].bfill().ffill()

    mask = (df_merged_EuroZone.index >= '2007-05') & (df_merged_EuroZone.index <= end_date)
    df_merged_EuroZone = df_merged_EuroZone.loc[mask]

    df_merged_EuroZone.loc[:, df_merged_EuroZone.columns != 'fed-interest-rate-decision US'] = df_merged_EuroZone.loc[:, df_merged_EuroZone.columns != 'fed-interest-rate-decision US'].ffill()

######################################################

    with open('./data/EURUSD=X.pkl', 'rb') as f:
        dfs = pickle.load(f)

    dfr0 = pd.read_csv("./data/united-kingdom.boe-interest-rate-decision.csv", sep='\t')
    dfr1 = pd.read_csv("./data/switzerland.snb-interest-rate-decision.csv", sep='\t')

    dfr0['Date'] = pd.to_datetime(dfr0['Date'], format='%Y-%m').dt.strftime("%Y-%m")
    dfr1['Date'] = pd.to_datetime(dfr1['Date'], format='%Y-%m').dt.strftime("%Y-%m")

    dfr0 = dfr0.rename(columns={'ActualValue': 'boe-interest-rate-decision UK'}).set_index('Date').sort_values(by="Date", ascending=[0])
    dfr1 = dfr1.rename(columns={'ActualValue': 'snb-interest-rate-decision Switzerland'}).set_index('Date').sort_values(by="Date", ascending=[0])


    dfr0 = dfr0.bfill().ffill()
    dfr1 = dfr1.bfill().ffill()

    mask = (dfs.index >= '2007-05') & (dfs.index <= end_date)
    dfs = dfs.loc[mask]
    data_frames_Merged=[df_merged,df_merged_EuroZone,dfr0[['boe-interest-rate-decision UK']],dfr1[['snb-interest-rate-decision Switzerland']],dfs['EUR/USD'].iloc[::-1]]




    dfm = reduce(lambda  left,right: pd.merge(left,right,on='Date',
                                                how='left'), data_frames_Merged)


    dfm = dfm.sort_values(by="Date", ascending=[0])
    dfm=dfm [~dfm .index.duplicated(keep='first')]
    dfm['boe-interest-rate-decision UK'] = dfm['boe-interest-rate-decision UK'].bfill().ffill()
    dfm['snb-interest-rate-decision Switzerland'] = dfm['snb-interest-rate-decision Switzerland'].bfill().ffill()
    #df['EUR/USD'] = df['EUR/USD'].bfill().ffill().interpolate()

    mask = (dfm.index >= '2007-05') & (dfm.index <= end_date)
    dfm = dfm.loc[mask]

    
    dfm[1:]=dfm[1:].bfill()

    for i in dfm.columns:
            dfm[i]=dfm[i].shift(-1, axis = 0)
            dfm[i] = dfm[i].ffill()
    
    #df['EUR/USD']=dfs['EUR/USD'].values

    # for i in dfm.columns:

    #     if np.isnan(dfm[i][:1].item()) :
    #         print(i)
    #         dfm[i] = dfm[i].ffill(limit=None)
    #         dfm[i]=dfm[i].shift(-1, axis = 0)
    #         dfm[i] = dfm[i].ffill(limit=None)
    #         dfm = dfm.rename(columns={i: i + ' lagged'})

        

            
        

    
    dfm.to_csv('./data/macro_data_final2.csv',index=True)


def update_graph_correlation():    
    df_corr = df.corr() # Generate correlation matrix

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x = df_corr.columns,
            y = df_corr.index,
            z = np.array(df_corr)
        )
    )
    fig.update_layout(
        autosize=False,
        width=1200,
        height=1200,
#         margin=dict(
#             l=50,
#             r=50,
#             b=100,
#             t=100,
#             pad=4
#         ),

    )

    return [dcc.Graph(figure=fig)]
def get_prediction():
    return [dcc.Graph(figure=model_performance('lgbm_clf')[1])]




layout = html.Div([
                   html.Div([
                                            dcc.Dropdown(
                                                id='my_dropdown',
                                                options=listoptions,
                                                value=['EUR/USD'],
                                                multi=True,
                                                clearable=False,
                                                style={"width": "50%",'textalign':"center"}
                                            ),
                                        ],style={"width": "100%",'textalign':"center"}),
                                        

                                        html.Div([
                                            dcc.Graph(id='the_graph')
                                        ]),
                                        html.Div([
                                            dcc.Graph(id='the_graph_corr',style={'position':"center"})
                                        ]),

                                                html.Button(
                                                    ['Update'],
                                                    id='btn'
                                                ),
                                                dcc.DatePickerSingle(
                                                    id='my-date-picker-single',
                                                    display_format='MM YYYY',
                                                    min_date_allowed=date(2007, 6, 1),
                                                    max_date_allowed=date(2030, 9, 1),
                                                    date=date(2022, 8, 1)
                                                
                                                ),
                                                
                                                    
                                                    html.Div(dash_table.DataTable( 
                                                        
                                                    id='datatable-paging',   
                                                    data=data_frame.to_dict('records'),
                                                    columns=[{'id': c, 'name': c} for c in data_frame.columns],
                                                        page_current=0,
                                                    page_size=PAGE_SIZE,
                                                    page_action='custom',                                                      

                                                        style_data_conditional=[
                                                            
                {
                    'if': {
                        'filter_query': '{{{}}} is blank'.format(col),
                        'column_id': col
                    },
                    'backgroundColor': 'tomato',
                    'color': 'white'
                } for col in df.columns                                                            
                                                        ],
                                                    style_table={'height': '100%', 'overflowY': 'auto'}
                                                    
                                                    )   
                                                    ),

                                        html.Div(children=update_graph_correlation(),id="graph_correlation"),
                                        #html.Div(children=get_prediction(),id='stats_prediction')
       ])
                                            
@callback(
    Output(component_id='the_graph_corr', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph_correlation_(my_dropdown):    
     # Generate correlation matrix
    df = pd.read_csv("./data/macro_data_final2.csv").set_index('Date')
    df = df.iloc[::-1]
    df1=pd.DataFrame()
    for label in my_dropdown:
        
        
        df1[label] = df[label]

        df_corr = df1.corr()

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                x = df_corr.columns,
                y = df_corr.index,
                z = np.array(df_corr)
            )
        )
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
    #         margin=dict(
    #             l=50,
    #             r=50,
    #             b=100,
    #             t=100,
    #             pad=4
    #         ),

        )

    return fig

@callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    Input('my-date-picker-single', 'date'),
    Input("btn", "n_clicks")
    
    )

def update_table(page_current,page_size,date_value,n_clicks):
    date_object = date.fromisoformat(date_value)
    date_string = date_object.strftime('%Y-%m')
    if n_clicks is None:
        #update_data(date_string)
        data_frame=pd.read_csv("./data/macro_data_final2.csv")
        return data_frame.iloc[
                page_current*page_size:(page_current+ 1)*page_size
                ].to_dict('records')

    #data_frame=update_data()
    update_data(date_string)
    data_frame=pd.read_csv("./data/macro_data_final2.csv")
    return data_frame.iloc[
                page_current*page_size:(page_current+ 1)*page_size
            ].to_dict('records')




@callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    import plotly
    df = pd.read_csv("./data/macro_data_final2.csv").set_index('Date')
    df = df.iloc[::-1]
    full_data= df.copy()
    full_data['Date'] = full_data.index
    fig = plotly.tools.make_subplots(specs=[[{"secondary_y": True}]])
#     a,fig =  model_performance('lgbm_clf')

    for label in my_dropdown:
        if label == "Volume":
            fig.add_trace(go.Bar(
            x = full_data["Date"],
            y = full_data["Volume"],
            #mode = "lines",
            name = label),
                secondary_y = True
        )
           
        else:
            fig.add_trace(go.Scatter(
            x = full_data["Date"],
            y = full_data[label],
            mode = 'lines',
            name = label,
        ))
   
       
    return fig

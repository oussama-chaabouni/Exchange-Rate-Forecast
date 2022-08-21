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



df = pd.read_csv("macro_data_final2.csv").set_index('Date')
df = df.iloc[::-1]

data_frame=pd.read_csv("macro_data_final2.csv")
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


#print(len(indicators_US)+len(indicators_EU))


def update_data():
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
    if df_merged[:1].isnull().sum(axis = 1).item()> 0 :
        df_merged = df_merged[1:len(df_list[0])-2]
    else:
        df_merged = df_merged[0:len(df_list[0])-2]
    df_merged.loc[:, df_merged.columns != 'fed-interest-rate-decision US'] = df_merged.loc[:, df_merged.columns != 'fed-interest-rate-decision US'].interpolate().bfill().ffill()

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
    if df_merged_EuroZone[:1].isnull().sum(axis = 1).item()> 0 :
        df_merged_EuroZone = df_merged_EuroZone[1:len(df_list[0])-2]
    else:
        df_merged_EuroZone = df_merged_EuroZone[0:len(df_list[0])-2]
    df_merged_EuroZone.loc[:, df_merged_EuroZone.columns != 'fed-interest-rate-decision US'] = df_merged_EuroZone.loc[:, df_merged_EuroZone.columns != 'fed-interest-rate-decision US'].interpolate().bfill().ffill()

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


    data_frames_Merged=[df_merged,df_merged_EuroZone,dfr0[['boe-interest-rate-decision UK']],dfr1[['snb-interest-rate-decision Switzerland']],dfs[['EUR/USD']]]

    df = reduce(lambda  left,right: pd.merge(left,right,on='Date',
                                                how='left'), data_frames_Merged)


    df = df.sort_values(by="Date", ascending=[0])
    df=df [~df .index.duplicated(keep='first')]
    df['boe-interest-rate-decision UK'] = df['boe-interest-rate-decision UK'].bfill().ffill()
    df['snb-interest-rate-decision Switzerland'] = df['snb-interest-rate-decision Switzerland'].bfill().ffill()
    if df[:1].isnull().sum(axis = 1).item()> 0 :
        df = df[1:len(df_list[0])-2]
    else:
        df = df[0:len(df_list[0])-2]

    df['EUR/USD'] = dfs['EUR/USD'][1:].values

    df.to_csv('./data/macro_data_final2.csv',index=True)

    



n_periods=6

def stationary(feature):
  stats, p, lags, critical_values = kpss(df[feature], 'c',nlags='legacy')
  if p < 0.05 :
    return False
  else:
    return True



def create_model(feature,TEST_SIZE):
  X = df.drop(labels=[feature], axis=1)
  y = df[feature]
  corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.6
  to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

# Drop features 
  X.drop(to_drop, axis=1, inplace=True)

  df_=X.copy()
  df_[feature] = y
  train, test = df_.iloc[:-TEST_SIZE], df_.iloc[-TEST_SIZE:]
  variables=feature
  model = pm.auto_arima(train[feature], exogenous= train.drop(variables, axis=1),
                        m=12, seasonal=False,stationary=stationary(feature),
                      max_order=None, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)

  return model


def train_model(feature,TEST_SIZE,model):
  TEST_SIZE = 6
  X = df.drop(labels=[feature], axis=1)
  y = df[feature]
  corr_matrix = X.corr().abs()

  # Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Find features with correlation greater than 0.6
  to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

  # Drop features 
  X.drop(to_drop, axis=1, inplace=True)

  df_=X.copy()
  df_[feature] = y
  train, test = df_.iloc[:-TEST_SIZE], df_.iloc[-TEST_SIZE:]

  variables=feature
  model.fit(train[feature], exogenous= train.drop(variables, axis=1))
  forecast, confint=model.predict(n_periods=TEST_SIZE, return_conf_int=True, exogenous= test.drop(variables, axis=1))

  return forecast,confint




def plot_predictions(feature,TEST_SIZE,model):
  forecast,confint,test = train_model(feature,TEST_SIZE,model)
  forecast_dataframe = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
  return pd.concat([df[feature],forecast_dataframe],axis=1).plot(figsize=(14,10))




def plot_forecasts(feature,model_ppi):

  TEST_SIZE = 6
  X = df.drop(labels=[feature], axis=1)
  y = df[feature]
  corr_matrix = X.corr().abs()

  # Select upper triangle of correlation matrix
  upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

  # Find features with correlation greater than 0.6
  to_drop = [column for column in upper.columns if any(upper[column] > 0.6)]

  # Drop features 
  X.drop(to_drop, axis=1, inplace=True)

  df_=X.copy()
  df_[feature] = y
  train, test = df_.iloc[:-TEST_SIZE], df_.iloc[-TEST_SIZE:]
  variables=feature

    # Create traces
  trace0 = go.Scatter(
      x = df.index,
      y = df[feature].values,
      mode = 'lines',
      name = feature)

  x=df[feature].index.values
  for i in range(1,n_periods+1):
  
    x=np.append(x,(datetime.strptime(df[feature].index[-1], "%Y-%m") + relativedelta(months=+i)).strftime("%Y-%m"))

    
  fc, confint = model_ppi.predict(n_periods=n_periods, return_conf_int=True, exogenous= test.drop(variables, axis=1))
  index_of_fc = x[-n_periods:]


  trace2 =go.Scatter(x=index_of_fc, y=confint[:, 0], name='ARIMA model 95% Lower CI', mode = 'lines',
                    marker = dict(size=10, color='red'),opacity = 0.3)

  trace3 =go.Scatter(x=index_of_fc, y=confint[:, 1],name='ARIMA model 95% Upper CI', mode = 'lines',
                    marker = dict(size=10, color='red'),opacity = 0.3,fill='tonexty')

  trace4 =go.Scatter(x=index_of_fc, y=fc,name='ARIMA model mean projected values', mode = 'lines',
                    marker = dict(size=10, color='red'),opacity = 0.7)

  trace5 =go.Scatter(x=[index_of_fc[0],df.index[-1]], y=[fc[0],df[feature][-1]],name='ARIMA model mean projected values', mode = 'lines',
                    marker = dict(size=10, color='red'),opacity = 0.7)

  layout = go.Layout(title= 'Forecasting 95% CI - ' + feature,
      xaxis = dict(ticks='', nticks=43),
      yaxis = dict(nticks=20), legend=dict(x=0.1, y=1))



  data = [trace0,trace2,trace3,trace4,trace5]
  fig =go.Figure(data=data, layout=layout)
  return fig





    
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
# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID])
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.BOOTSTRAP]



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

app = Dash(__name__, external_stylesheets=external_stylesheets)

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
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

sidebar = html.Div(
    [
        html.H2("EUR/USD Exchange Rate Forecast", className="display-4"),
        html.Hr(),
        # html.P(
        #     "A simple sidebar layout with navigation links", className="lead"
        # ),
        dbc.Nav(
            [
                dbc.NavLink("Data", href="/", active="exact"),
                dbc.NavLink("Get New Data", href="/page-1", active="exact"),
                dbc.NavLink("Forecast", href="/page-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div(id="page-content", style=CONTENT_STYLE)

PAGE_SIZE = 10
app.layout = html.Div([
    html.Div([dcc.Location(id="url"), sidebar, content])
    
])
#---------------------------------------------------------------
# Creating random DataFrame
columns = list(df.columns)
full_data= df.copy()
full_data['Date'] = full_data.index

#---------------------------------------------------------------



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return html.Div([
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
                                                
                                                    
                                                    html.Div(dash_table.DataTable( 
                                                    id='datatable-paging',   
                                                    data=data_frame.to_dict('records'),
                                                    columns=[{'id': c, 'name': c} for c in data_frame.columns],
                                                        page_current=0,
                                                    page_size=PAGE_SIZE,
                                                    page_action='custom',

                                                    style_table={'height': '100%', 'overflowY': 'auto'}
                                                    
                                                    )   
                                                    ),

                                        html.Div(children=update_graph_correlation(),id="graph_correlation"),
                                        #html.Div(children=get_prediction(),id='stats_prediction')
                                    ])
                                            

        
    elif pathname == "/page-1":
        return html.Div([
                                    
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



    elif pathname == "/page-2":
        return html.Div([
                                    html.Div([
                                             dcc.Dropdown(
                                              id='my_dropdown_forecast',
                                              options=df.columns,
                                              value=['producer-price-index-yy US'],
                                              multi=False,
                                              clearable=False,
                                              style={"width": "100%"}
                                       ),
                                        html.Div([
                                            dcc.Graph(id='the_graph_forecast')
                                        ]),
                                        html.Div(id='container-model-summary',children='', style={'text-align':'center','whiteSpace': 'pre-wrap'}),

                                        html.Div([
                                            html.Img(id='the_graph_diag')
                                        ]),
                                        
                                     ])
                              ])


        



    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )




@app.callback(Output("my_dropdown2", "value"), Input("select-all", "n_clicks"),prevent_initial_call=True)
def select_all(n_clicks):
    return [option["value"] for option in listindicators]

@app.callback(Output("my_dropdown3", "value"), Input("select-all2", "n_clicks"),prevent_initial_call=True)
def select_all(n_clicks):
    return [option["value"] for option in listindicators_eu]

@app.callback(
    Output('datatable-paging', 'data'),
    Input('datatable-paging', "page_current"),
    Input('datatable-paging', "page_size"),
    [Input("btn", "n_clicks")]
    
    )

def update_table(page_current,page_size,n_clicks):
    if n_clicks is None:
        return data_frame.iloc[
            page_current*page_size:(page_current+ 1)*page_size
        ].to_dict('records')

    update_data()
    
    #data_frame=update_data()

    return data_frame.iloc[
            page_current*page_size:(page_current+ 1)*page_size
        ].to_dict('records')

    








@app.callback(    
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

@app.callback(    
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


    

@app.callback(    
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

                
@app.callback(    
    Output('container-button-basic-eur-usd', 'children'),
    Input('submit-val-eur-usd', 'n_clicks'),
    prevent_initial_call=True
)

def update_exchange_rate(n_clicks):
    

    dfs= web.get_data_yahoo('EURUSD=X','06/01/2007',interval='m')
    dfs = dfs.iloc[::1]
    dfs.index =  pd.to_datetime(dfs.index, format='%Y-%m').strftime("%Y-%m")
    dfs = dfs.rename(columns={'Close': 'EUR/USD'})   
    dfs=dfs.sort_values(by="Date", ascending=[0])

    
    with open('EURUSD=X.pkl', 'wb') as f:
        pickle.dump(dfs, f)


    #dfs.to_csv("./data/EURUSD=X.csv",index=True,mode='a')

    return 'Finished Updating data'


@app.callback(
    Output(component_id='the_graph_forecast', component_property='figure'),
    [Input(component_id='my_dropdown_forecast', component_property='value')]
)


def update_forecast(my_dropdown_forecast):
    fi =go.Figure()
    i = my_dropdown_forecast


    if i =='EUR/USD':
        with open('./models/arima_EUR_USD.pkl','rb') as f:
            mp = pickle.load(f)
            
            
    else:
        with open('./models/arima_'+i+'.pkl','rb') as f:
            mp = pickle.load(f)
            print('ok')
    
    fi=plot_forecasts(i,mp)

    return fi



@app.callback(
    Output(component_id='the_graph_diag', component_property='src'),
    [Input(component_id='my_dropdown_forecast', component_property='value')]
)


def update_forecast_(my_dropdown_forecast):
    import matplotlib.pyplot as plt
    i = my_dropdown_forecast


    if i =='EUR/USD':
        with open('./models/arima_EUR_USD.pkl','rb') as f:
            mp = pickle.load(f)
            
            
    else:
        with open('./models/arima_'+i+'.pkl','rb') as f:
            mp = pickle.load(f)
            print('ok')
    


    
    m=mp.plot_diagnostics(figsize=(14,10))

    buf = io.BytesIO()
    m.savefig(buf, format = "png") # save to the above file object
    data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements
    return "data:image/png;base64,{}".format(data)









@app.callback(
    Output(component_id='container-model-summary', component_property= 'children'),
    [Input(component_id='my_dropdown_forecast', component_property='value')]
)


def update_forecast_summary(my_dropdown_forecast):
    import matplotlib.pyplot as plt
    i = my_dropdown_forecast


    if i =='EUR/USD':
        with open('./models/arima_EUR_USD.pkl','rb') as f:
            mp = pickle.load(f)
            
            
    else:
        with open('./models/arima_'+i+'.pkl','rb') as f:
            mp = pickle.load(f)
            print('ok')
    


    return str(mp.summary())



@app.callback(
    Output(component_id='the_graph_corr', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)


def update_graph_correlation_(my_dropdown):    
     # Generate correlation matrix
    df = pd.read_csv("macro_data_final2.csv").set_index('Date')
    df = df.iloc[::-1]
    df1=pd.DataFrame()
    for label in my_dropdown:
        
        
        df1[label] = df[label]

        print(df1)
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









@app.callback(
    Output(component_id='the_graph', component_property='figure'),
    [Input(component_id='my_dropdown', component_property='value')]
)

def update_graph(my_dropdown):
    import plotly
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



if __name__ == "__main__":
    app.run_server(debug=False, port = 9001)
    
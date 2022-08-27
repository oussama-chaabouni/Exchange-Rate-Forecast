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
import pmdarima as pm
import dash_bootstrap_components as dbc


df = pd.read_csv("./data/macro_data_final2.csv").set_index('Date')
df = df.iloc[::-1]

data_frame=pd.read_csv("./data/macro_data_final2.csv")
data_frame = data_frame.iloc[::-1]
n_periods=6

def plot_forecasts(feature,model_ppi,data):
  df = data.iloc[::-1].copy()
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





def plot_forecasts_features(list,feature,model_ppi,data):
  df = data.iloc[::-1].copy()
  TEST_SIZE = 6
  X=df[list]

  y = df[feature]


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






  

def stationary(feature,data):
  df = data.iloc[::-1].copy()
  stats, p, lags, critical_values = kpss(df[feature], 'c',nlags='legacy')
  if p < 0.05 :
    return False
  else:
    return True



def create_model(feature,TEST_SIZE,data):
  df = data.iloc[::-1].copy()
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
                        m=12, seasonal=False,stationary=stationary(feature,data),
                      max_order=None, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)

  return model


def create_model_feat(list,feature,TEST_SIZE,data):
  df = data.iloc[::-1].copy()

  X=df[list]

  y = df[feature]


  df_=X.copy()
  df_[feature] = y
  train, test = df_.iloc[:-TEST_SIZE], df_.iloc[-TEST_SIZE:]
  variables=feature
  model = pm.auto_arima(train[feature], exogenous= train.drop(variables, axis=1),
                        m=12, seasonal=False,stationary=stationary(feature,data),
                      max_order=None, test='adf',error_action='ignore',  
                           suppress_warnings=True,
                      stepwise=True, trace=True)

  return model






def train_model(feature,TEST_SIZE,model,data):
  df = data.iloc[::-1].copy()
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




def plot_predictions(feature,TEST_SIZE,model,data):
  df = data.iloc[::-1].copy()
  forecast,confint,test = train_model(feature,TEST_SIZE,model)
  forecast_dataframe = pd.DataFrame(forecast,index = test.index,columns=['Prediction'])
  return pd.concat([df[feature],forecast_dataframe],axis=1).plot(figsize=(14,10))

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
                      href='/forecast/'+currencies,
                      id=str(currencies))


def generate_currency(currencies,country1,country2):
  return html.Div([
                                    html.Div([
                                             dcc.Dropdown(
                                              id='my_dropdown_forecast'+str(currencies),
                                              options=pd.read_csv("./data/macro_data_"+str(currencies)+'.csv').set_index('Date').columns,
                                              value=[''],
                                              multi=False,
                                              clearable=False,
                                              style={"width": "100%"}
                                       ),
                                        html.Div([
                                            dcc.Graph(id='the_graph_forecast'+str(currencies))
                                        ]),
                                        html.Div(id='container-model-summary'+str(currencies),children='', style={'text-align':'center','whiteSpace': 'pre-wrap'}),

                                        html.Div([
                                            html.Img(id='the_graph_diag'+str(currencies))
                                        ]),
                                        dcc.Dropdown(
                                                id='my_dropdown_features'+str(currencies),
                                                options=pd.read_csv("./data/macro_data_"+str(currencies)+'.csv').set_index('Date').columns,
                                                #df.loc[:, df.columns != 'EUR/USD'].columns
                                                value=[''],
                                                multi=True,
                                                clearable=False,
                                                style={"width": "50%",'textalign':"center"}
                                            ),
                                        ],style={"width": "100%",'textalign':"center"}),

                                         html.Div([
                                            dcc.Graph(id='the_graph_features_forecast'+str(currencies))
                                        ]),   
                                        html.Div([
                                            html.Img(id='the_graph_features'+str(currencies))
                                        ]),                                    
                                     
                              ])




sidebar_forecast = html.Div(
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
content_forecast = html.Div(id="page-content_forecast", style=CONTENT_STYLE)


layout = html.Div([
    html.Div([dcc.Location(id="url_forecast"), sidebar_forecast, content_forecast])
    
])


@callback(Output("page-content_forecast", "children"), [Input("url_forecast", "pathname")])
def render_page_content_(pathname="/forecast"):
    for i in df_currencies['currencies']:
        if pathname == "/forecast/" + i: 
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

                return generate_currency(i,'United_Kingdom','United_States')


            if i=='GBP_CHF':

                return generate_currency(i,'United_Kingdom','Switzerland')


            if i=='GBP_JPY':

                return generate_currency(i,'United_Kingdom','Japan')   


    # If the user tries to reach a different page, return a 404 message




def update_forecast(my_dropdown_forecast,path):
  currency = path.split('/forecast/', 1)[-1]

  data = pd.read_csv("./data/macro_data_"+currency+'.csv').set_index('Date')
  fi =go.Figure()
  i = my_dropdown_forecast
  model=create_model(i,6,data)
  m=model.plot_diagnostics(figsize=(14,10))

  # if i =='EUR/USD':
  #     with open('./models/arima_EUR_USD.pkl','rb') as f:
  #         mp = pickle.load(f)
            
            
  # else:
  #     with open('./models/arima_'+i+'.pkl','rb') as f:
  #         mp = pickle.load(f)
  #         print('ok')
    
  fi=plot_forecasts(i,model,data)

  buf = io.BytesIO()
  m.savefig(buf, format = "png") # save to the above file object
  data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements


  return fi,"data:image/png;base64,{}".format(data),str(model.summary())


for i in df_currencies['currencies']:
  callback(
      Output(component_id='the_graph_forecast'+i, component_property='figure'),
      Output(component_id='the_graph_diag'+i, component_property='src'),
      Output(component_id='container-model-summary'+i, component_property= 'children'),
      [Input(component_id='my_dropdown_forecast'+i, component_property='value')],
      State('url_forecast', 'pathname')
  )(update_forecast)



def create_model_features(my_dropdown_forecast,path):
  currency = path.split('/forecast/', 1)[-1]
  feature=currency.replace("_", "")
  dataf = pd.read_csv("./data/macro_data_"+currency+'.csv').set_index('Date')
  fi =go.Figure()
  model=create_model_feat(my_dropdown_forecast,feature,6,dataf)

  m=model.plot_diagnostics(figsize=(14,10))
  with open('./models/temp_models/model_EUR_USD','wb') as f:
      pickle.dump(m,f)

  buf = io.BytesIO()
  m.savefig(buf, format = "png") # save to the above file object
  data = base64.b64encode(buf.getbuffer()).decode("utf8") # encode to html elements

  fi=plot_forecasts_features(my_dropdown_forecast,feature,model,dataf)

  return "data:image/png;base64,{}".format(data),fi



for i in df_currencies['currencies']:

  callback(
      Output(component_id='the_graph_features'+i, component_property='src'),
      Output(component_id='the_graph_features_forecast'+i, component_property='figure'),
      [Input(component_id='my_dropdown_features'+i, component_property='value')],
      State('url_forecast', 'pathname'),

      prevent_initial_call=True
  )(create_model_features)




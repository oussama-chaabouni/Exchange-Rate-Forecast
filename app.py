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
from pages import home, scrape_data,forecast


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




#print(len(indicators_US)+len(indicators_EU))




    














    

# you need to include __name__ in your Dash constructor if
# you plan to use a custom CSS or JavaScript in your Dash apps
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.GRID])
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css',dbc.themes.BOOTSTRAP]





app = Dash(__name__, external_stylesheets=external_stylesheets,suppress_callback_exceptions=True)

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
                dbc.NavLink("Data", href="/home", active="exact"),
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


app.layout = html.Div([
    html.Div([dcc.Location(id="url", refresh=False), sidebar, content])
    
])

app.validation_layout = html.Div([sidebar, home.layout])
#---------------------------------------------------------------
# Creating random DataFrame

#---------------------------------------------------------------

data = {'currencies': ['EUR_USD', 'EUR_GBP', 'EUR_AUD', 'EUR_CAD','EUR_JPY','EUR_CHF','USD_JPY','USD_CAD','USD_CHF','GBP_USD','GBP_CHF','GBP_JPY']}
df_currencies = pd.DataFrame(data)

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):

    for i in df_currencies['currencies']:
        if pathname == "/home/"+i:
            return home.layout    

    for i in df_currencies['currencies']:
        if pathname == "/page-1/" + i: 
            
            return scrape_data.layout

        

                                            

    if pathname == "/page-1":
        return scrape_data.layout

    if pathname == "/home":
        return home.layout

    if pathname == "/page-2":
        return forecast.layout


    




if __name__ == "__main__":
    app.run_server(debug=False, port = 9001)
    
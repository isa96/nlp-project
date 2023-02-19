
import numpy as np
import pandas as pd
import seaborn as sns

# data process #
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer()
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Model preparation #
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import pickle

# dash #
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from dash.dependencies import Input, Output, State

#to make Dataframe
def generate_table(dataframe, page_size=10):
    return dash_table.DataTable(
        id='dataTable',
        columns=[{"name": i,"id": i} for i in dataframe.columns],
        data=dataframe.to_dict('records'),
        page_action="native",
        page_current=0,
        page_size=page_size,
        style_cell={
        'overflow': 'hidden',
        'textOverflow': 'ellipsis',
        'maxWidth': 0,}
    )

#load data
df = pd.read_csv('df_sample.csv')

#StyleSheet
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Predict #
#...1)  stopwords
stop = stopwords.words('english')
stop.remove('not')
new_stopwords = []
for item in stop:
    new_words = [char for char in item if char not in string.punctuation]
    new_words = ''.join(new_words)
    new_stopwords.append(new_words)
new_stopwords.append("im")
new_stopwords.append("i'm")
new_stopwords[:10]

#...2)  clothes
clothes = ['top', 'dress', 'blouse', 'shirt', 'skirt', 'jeans', 'jean', 'jumpsuit','color','camisole']

#to Process Text - part of Model
def clean_process(text):
    #make lowercase
    clean_text = text.lower()

    #remove punctuation and numbers#
    clean_text = [char for char in clean_text if char not in string.punctuation]
    clean_text = [char for char in clean_text if char not in string.digits]
    clean_text = ''.join(clean_text)
    
    #remove spasi kelebihan di depan/akhir review#
    clean_text = clean_text.strip()
    
    #Spelling Correction#
    clean_text = TextBlob(clean_text).correct()

    #remove stopwords#
    clean_text = [word for word in clean_text.split(' ') if word not in stopwords.words('english')]
    clean_text = [word for word in clean_text if word not in new_stopwords]
    
    #make it whole again#
    clean_text = ' '.join(clean_text)
    
    #stringnya di-tokenize dulu menjadi token berupa kata (word token)#
    clean_text = clean_text.split()
    
    #setiap tokennya di lemmatize
    new_string=[]
    for word in clean_text:
        x_word = lemmatizer.lemmatize(word)
        new_string.append(x_word)
        
    return new_string


## VISUALISATION ##

#FIG: Age Distribution
x = [df['Age']]
fig = ff.create_distplot(x, ['Consumer Age Distribution'])
fig.update_layout(
    title="Consumer's Age Distribution Plot")

#FIG 1: Length Distribution
rl = [df['Review Length']] 
fig_1 = ff.create_distplot(rl, ['Review Length Distribution'])
fig_1.update_layout(
    title="Review Length Distribution Plot")

#FIG 2: Percentage of Recommend item based on Department Name
fig_2 = px.pie(df, values='Recommend', names='Department Name')
fig_2.update_layout(
    title="Percentage of Recommend Item based on Department Name")

#FIG 3: Correlation scatterplot department name and rating
fig_3 = px.scatter(df, x='Department Name', y='Review Length', facet_col="Rating")
fig_3.update_layout(
    title="Scatterplot of Department Name and Review Length")

#FIG 4:
fig_4 = px.scatter(df, y="Review Length", x="Rating", facet_col="Recommend")
fig_4.update_layout(
    title="Scatterplot of Review Length, Rating, and Recommend")

## Apply app-dash
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

## Insert Body Dash
app.layout = html.Div(
    children=[
        html.Br(),
        html.H2('Dash Final Project'),
        html.Div(children=''''''),
        html.Br(),

        #TAB UTAMA
        dcc.Tabs(children=[
            #TAB 1
            dcc.Tab(
                value='Tab1',
                label='Dataset',
                children=[
                    #ISI TAB 1.1
                        html.Div(children=[
                            #ROW 1
                            html.Div([
                                html.P('Recommend'),
                                dcc.Dropdown(value='',
                                            id='filter-recommend',
                                            options=[{'label': 'No','value': 0}, 
                                                    {'label': 'Yes','value': 1},
                                                    {'label': 'None','value': ''}])
                                    ],className='col-3'),
                                
                            html.Div([
                                html.P('Rating'),
                                dcc.Dropdown(value='',
                                            id='filter-rating',
                                            options=[{'label': '1','value': 1}, 
                                            {'label': '2','value': 2},
                                            {'label': '3','value': 3},
                                            {'label': '4','value': 4},
                                            {'label': '5','value': 5},
                                            {'label': 'None','value': ''}])
                                    ],className='col-3'),
                                
                            html.Div([
                                html.P('Department Name'),
                                dcc.Dropdown(value='',
                                            id='filter-department',
                                            options=[{'label': 'Intimate','value': 'Intimate'}, 
                                                    {'label': 'Dresses','value': 'Dresses'},                     
                                                    {'label': 'Bottoms','value': 'Bottoms'},
                                                    {'label': 'Tops','value': 'Tops'},
                                                    {'label': 'Jackets','value': 'Jackets'},
                                                    {'label': 'Trend','value': 'Trend'},
                                                    {'label': 'None','value': ''}])
                                    ],className='col-3'),
                            
                            ],className='row'),
                            
                            #GANTI ROW
                            #ROW 2

                            html.Br(),
                            html.Div(children =[
                                html.Button('search',id = 'filter')
                                ],className = 'row col-4'),

                            html.Br(), 
                            #ROW 4
                            html.Div(id='div-table',
                                     children=[generate_table(df)])
                ]),

            #TAB 2
            dcc.Tab(
                value='Tab2',
                label='Visualisation',
                children=[
                    #ISI TAB 2.1
                    html.Div([
                        dcc.Graph(figure= fig)
                        ]),

                    html.Br(),
                    html.Div([
                        dcc.Graph(figure= fig_1)
                        ]),

                    html.Br(),
                    html.Div([
                        dcc.Graph(figure= fig_2)
                        ]),
                    
                    html.Br(),
                    html.Div([
                        dcc.Graph(figure= fig_4)
                        ]),

                    html.Br(),
                    html.Div([
                        dcc.Graph(figure= fig_3)
                        ]),
               ]),

            #TAB 3
            dcc.Tab(
                value='Tab3',
                label='Prediction Model',
                children=[
                    
                    html.Div([
                    html.P('Reviews: '),
                    dcc.Input(value='',
                            id='reviews',
                            type='text',
                            placeholder='Enter your review here...',
                            )]
                    , className= 'col-3'),
                    
                    html.Br(),
                    html.Div(children =[
                    html.Button('Predict', id ='prediction')
                    ]),

                    html.Br(),
                    html.Div([
                    html.P('Hasil Prediksi: '),
                    html.Div(id='result')]
                    )
                    
                    ])
            
            ],
            ## Tabs Content Style
            content_style={
                'fontFamily': 'Arial',
                'borderBottom': '1px solid #d6d6d6',
                'borderLeft': '1px solid #d6d6d6',
                'borderRight': '1px solid #d6d6d6',
                'padding': '44px'
             }
            
            )
        
        ],
    
    #Div Paling luar Style
    style={
        'maxWidth': '1200px',
        'margin': '0 auto'
        })

## APP CALLBACK
@app.callback(
    Output(component_id = 'div-table', component_property = 'children'),
    [Input(component_id = 'filter', component_property = 'n_clicks')],
    [State(component_id = 'filter-recommend', component_property = 'value'), 
    State(component_id = 'filter-rating', component_property = 'value'),
    State(component_id = 'filter-department', component_property = 'value'),
    ])

## FUNCTION UPDATE DF #
def update_table(n_clicks, recommend, rating, department):
    df = pd.read_csv('df_sample.csv')
    if recommend != '':
        df = df[df['Recommend'] == recommend]
    if rating != '':
        df = df[df['Rating'] == rating]
    if department != '':
        df = df[df['Department Name'] == department]
    children = [generate_table(df, page_size = 10)]
    return children

#APP PREDICT#
@app.callback(
    Output(component_id = 'result', component_property = 'children'),
    [Input(component_id = 'prediction', component_property = 'n_clicks')],
    [State(component_id = 'reviews', component_property = 'value')]
    )

#FUNCTION UNTUK PREDICT#
def predict_input(n_clicks, review):
    if (str(review) == '') or (str(review)== 'None'):
        return ''
    else:
        isian = {0:'Not Recommended', 1:'Recommended'}
        model = pickle.load(open('pipeline.sav', 'rb'))
            
        d = {'col': str(review)}
        df_res = pd.DataFrame(data=d, index=[0])
            
        result = model.predict(df_res['col'])[0]
        prob_num = model.predict_proba(df_res['col'])
        prob = (prob_num[0][result])*100

        return f"{isian[result]} dengan probability {round(prob, 2)} %"

#Run APP/dash
if __name__ == '__main__':
    app.run_server(debug=False)

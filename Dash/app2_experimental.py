# Importamos las librerias mínimas necesarias
from pickletools import markobject
from pydoc import classname
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import logging
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
import dash_daq as daq
from plotly.graph_objects import Layout
import dash_daq as daq
import pickle as pk


# Para este Dash, vamos a seleccionar un fichero de datos y realizar un dashboard descriptivo
# sobre un conjunto de datos

df = pd.read_csv('../Datos/heart_2020_cleaned.csv')
df['Race'] = df['Race'].map({'American Indian/Alaskan Native': 'Native American',
                             'White':'White', 'Black':'Black','Asian':'Asian','Hispanic':'Hispanic','Other':'Other'})

# Crear opciones para las variables categóricas

cols_categoricas = ["HeartDisease","Smoking","AlcoholDrinking", "Stroke", 
"DiffWalking", "Sex", "AgeCategory","Race", "Diabetic", "PhysicalActivity",  "GenHealth", "Asthma","KidneyDisease", "SkinCancer"]


dropdown_categoricas = []
for col in cols_categoricas:
    dropdown_categoricas.append({'value': col, 'label': col})


cols_numericas = ["BMI","MentalHealth","PhysicalHealth", "SleepTime"]

dropdown_numericas = []
for col in cols_numericas:
    dropdown_numericas.append({'value': col, 'label': col})

app = dash.Dash()

#app.config.suppress_callback_exceptions = True

logging.getLogger('werkzeug').setLevel(logging.INFO)

app.layout = html.Div(

    children= [

        # 1. TÍTULO

        html.Div(
            children = [
                html.H1(
                    children = [
                        "HEART DISEASE ANALYSIS"
                        ],
                    id = "titulo",
                    style = {
                        "text-align": "left",
                        "margin-bottom": "20px",
                        "height": "50px",
                        "margin-top":"2%",
                        "padding-right": "5%",
                        "padding-left": "5%",
                        "font-family": "verdana",
                        "background-color":"rgb(255,255,255)"
                    }
                ),
            ],
            id = "div_titulo",
            style = {
                "background-color":"rgb(255,255,255)",
                "border-bottom":"1px solid rgba(204,202,202,0.9)",
                "color": "rgb(67,67,67)"
            }
        ),


        ## 2. PRIMER DIV DE CONTENIDO: DISTRIBUCIÓN CATEGÓRICAS Y PROPORCIÓN TARGET EN CATEGORICAS

        html.Div(
            children=[

                html.Div(
                    children=[
                        ## 2.0 Overview

                        html.Div(
                            children=[
                                html.H3(
                                    children = [
                                        "BASIC OVERVIEW"
                                    ],
                                    id = "titulo_overview",
                                    style = {
                                        "text-align": "left",
                                        "margin": "2.5%",
                                        "margin-bottom": "4%",
                                        "text-align": "left",
                                        "font-family": "verdana",
                                        "font-weight": "600",
                                        "color": "rgb(67,67,67)"
                                    }
                                ),

                                html.P(
                                    children = [
                                        html.P(children=["OBSERVATIONS |"],
                                            style={
                                                "display":"inline-block",
                                                "margin":"0px", "padding":"0px"
                                            }
                                        ),
                                        html.P(children=["{}".format(len(df.index))],
                                            style={
                                                "display":"inline-block",
                                                #"color": "rgb(127,127,127)",
                                                "margin":"0px", "padding":"0px",
                                                "margin-left": "1%",
                                                "font-weight": "500",
                                            }
                                        )
                                    ],
                                    style={
                                        "text-align": "left",
                                        "margin": "2.5%",
                                        "margin-bottom": "2%",
                                        "text-align": "left",
                                        "font-family": "verdana",
                                        "font-weight": "600",
                                        "color": "rgb(77,77,77)",
                                        "margin-left" : "4%"
                                    }
                                ),
                                html.P(
                                    children = [
                                        html.P(children=["NUMBER OF VARIABLES |"],
                                            style={
                                                "display":"inline-block",
                                                "margin":"0px", "padding":"0px"
                                            }
                                        ),
                                        html.P(children=["{}".format(len(df.columns))],
                                            style={
                                                "display":"inline-block",
                                                #"color": "rgb(127,127,127)",
                                                "margin":"0px", "padding":"0px",
                                                "margin-left": "1%",
                                                "font-weight": "500",
                                            }
                                        )
                                    ],
                                    style={
                                        "text-align": "left",
                                        "margin": "2.5%",
                                        "margin-bottom": "2%",
                                        "text-align": "left",
                                        "font-family": "verdana",
                                        "font-weight": "600",
                                        "color": "rgb(77,77,77)",
                                        "margin-left" : "4%"
                                    }
                                ),
                                html.P(
                                    children = [
                                        html.P(children=["HEART DISEASE DISTRIBUTION |"],
                                            style={
                                                "display":"inline-block",
                                                "margin":"0px", "padding":"0px"
                                            }
                                        ),
                                        html.P(
                                            children=["NO {}%  -  YES {}%".format(
                                                round(df[df["HeartDisease"] == 'No'].count()[0]/len(df.index)*100,2),
                                                round(df[df["HeartDisease"] == 'Yes'].count()[0]/len(df.index)*100,2)
                                            )
                                        ],
                                            style={
                                                "display":"inline-block",
                                                #"color": "rgb(127,127,127)",
                                                "margin":"0px", "padding":"0px",
                                                "margin-left": "1%",
                                                "font-weight": "500",
                                            }
                                        )
                                    ],
                                    style={
                                        "text-align": "left",
                                        "margin": "2.5%",
                                        "margin-bottom": "6%",
                                        "text-align": "left",
                                        "font-family": "verdana",
                                        "font-weight": "600",
                                        "color": "rgb(77,77,77)",
                                        "margin-left":"4%"
                                    }
                                ),
                            ],
                            id="div_general_overview",
                            style={
                                "background-color":"rgb(255,255,255)",
                                "border":"1px solid rgb(204,202,202)",
                                "margin":"1.5%",
                                "margin-left":"2%",
                                "margin-right":"0.5%",
                                "margin-top":"0px",
                                "display": "inline-block",
                                "border-radius" : "2px",
                                "height":"195px",
                                "width":"95%",
                                "padding-bottom":"0.4%"
                            }
                        ),

                        ## 2.1 Distribucion variables categoricas

                        html.Div(
                            children = [

                                html.H3(
                                    children = [
                                        "VARIABLES CATEGÓRICAS: DISTRIBUCIÓN"
                                    ],
                                    id = "distribucion_categoricas",
                                    style = {
                                        "text-align": "left",
                                        "margin": "2.5%",
                                        "margin-bottom": "4%",
                                        "text-align": "left",
                                        "font-family": "verdana",
                                        "font-weight": "600",
                                        "color": "rgb(67,67,67)"
                                    }
                                ),

                                html.Div(
                                    children = [
                                        dcc.Dropdown(
                                            options = dropdown_categoricas,
                                            value="Race",
                                            placeholder = "Selecciona una variable categorica",
                                            id = "dropdown_categoricas",
                                            style = {
                                                "display": "block",
                                                "width": "300px",
                                                "margin-left": "10px",
                                                'font-size' : '85%',
                                                'font-family':'verdana'
                                            }
                                        ),
                                        dcc.Graph(
                                            id = "dropdown_piechart_distribucion_categoricas",
                                            style = {
                                                "display": "none",
                                            }
                                        )
                                    ]
                                ),

                            ],
                            id = "div_distribucion_categoricas",
                            style = {
                                "background-color":"rgb(255,255,255)",
                                "border":"1px solid rgb(204,202,202)",
                                "margin":"1.5%",
                                "margin-left":"2%",
                                "margin-right":"0.5%",
                                "display": "inline-block",
                                "border-radius" : "2px",
                                "width":"95%",
                                "height":"36.6em"
                            }
                        ),
                    ],
                    style={
                        "display":"inline-block",
                        "margin":"1%",
                        "margin-top":"15px",
                        "padding":"0px",
                        "margin-left":"2%",
                        "margin-right":"0px",
                        "height":"114vh",
                        "width":"50%",
                        "padding-bottom": "500em",
                        "margin-bottom": "-500em",
                    },
                    id="div_contenedor_overview_y_distribucion_categoricas"
                ),


                ## 2.2 Porcentaje Heart Disease en Variables Categoricas

                html.Div(
                    children=[

                        html.H3(
                            children = [
                                "VARIABLES CATEGÓRICAS: PORCENTAJE DE HEART DISEASE"
                            ],
                            id = "titulo_distribucion_target_en_categoricas",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)"
                            }
                        ),
                        html.Div(
                            children = [

                                dcc.Dropdown(
                                            options = dropdown_categoricas,
                                            value="Race",
                                            placeholder = "Selecciona una variable categorica",
                                            id = "dropdown_porcentaje_heart_disease_variables_categoricas",
                                            style = {
                                                "display": "block",
                                                "width": "300px",
                                                "margin-left": "10px",
                                                'font-size' : '85%',
                                                'font-family':'verdana'
                                            }
                                        ),

                                html.Div(
                                    children = [

                                        html.Div(
                                            children = [
                                                dcc.Graph(
                                                    id = "hist_porcentaje_heart_disease_categoricas",  
                                                    style={'display': 'inline-block'}
                                                ),

                                            ], style={
                                                #"display": "inline-block",
                                                "margin-left":"0px",
                                                "margin-right":"0px",
                                                "width":"650px",
                                                "height":"22em"
                                                }

                                        ),

                                        html.Div(
                                            children = [
                                                dcc.Graph(
                                                    id = "hist_total_heart_disease_categoricas", 
                                                    style={'display': 'inline-block'}
                                                )
                                            ], style={
                                                #"display": "inline-block",
                                                "margin-left":"0px",
                                                "margin-right":"0px",
                                                "width":"100px",
                                                "height":"21.5em"
                                            }
                                        )

                                    ],
                                ),
                            ],
                        ),
                        
                    ],
                    id = "div_proporcion_target_categorias",
                    style={
                        "background-color":"rgb(255,255,255)",
                        "border":"1px solid rgb(204,202,202)",
                        "margin":"1%",
                        "margin-top":"16px",
                        "margin-left":"0.25%",
                        "margin-right":"0%",
                        "padding-left":"0.5%",
                        "display": "inline-block",
                        "border-radius" : "2px",
                        "width":"40%",
                        "height":"50.4em",
                    }
                ),
            ],
            id="div_contenedor_de_distribucion_categoricas_y_proporcion_target_categorias_por_separado",
            style = {
                "background-color":"transparent",
                "margin":"0px",
                "display": "inline-block",
                "width":"100%",
                #"height":"114vh"
                "height":"820px"
            }
        ),

        
        ## 3. Distribucion variables numericas

        html.H2(
            children = [
                "Variables Numéricas: Distribución"
            ],
            id = "distribucion_numericas",
            style = {
                "text-align": "center",
                "margin-bottom": "20px",
                "height": "50px",
                "margin-top":"40px",
                "text-align": "left"

            }
        ),
        html.Div(
            children = [
                dcc.Dropdown(
                            options = dropdown_numericas,
                            value="BMI",
                            placeholder = "Selecciona una variable numerica",
                            id = "dropdown_numericas",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.Graph(
                            id = "dropdown_histograma_distribucion_numericas",
                            style = {
                                "display": "none"
                            }
                        )
            ]

        ),

        ## SLIDER TAMAÑO BINS HISTOGRAMA
        html.Div(
            children=[
                html.P(
                    children = [
                         "Seleccione el tamaño de los bins del histograma: "
                    ],
                    id ="descripcion_slider",
                    style = {
                        "text-align": "left",
                        "margin-bottom": "20px"
                    }
                ),

                dcc.Slider(id="slider_histograma_numericas", min=1, max=6, step=1, value=2, marks={'1': '1', '2': '2','3':'3','4':'4','5':'5','6':'6'})
            ],
            style={
                "margin-left": "25%",
                "margin-right": "25%",
                "text-align": "center"
            }
        ),

        

        ## 5. Comparacion Heart Disease según variables categóricas y numéricas
        html.H2(
            children = [
                "Heart Disease según variables numéricas y categóricas"
            ],
            id = "comparacion_heart_disease_categorica_numerica",
            style = {
                "text-align": "center",
                "margin-bottom": "20px",
                "height": "50px",
                "margin-top":"40px",
                "text-align": "left"

            }
        ),
        html.Div(
            children = [
                dcc.Dropdown(
                            options = dropdown_categoricas,
                            value="Race",
                            placeholder = "Selecciona una variable categorica",
                            id = "dropdown_cat_comparacion_categorica_numerica",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.Dropdown(
                            options = dropdown_numericas,
                            value="BMI",
                            placeholder = "Selecciona una variable numerica",
                            id = "dropdown_num_comparacion_categorica_numerica",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.RadioItems(
                    id="radio_item_box_violin_selector",
                    options=[
                        {'label': '                     Box Plot                                   ', 'value': 'Box Plot'},
                        {'label': '                     Violin Plot', 'value': 'Violin Plot'}
                    ],
                    value='Box Plot', style={"margin-top":"25px", "margin-left":"10px"}
                ),
                
                dcc.Graph(
                            id = "comparacion_boxplot_heart_disease_segun_var_cat_y_num",
                            style = {
                                "display": "none",
                            }
                        ),
                
                dcc.Graph(
                            id = "comparacion_violinplot_heart_disease_segun_var_cat_y_num",
                            style = {
                                "display": "none",
                            }
                        )
            ]

        ),

        ## 6. Correlación entre variables numéricas
        html.H2(
            children = [
                "Correlación entre variables numéricas"
            ],
            id = "titulo_correlacion_entre_variables_numericas",
            style = {
                "text-align": "center",
                "margin-bottom": "20px",
                "height": "50px",
                "margin-top":"40px",
                "text-align": "left"

            }
        ),
        html.Div(
            children = [
                dcc.Dropdown(
                            options = dropdown_numericas,
                            value="SleepTime",
                            placeholder = "Selecciona una variable numerica",
                            id = "dropdown_1_scatter_correlacion_numerica_numerica",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.Dropdown(
                            options = dropdown_numericas,
                            value="BMI",
                            placeholder = "Selecciona una variable numerica",
                            id = "dropdown_2_scatter_correlacion_numerica_numerica",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.Graph(
                            id = "scatter_correlacion_numerica_numerica",
                            style = {
                                "display": "none"
                            }
                        )

            ]
        )
        



    ], style={
          "font-family": "verdana",
          "background-color":"rgb(244,242,242)",
          "margin" : "0px",
          "padding" : "0px",
          "border" : "0px"
        },
)


# 1.A. CALLBACK DE PIE CHARTS PARA VER DISTRIBUCION DE CATEGORICAS (Primer grafico que se ve en el dash)
@app.callback(
    Output("dropdown_piechart_distribucion_categoricas", "figure"),
    Output("dropdown_piechart_distribucion_categoricas", "style"),
    Input("dropdown_categoricas", "value")
)

def pie_chart_distribucion_categoricas_dropdown(dropdown_categoricas):

    diccionario_columnas_categoricas = {
        "HeartDisease": "Heart Disease",
        "Smoking": "Smoking",
        "AlcoholDrinking": "Alcohol Drinking",
        "Stroke": "Stroke",
        "DiffWalking": "DiffWalking",
        "Sex":"Sex",
        "AgeCategory": "Age Category",
        "Race": "Race",
        "Diabetic": "Diabetic",
        "PhysicalActivity": "Physical Activity",
        "GenHealth": "GenHealth",
        "Asthma": "Asthma",
        "KidneyDisease": "Kidney Disease",
        "SkinCancer": "Skin Cancer"
    
    }

    if dropdown_categoricas:
        
        proporcion = df[dropdown_categoricas].value_counts()/df[dropdown_categoricas].count()
        data = [
            go.Pie(
                labels=df[dropdown_categoricas].unique(), 
                values=proporcion,
                textinfo='label+percent',
                insidetextorientation='radial', 
                marker_colors = ["lightblue", "mediumseagreen", "gold", "darkorange", "indigo"],
                rotation = -55,
                sort = False)
        ]

        layout = go.Layout(title = "Distribucion de " + diccionario_columnas_categoricas[dropdown_categoricas])
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(width=600)
        
        return (fig,{"display":"inline-block","padding":"2%",})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})


# 2.A. CALLBACK DE HISTOGRAMAS PARA VER DISTRIBUCION DE NUMERICAS (Segundo grafico que se ve en el dash)
@app.callback(
    Output("dropdown_histograma_distribucion_numericas", "figure"),
    Output("dropdown_histograma_distribucion_numericas", "style"),
    Input("dropdown_numericas", "value"),
    Input("slider_histograma_numericas", "value")
)
        
def histograma_distribucion_numericas_dropdown(dropdown_numericas,slider_histograma_numericas):
    
    diccionario_variables_numericas = {
        "BMI":"Body Mass Index",
        "MentalHealth":"Mental Health",
        "PhysicalHealth":"Physical Health",
        "SleepTime":"Sleep Time"
    }

    if dropdown_numericas:
        
        data = [
        go.Histogram(
            x = df[dropdown_numericas],
            marker_color = "firebrick",
            xbins=dict(
                start= 0,
                size=slider_histograma_numericas
            ),
            opacity=0.6,
        )]
        layout = go.Layout(title = "Distribución de " + diccionario_variables_numericas[dropdown_numericas], 
                    xaxis_title = diccionario_variables_numericas[dropdown_numericas], yaxis_title = "Frecuencia",
                    barmode = "overlay", bargap = 0.1)
        
        fig = go.Figure(data = data, layout = layout)
        
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})


# 3.A. CALLBACK DE HISTOGRAMAS PARA VER PORCENTAJE HEART DISEASE EN CATEGORICAS (Tercer grafico que se ve en el dash)

@app.callback(
    Output("hist_porcentaje_heart_disease_categoricas", "figure"),
    Output("hist_porcentaje_heart_disease_categoricas", "style"),
    Input("dropdown_porcentaje_heart_disease_variables_categoricas", "value")
) 

def hist_porcentaje_heart_disease_categoricas_dropdown(dropdown_porcentaje_heart_disease_variables_categoricas):

    diccionario_columnas_categoricas = {
        "HeartDisease": "Heart Disease",
        "Smoking": "Smoking",
        "AlcoholDrinking": "Alcohol Drinking",
        "Stroke": "Stroke",
        "DiffWalking": "DiffWalking",
        "Sex":"Sex",
        "AgeCategory": "Age Category",
        "Race": "Race",
        "Diabetic": "Diabetic",
        "PhysicalActivity": "Physical Activity",
        "GenHealth": "GenHealth",
        "Asthma": "Asthma",
        "KidneyDisease": "Kidney Disease",
        "SkinCancer": "Skin Cancer"
    
    }
    

    if dropdown_porcentaje_heart_disease_variables_categoricas:
        
        totals_per_col = df[dropdown_porcentaje_heart_disease_variables_categoricas].value_counts()
        unique_values = df[dropdown_porcentaje_heart_disease_variables_categoricas].unique()
    
    
        y = [df[(df[dropdown_porcentaje_heart_disease_variables_categoricas]==item) & (df['HeartDisease']=='Yes')][dropdown_porcentaje_heart_disease_variables_categoricas].count()/totals_per_col[item] for item in unique_values]
        trace = go.Bar(x = df[dropdown_porcentaje_heart_disease_variables_categoricas].unique(),
                    y = y,
                    name = "HeartDisease",
                    marker_color = "firebrick",  # firebrick mediumseagreen darkcian
                    text= ["{0}%".format(round(value*100,1)) for value in y],
                    textposition="auto",
                    opacity=0.8,
                    textangle=0,
                    textfont_size = 20,
                    textfont_color= "white",
                   )

        data = [trace]
        layout = go.Layout(
            title = "% Heart Disease by " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], 
            xaxis_title = diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], 
            yaxis_title = "% of heart disease")
        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(width=500)
        fig.update_layout(height=350)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )

        return (fig,{"display":"inline-block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})

## SEGUNDA GRAFICA TOTAL HEART DISEASE SEGUN CATEGORICAL
@app.callback(
    Output("hist_total_heart_disease_categoricas", "figure"),
    Output("hist_total_heart_disease_categoricas", "style"),
    Input("dropdown_porcentaje_heart_disease_variables_categoricas", "value")
) 

def hist_porcentaje_heart_disease_categoricas_dropdown(dropdown_porcentaje_heart_disease_variables_categoricas):

    diccionario_columnas_categoricas = {
        "HeartDisease": "Heart Disease",
        "Smoking": "Smoking",
        "AlcoholDrinking": "Alcohol Drinking",
        "Stroke": "Stroke",
        "DiffWalking": "DiffWalking",
        "Sex":"Sex",
        "AgeCategory": "Age Category",
        "Race": "Race",
        "Diabetic": "Diabetic",
        "PhysicalActivity": "Physical Activity",
        "GenHealth": "GenHealth",
        "Asthma": "Asthma",
        "KidneyDisease": "Kidney Disease",
        "SkinCancer": "Skin Cancer"
    
    }
    

    if dropdown_porcentaje_heart_disease_variables_categoricas:
        
        data = [
            go.Histogram(
                x = df[df["HeartDisease"] == 'Yes'][dropdown_porcentaje_heart_disease_variables_categoricas],
                marker_color = "red",  # darkorange
                opacity = 0.75,
                name = "Heart Disease"
            ),
            go.Histogram(
                x = df[df["HeartDisease"] == 'No'][dropdown_porcentaje_heart_disease_variables_categoricas],
                marker_color = "blue",  # mediumseagreen
                opacity = 0.75,
                name = "No Heart Disease"
            )
        ]
        layout = go.Layout(title = "Heart Disease según " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], xaxis_title =diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], yaxis_title = "Count")  # yaxis_title = "Heart Disease según " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas]

        fig = go.Figure(data = data, layout = layout)
        fig.update_layout(width=500)
        fig.update_layout(height=330)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        
        return (fig,{"display":"inline-block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})


# 4.A. CALLBACK DE BOXPLOT PARA VER COMPARACION HEART DISEASE SEGUN VARIABLE CATEGORICA Y NUMERICA (Cuarto grafico que se ve en el dash)

@app.callback(
    Output("comparacion_boxplot_heart_disease_segun_var_cat_y_num", "figure"),
    Output("comparacion_boxplot_heart_disease_segun_var_cat_y_num", "style"),
    Input("dropdown_cat_comparacion_categorica_numerica", "value"),
    Input("dropdown_num_comparacion_categorica_numerica", "value"),
    Input("radio_item_box_violin_selector", "value")
) 


def boxplot_comparacion_heart_disease_categorica_y_numerica_dropdown(dropdown_cat_comparacion_categorica_numerica,dropdown_num_comparacion_categorica_numerica,radio_item_box_violin_selector):
    
    diccionario_columnas_categoricas = {
        "HeartDisease": "Heart Disease",
        "Smoking": "Smoking",
        "AlcoholDrinking": "Alcohol Drinking",
        "Stroke": "Stroke",
        "DiffWalking": "DiffWalking",
        "Sex":"Sex",
        "AgeCategory": "Age Category",
        "Race": "Race",
        "Diabetic": "Diabetic",
        "PhysicalActivity": "Physical Activity",
        "GenHealth": "GenHealth",
        "Asthma": "Asthma",
        "KidneyDisease": "Kidney Disease",
        "SkinCancer": "Skin Cancer"
    
    }
    diccionario_variables_numericas = {
        "BMI":"Body Mass Index",
        "MentalHealth":"Mental Health",
        "PhysicalHealth":"Physical Health",
        "SleepTime":"Sleep Time"
    }


    if dropdown_cat_comparacion_categorica_numerica and dropdown_num_comparacion_categorica_numerica and (radio_item_box_violin_selector=="Box Plot"):
        
        fig = px.box(df, y=dropdown_cat_comparacion_categorica_numerica, x=dropdown_num_comparacion_categorica_numerica, color="HeartDisease",title="Heart Disease distribution by" + diccionario_variables_numericas[dropdown_num_comparacion_categorica_numerica] + "and" + diccionario_columnas_categoricas[dropdown_cat_comparacion_categorica_numerica])
        fig.update_traces(quartilemethod="exclusive") 

        return (fig,{"display":"block", "height":"750px"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})



## VIOLIN PLOT
@app.callback(
    Output("comparacion_violinplot_heart_disease_segun_var_cat_y_num", "figure"),
    Output("comparacion_violinplot_heart_disease_segun_var_cat_y_num", "style"),
    Input("dropdown_cat_comparacion_categorica_numerica", "value"),
    Input("dropdown_num_comparacion_categorica_numerica", "value"),
    Input("radio_item_box_violin_selector", "value")
)


def boxplot_comparacion_heart_disease_categorica_y_numerica_dropdown(dropdown_cat_comparacion_categorica_numerica,dropdown_num_comparacion_categorica_numerica,radio_item_box_violin_selector):
    
    diccionario_columnas_categoricas = {
        "HeartDisease": "Heart Disease",
        "Smoking": "Smoking",
        "AlcoholDrinking": "Alcohol Drinking",
        "Stroke": "Stroke",
        "DiffWalking": "DiffWalking",
        "Sex":"Sex",
        "AgeCategory": "Age Category",
        "Race": "Race",
        "Diabetic": "Diabetic",
        "PhysicalActivity": "Physical Activity",
        "GenHealth": "GenHealth",
        "Asthma": "Asthma",
        "KidneyDisease": "Kidney Disease",
        "SkinCancer": "Skin Cancer"
    
    }
    diccionario_variables_numericas = {
        "BMI":"Body Mass Index",
        "MentalHealth":"Mental Health",
        "PhysicalHealth":"Physical Health",
        "SleepTime":"Sleep Time"
    }


    if dropdown_cat_comparacion_categorica_numerica and dropdown_num_comparacion_categorica_numerica and (radio_item_box_violin_selector=="Violin Plot"):
        
        fig = go.Figure()

        fig.add_trace(go.Violin(x=df[dropdown_cat_comparacion_categorica_numerica][ df['HeartDisease'] == 'Yes' ],
                            y=df[dropdown_num_comparacion_categorica_numerica][ df['HeartDisease'] == 'Yes' ],
                            legendgroup='Yes', scalegroup='Yes', name='Yes',
                            side='negative',
                            line_color='red')
                )
        fig.add_trace(go.Violin(x=df[dropdown_cat_comparacion_categorica_numerica][ df['HeartDisease'] == 'No' ],
                            y=df[dropdown_num_comparacion_categorica_numerica][ df['HeartDisease'] == 'No' ],
                            legendgroup='No', scalegroup='No', name='No',
                            side='positive',
                            line_color='blue')
                )
        fig.update_traces(meanline_visible=True)
        fig.update_layout(violingap=0, violinmode='overlay')
      
  
    
        
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})


# 5.A CALLBACK SCATTER PLOT VARIABLES NUMERICAS
@app.callback(
    Output("scatter_correlacion_numerica_numerica", "figure"),
    Output("scatter_correlacion_numerica_numerica", "style"),
    Input("dropdown_1_scatter_correlacion_numerica_numerica", "value"),
    Input("dropdown_2_scatter_correlacion_numerica_numerica", "value"),
)

def scatter_plot_correlacion_numerica_numerica(dropdown_1_scatter_correlacion_numerica_numerica,dropdown_2_scatter_correlacion_numerica_numerica):
    
    
    diccionario_variables_numericas = {
        "BMI":"Body Mass Index",
        "MentalHealth":"Mental Health",
        "PhysicalHealth":"Physical Health",
        "SleepTime":"Sleep Time"
    }


    if dropdown_1_scatter_correlacion_numerica_numerica and dropdown_2_scatter_correlacion_numerica_numerica:
        fig = px.scatter(df, x=dropdown_1_scatter_correlacion_numerica_numerica, y=dropdown_2_scatter_correlacion_numerica_numerica, color="HeartDisease",
        title="Correlation between " + diccionario_variables_numericas[dropdown_1_scatter_correlacion_numerica_numerica] + "and " + diccionario_variables_numericas[dropdown_2_scatter_correlacion_numerica_numerica])
    
        
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})

if __name__ == '__main__':
    app.run_server()
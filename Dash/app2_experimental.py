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

def plot_correlation_matrix_numericals():
    df_numerical = df[['BMI','PhysicalHealth','MentalHealth', 'SleepTime']]
    # Compute the correlations
    corr_matrix = df.corr()

    # Generate the heatmap
    fig = px.imshow(corr_matrix, color_continuous_scale='Viridis')  # color_continuous_scale=["yellow", "red"]
    fig.update_layout(title="Correlation Matrix")
    fig.update_layout(height=500,width=400)

    return fig


# 6.A FUNCION RADAR CHART
def radar_chart():
    scaler = MinMaxScaler()
    numerical = df.loc[:, ["BMI","PhysicalHealth","MentalHealth", "SleepTime", "HeartDisease","AgeCategory", "Smoking", "KidneyDisease", "Stroke", "SkinCancer", "PhysicalActivity", "GenHealth"]]
    genhealth_mapping = {"Excellent":4,"Very good":3,"Good":2,"Fair":1,"Poor":0}
    agecategory_mapping = {"18-24":0,"25-29":1,"30-34":2,"35-39":3,"40-44":4,"45-49":5,"50-54":6,"55-59":7,
                            "60-64":8,"65-69":9,"70-74":10,"75-79":11,"80 or older":12}

    numerical['GenHealth']= numerical['GenHealth'].map(genhealth_mapping)
    numerical['AgeCategory']= numerical['AgeCategory'].map(agecategory_mapping)


    encoder = OrdinalEncoder()
    result = encoder.fit_transform(numerical.drop(['HeartDisease'], axis=1))
    numerical = pd.DataFrame(result, columns = numerical.drop(['HeartDisease'], axis=1).columns)
    scaler.fit(numerical)
    numerical_scaled = scaler.transform(numerical)
    numerical_scaled = pd.DataFrame(numerical_scaled, columns = numerical.columns)
    numerical_scaled["HeartDisease"] = df["HeartDisease"]
    numerical_yes_HeartDisease = numerical_scaled[numerical_scaled['HeartDisease'] == 'Yes' ] #rojo
    numerical_no_HeartDisease = numerical_scaled[numerical_scaled['HeartDisease'] == 'No' ]
    categories = ['BMI','PhysicalHealth','MentalHealth',
            'AgeCategory', 'Smoking', 'KidneyDisease', 'Stroke', 'SkinCancer','PhysicalActivity', 'GenHealth']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
          r=[numerical_no_HeartDisease['BMI'].mean(),numerical_no_HeartDisease['PhysicalHealth'].mean(), 
             numerical_no_HeartDisease['MentalHealth'].mean(), 
             numerical_no_HeartDisease['AgeCategory'].mean(), numerical_no_HeartDisease['Smoking'].mean(),
            numerical_no_HeartDisease['KidneyDisease'].mean(), numerical_no_HeartDisease['Stroke'].mean(),
            numerical_no_HeartDisease['SkinCancer'].mean(), numerical_no_HeartDisease['PhysicalActivity'].mean(),
            numerical_no_HeartDisease['GenHealth'].mean()],
          theta=categories,
          fill='toself',
          name='No Heart Disease',marker=dict(color="blue")
    ))
    fig.add_trace(go.Scatterpolar(
          r=[numerical_yes_HeartDisease['BMI'].mean(),numerical_yes_HeartDisease['PhysicalHealth'].mean(), 
             numerical_yes_HeartDisease['MentalHealth'].mean(),
            numerical_yes_HeartDisease['AgeCategory'].mean(), numerical_yes_HeartDisease['Smoking'].mean(),
            numerical_yes_HeartDisease['KidneyDisease'].mean(), numerical_yes_HeartDisease['Stroke'].mean(),
            numerical_yes_HeartDisease['SkinCancer'].mean(), numerical_yes_HeartDisease['PhysicalActivity'].mean(),
            numerical_yes_HeartDisease['GenHealth'].mean()],
          theta=categories, 
          fill='toself',
          name='Yes Heart Disease',marker=dict(color="red")
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
      showlegend=False
    ),

    fig.update_layout(title="Radar chart", width=550,height=550)

    return fig


# FUNCION PARA LEER EL MODELO Y REALIZAR LAS PREDICCIONES
def realizar_prediccion(smoking_value, alcohol_value, stroke_value, diffwalking_value, 
    sex_value, age_value, race_value, diabetic_value, physicalactivity_value, genhealth_value, asthma_value,
    kidneydisease_value, skincancer_value, bmi_value, sleeptime_value, mentalhealth_value, physicalhealth_value):
    prob = 0

    # Leer los modelos
    cluster = pk.load(open('../Modelo/cluster.plk', 'rb'))
    scaler = pk.load(open('../Modelo/scaler.plk', 'rb'))
    modelo_heart_disease = pk.load(open('../Modelo/modelo_heart_disease.plk', 'rb'))

    new_obs = pd.DataFrame([[bmi_value, smoking_value, alcohol_value, stroke_value, physicalhealth_value,
                            mentalhealth_value, diffwalking_value, sex_value, age_value, race_value, 
                            diabetic_value, physicalactivity_value, genhealth_value, sleeptime_value,
                            asthma_value, kidneydisease_value, skincancer_value]], index=[0],
                           columns=df.columns.drop('HeartDisease'))
    
    genhealth_mapping = {"Excellent":4,"Very good":3,"Good":2,"Fair":1,"Poor":0}
    agecategory_mapping = {"18-24":0,"25-29":1,"30-34":2,"35-39":3,"40-44":4,"45-49":5,"50-54":6,"55-59":7,
                           "60-64":8,"65-69":9,"70-74":10,"75-79":11,"80 or older":12}
    new_obs['GenHealth']= new_obs['GenHealth'].map(genhealth_mapping)
    new_obs['AgeCategory']= new_obs['AgeCategory'].map(agecategory_mapping)
    # new_obs = pd.get_dummies(new_obs, drop_first=True)
    obs_encoded = pd.DataFrame([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]], index=[0],
                               columns = ['BMI', 'PhysicalHealth', 'MentalHealth', 'AgeCategory', 'GenHealth',
                                          'SleepTime', 'Smoking_Yes', 'AlcoholDrinking_Yes',
                                          'Stroke_Yes', 'DiffWalking_Yes', 'Sex_Male', 'Race_Asian', 'Race_Black',
                                          'Race_Hispanic', 'Race_Other', 'Race_White',
                                          'Diabetic_No, borderline diabetes', 'Diabetic_Yes',
                                          'Diabetic_Yes (during pregnancy)', 'PhysicalActivity_Yes', 'Asthma_Yes',
                                          'KidneyDisease_Yes', 'SkinCancer_Yes'])
    obs_encoded['SleepTime']=new_obs['SleepTime']
    obs_encoded['PhysicalHealth']=new_obs['PhysicalHealth']
    obs_encoded['MentalHealth']=new_obs['MentalHealth']
    obs_encoded['GenHealth'] = new_obs['GenHealth']
    obs_encoded['AgeCategory'] = new_obs['AgeCategory']
    obs_encoded['Smoking_Yes'] = [1 if smoking_value=='Yes' else 0]
    obs_encoded['AlcoholDrinking_Yes'] = [1 if alcohol_value=='Yes' else 0]
    obs_encoded['Stroke_Yes'] = [1 if stroke_value=='Yes' else 0]
    obs_encoded['DiffWalking_Yes'] = [1 if diffwalking_value=='Yes' else 0]
    obs_encoded['Sex_Male'] = [1 if sex_value=='Male' else 0]
    obs_encoded['Race_Black'] = [1 if race_value=='Black' else 0]
    obs_encoded['Race_Hispanic'] = [1 if race_value=='Hispanic' else 0]
    obs_encoded['Race_Other'] = [1 if race_value=='Other' else 0]
    obs_encoded['Race_White'] = [1 if race_value=='White' else 0]
    obs_encoded['Diabetic_No, borderline diabetes'] = [1 if diabetic_value=='No, borderline diabetes' else 0]
    obs_encoded['Diabetic_Yes'] = [1 if diabetic_value=='Yes' else 0]
    obs_encoded['Asthma_Yes'] = [1 if asthma_value=='Yes' else 0]
    obs_encoded['KidneyDisease_Yes'] = [1 if kidneydisease_value=='Yes' else 0]
    obs_encoded['SkinCancer_Yes'] = [1 if skincancer_value=='Yes' else 0]
    
    # Variables que se eliminaran pero que hay que poner antes del scaler para que no de error
    obs_encoded['BMI']=0
    obs_encoded['Diabetic_Yes (during pregnancy)']=0
    obs_encoded['PhysicalActivity_Yes']=0
    obs_encoded['Race_Asian']=0
    
    
    obs_encoded_scaled = pd.DataFrame(scaler.transform(obs_encoded), columns=obs_encoded.columns)
    
    obs_encoded_scaled = obs_encoded_scaled.drop(['BMI','Diabetic_Yes (during pregnancy)','PhysicalActivity_Yes','Race_Asian'],axis=1)
    
    obs_encoded_scaled['cluster'] = cluster.predict(obs_encoded_scaled)
    
    probability = modelo_heart_disease.predict_proba(obs_encoded_scaled)
    #prediction = modelo_heart_disease.predict(obs_encoded_scaled)
    
    return probability[0][1]

dropdown_modelo_yes_no = [{'value': "Yes", 'label':"Yes"}, {'value': "No", 'label':"No"}]
dropdpown_modelo_sex = [{'value': "Male", 'label':"Male"}, {'value': "Female", 'label':"Female"}]
dropdpown_modelo_agecategory = [{'value': "18-24", 'label':"18-24"},
    {'value': "25-29", 'label':"25-29"}, {'value': "30-34", 'label':"30-34"},
    {'value': "35-39", 'label':"35-39"}, {'value': "40-44", 'label':"40-44"}, {'value': "45-49", 'label':"45-49"},
    {'value': "50-54", 'label':"50-54"}, {'value': "55-59", 'label':"55-59"},
    {'value': "60-64", 'label':"60-64"}, {'value': "65-69", 'label':"65-69"},
    {'value': "70-74", 'label':"70-74"}, {'value': "75-79", 'label':"75-79"},
    {'value': "80 or older", 'label':"80 or older"}]

dropdown_modelo_race = [{'value': "White", 'label':"White"}, {'value': "Hispanic", 'label':"Hispanic"},
    {'value': "Black", 'label':"Black"}, {'value': "Asian", 'label':"Asian"}, 
    {'value': "American Indian/Alaskan Native", 'label':"Native American"}, {'value': "Other", 'label':"Other"}]

dropdown_modelo_diabetic = [{'value': "No", 'label':"No"}, {'value': "No, borderline diabetes", 'label':"No, borderline diabetes"},
    {'value': "Yes", 'label':"Yes"}, {'value': "Yes (during pregnancy)", 'label':"Yes (during pregnancy)"}]

dropdown_modelo_genhealth = [{'value': "Excellent", 'label':"Excellent"}, {'value': "Very good", 'label':"Very good"},
    {'value': "Good", 'label':"Good"}, {'value': "Fair", 'label':"Fair"},
    {'value': "Poor", 'label':"Poor"}]





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

        html.Div(
            children = [
                html.Div(
                    children=[
                        html.H3(
                            children = [
                                "VARIABLES NUMÉRICAS: DISTRIBUCIÓN"
                            ],
                            id = "distribucion_numericas",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
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
                                                "margin-left": "10px",
                                                "display": "block",
                                                'font-size' : '85%',
                                                'font-family':'verdana'
                                            }
                                        ),
                                # Radio item para elegir distribución de las variables cuantitativas entre los que tienen Heart Disease y los que no tienen 
                                dcc.RadioItems(
                                    id="radio_item_dist_var_numerica_selector_general_yes_no",
                                    options=[
                                        {'label': '                     Distribución General                                   ', 'value': 'Distribución General'},
                                        {'label': '                     Distribución enfermos                                   ', 'value': 'Distribución YES Heart Disease'},
                                        {'label': '                     Distribución no enfermos', 'value': 'Distribución NO Heart Disease'}
                                    ],
                                    value='Distribución YES Heart Disease', style={"margin-top":"25px", "margin-left":"10px"}
                                ),
                                dcc.Graph(
                                            id = "histograma_distribucion_numericas_general_yes_no",  # dropdown_histograma_distribucion_numericas
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
                                        "margin-bottom": "15px",
                                        "font-size":"15px"
                                    }
                                ),

                                dcc.Slider(id="slider_histograma_numericas", min=1, max=6, step=1, value=2, marks={'1': '1', '2': '2','3':'3','4':'4','5':'5','6':'6'})
                            ],
                            style={
                                "margin-left": "25%",
                                "margin-right": "25%",
                                "margin-bottom" : "4px",
                                "text-align": "center"
                            }
                        ),
                    ],
                    id="div_histograma_numericas",
                    style={
                        "background-color":"rgb(255,255,255)",
                        "border":"1px solid rgb(204,202,202)",
                        "margin":"1%",
                        "margin-top":"48px",  # 32 px
                        "margin-left":"2.9%",
                        "margin-right":"0%",
                        "padding-left":"0.5%",
                        "display": "inline-block",
                        "border-radius" : "2px",
                        "width":"50em",
                        "padding-bottom":"1%",
                        "height":"40em"
                    }
                ),
                html.Div(
                    children = [
                        html.H3(
                            children = [
                                "MATRIZ DE CORRELACIÓN ENTRE VARIABLES NUMÉRICAS"
                            ],
                            id = "titulo_matriz_correlacion",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "8%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }
                        ),
                        dcc.Graph(
                            id = "plot_matriz_correlacion",
                            figure=plot_correlation_matrix_numericals(),
                            style = {
                                #"display": "none"
                            }
                        )
                    ],
                    id="div_matriz_correlacion",
                    style={
                        "background-color":"rgb(255,255,255)",
                        "border":"1px solid rgb(204,202,202)",
                        "margin":"1%",
                        "margin-top":"48px",  # 32 px
                        "margin-left":"1.5%",
                        "margin-right":"0%",
                        "padding-left":"0.5%",
                        "display": "inline-block",
                        "border-radius" : "2px",
                        "width":"26.5em",
                        "padding-bottom":"1%",
                        "verticalAlign": "top",
                        "height":"40em"
                    }
                ),
            ],
            style={"width":"100%","display":"inline"}
        ),

        ## 6. Correlación entre variables numéricas
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H3(
                            children = [
                                "Correlación entre variables numéricas"
                            ],
                            id = "titulo_correlacion_entre_variables_numericas",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
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
                                                "margin-left": "10px",
                                                "display": "block",
                                                'font-size' : '85%',
                                                'font-family':'verdana'
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
                                                "margin-left": "10px",
                                                "display": "block",
                                                'font-size' : '85%',
                                                'font-family':'verdana'
                                            }
                                        ),
                                dcc.Graph(
                                            id = "scatter_correlacion_numerica_numerica",
                                            style = {
                                                "display": "none"
                                            }
                                        )

                            ]
                        ),
                    ],
                    id="div_scatter_complejo_numericas",
                    style={
                        "background-color":"rgb(255,255,255)",
                        "border":"1px solid rgb(204,202,202)",
                        "margin":"1%",
                        "margin-top":"48px",  # 32 px
                        "margin-left":"2.9%",
                        "margin-right":"0%",
                        "padding-left":"0.5%",
                        "display": "inline-block",
                        "border-radius" : "2px",
                        "width":"42em",
                        "padding-bottom":"1%",
                        "height":"39em"
                    }
                ),
                html.Div(
                    children=[
                        html.H3(
                            children = [
                                "PERFIL DE PERSONAS SANAS Y ENFERMAS"
                            ],
                            id = "radar_chart_variables",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }
                        ),
                        html.Div(
                            children = [
                                dcc.Graph(
                                    figure=radar_chart(),
                                    id = "titulo_radar_chart",
                                    style = {
                                        "display": "inline-block",
                                        "margin-left":"2px","padding-left":"0px",
                                        "margin-right":"0px","padding-right":"0px"
                                    }
                                )
                            ]
                        ),
                    ],
                    id="div_radar_chart",
                    style={
                        "background-color":"rgb(255,255,255)",
                        "border":"1px solid rgb(204,202,202)",
                        "margin":"1%",
                        "margin-top":"48px",  # 32 px
                        "margin-left":"1.5%",
                        "margin-right":"0%",
                        "padding-left":"0.5%",
                        "display": "inline-block",
                        "border-radius" : "2px",
                        "width":"34.5em",
                        "padding-bottom":"1%",
                        "verticalAlign": "top",
                        "height":"39em"
                    }
                )
            ],
            style={"width":"100%","display":"inline"}
        ),

        html.Div(
            children=[
                ## 5. Comparacion Heart Disease según variables categóricas y numéricas
                html.H3(
                    children = [
                        "DISTRIBUCIÓN DE VARIABLES NUMERICAS PARA DISTINTAS CATEGORÍAS"
                    ],
                    id = "comparacion_heart_disease_categorica_numerica",
                    style = {
                        "text-align": "left",
                        "margin": "2.5%",
                        "margin-bottom": "4%",
                        "text-align": "left",
                        "font-family": "verdana",
                        "font-weight": "600",
                        "color": "rgb(67,67,67)",
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
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
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
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
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
            ],
            id="div_box_plot_violin",
            style={
                "background-color":"rgb(255,255,255)",
                "border":"1px solid rgb(204,202,202)",
                "margin":"1%",
                "margin-top":"48px",  # 32 px
                "margin-left":"2.9%",
                "margin-right":"0%",
                "padding-left":"0.5%",
                "display": "inline-block",
                "border-radius" : "2px",
                "width":"78.5em",
                "padding-bottom":"1%",
                "height":"48em"
            }
        ),

        html.Div(
            children=[
                        ## 7. Modelo
                html.H3(
                    children = [
                        "REALIZAR PREDICCIONES"
                    ],
                    id = "titulo_modelo",
                    style = {
                        "text-align": "left",
                        "margin": "2.5%",
                        "margin-bottom": "4%",
                        "text-align": "left",
                        "font-family": "verdana",
                        "font-weight": "600",
                        "color": "rgb(67,67,67)",
                        "text-align": "left"
                    }
                ),
                html.Div( ## SMOKING
                    children = [
                        html.H4( 
                            children = [
                                "Smoking?"
                            ],
                            id = "titulo_dropdown_modelo_smoking",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_smoking",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## ALCOHOL DRINKING
                    children = [
                        html.H4( 
                            children = [
                                "Alcohol drinking?"
                            ],
                            id = "titulo_dropdown_modelo_alcoholdrinking",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_alcoholdrinking",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## STROKE
                    children = [
                        html.H4( 
                            children = [
                                "Stroke?"
                            ],
                            id = "titulo_dropdown_modelo_stroke",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_stroke",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## DiffWalking
                    children = [
                        html.H4( 
                            children = [
                                "Difficulty Walking?"
                            ],
                            id = "titulo_dropdown_diffwalking",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_diffwalking",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Sex
                    children = [
                        html.H4( 
                            children = [
                                "Sex?"
                            ],
                            id = "titulo_dropdown_modelo_sex",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdpown_modelo_sex,
                                    value="Male",
                                    id = "dropdown_modelo_sex",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Age Category
                    children = [
                        html.H4( 
                            children = [
                                "Age?"
                            ],
                            id = "titulo_dropdown_modelo_age",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdpown_modelo_agecategory,
                                    value="18-24",
                                    id = "dropdown_modelo_agecategory",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Race
                    children = [
                        html.H4( 
                            children = [
                                "Race?"
                            ],
                            id = "titulo_dropdown_modelo_race",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_race,
                                    value="White",
                                    id = "dropdown_modelo_race",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Diabetic
                    children = [
                        html.H4( 
                            children = [
                                "Diabetes?"
                            ],
                            id = "titulo_dropdown_modelo_diabetic",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_diabetic,
                                    value="No",
                                    id = "dropdown_modelo_diabetic",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Physical Activity
                    children = [
                        html.H4( 
                            children = [
                                "Regular Physical Activity?"
                            ],
                            id = "titulo_dropdown_modelo_physicalactivity",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_physicalactivity",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## GenHealth
                    children = [
                        html.H4( 
                            children = [
                                "Describe your general health"
                            ],
                            id = "titulo_dropdown_modelo_genhealth",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_genhealth,
                                    value="Good",
                                    id = "dropdown_modelo_genhealth",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Asthma
                    children = [
                        html.H4( 
                            children = [
                                "Asthma?"
                            ],
                            id = "titulo_dropdown_modelo_asthma",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_asthma",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## KidneyDisease
                    children = [
                        html.H4( 
                            children = [
                                "Kidney Disease?"
                            ],
                            id = "titulo_dropdown_modelo_kidneydisease",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_kidneydisease",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## Skin Cancer
                    children = [
                        html.H4( 
                            children = [
                                "Skin Cancer?"
                            ],
                            id = "titulo_dropdown_modelo_skincancer",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Dropdown(
                                    options = dropdown_modelo_yes_no,
                                    value="No",
                                    id = "dropdown_modelo_skincancer",
                                    style = {
                                        "display": "block",
                                        "width": "300px",
                                        "margin-left": "10px",
                                        "margin-right": "20px",
                                        "display": "block",
                                        'font-size' : '85%',
                                        'font-family':'verdana'
                                    }
                                )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## BMI
                    children = [
                        html.H4( 
                            children = [
                                "Body Mass Index?"
                            ],
                            id = "titulo_dropdown_modelo_bmi",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        daq.NumericInput(
                            id="numeric_input_bmi",
                            min=1,
                            max=120,
                            value=23
                        )

                    ],
                    style={"display": "inline-block", "margin":"1%"}
                ),
                html.Div( ## SleepTime
                    children = [
                        html.H4( 
                            children = [
                                "Average Sleep Time?"
                            ],
                            id = "titulo_dropdown_modelo_sleeptime",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        daq.NumericInput(
                            id="numeric_input_sleeptime",
                            min=0,
                            max=24,
                            value=8
                        )

                    ],
                    style={"display": "inline-block", "margin":"1%", "margin-right": "10%"}
                ),

                html.Div( ## MentalHealth
                    children = [
                        html.H4( 
                            children = [
                                "For how many days during the past 30 days was your mental health not good?"
                            ],
                            id = "titulo_dropdown_modelo_mentalhealth",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Slider(id="slider_modelo_mentalhealth", min=0, max=30, step=1, value=2, marks={'1': '1', '2': '2','3':'3','4':'4','5':'5','6':'6',
                                                                                                            '7': '7', '8': '8','9':'9','10':'10','11':'11','12':'12',
                                                                                                            '13': '13', '14': '14','15':'15','16':'16','17':'17','18':'18',
                                                                                                            '19': '19', '20': '20','21':'21','22':'22','23':'23','24':'24',
                                                                                                            '25': '25', '26': '26','27':'27','28':'28','29':'29','30':'30'})

                    ],
                    style={"display": "inline-block", "margin":"1%", "width":"45%"}
                ),
                html.Div( ## PhysicalHealth
                    children = [
                        html.H4( 
                            children = [
                                "For how many days during the past 30 days was your physical health not good? Including injuries or physical illness"
                            ],
                            id = "titulo_dropdown_modelo_physicalhealth",
                            style = {
                                "text-align": "left",
                                "margin": "2.5%",
                                "margin-bottom": "4%",
                                "text-align": "left",
                                "font-family": "verdana",
                                "font-weight": "600",
                                "color": "rgb(67,67,67)",
                                "text-align": "left"
                            }

                        ),
                        dcc.Slider(id="slider_modelo_physicalhealth", min=0, max=30, step=1, value=2, marks={'1': '1', '2': '2','3':'3','4':'4','5':'5','6':'6',
                                                                                                            '7': '7', '8': '8','9':'9','10':'10','11':'11','12':'12',
                                                                                                            '13': '13', '14': '14','15':'15','16':'16','17':'17','18':'18',
                                                                                                            '19': '19', '20': '20','21':'21','22':'22','23':'23','24':'24',
                                                                                                            '25': '25', '26': '26','27':'27','28':'28','29':'29','30':'30'})

                    ],
                    style={"display": "inline-block", "margin":"1%", "width":"45%"}
                ),

                #BOTON
                html.Div(
                    children = [
                        html.Button('Realizar Predicción', id='button', 
                            style={
                                "display": "inline-block",
                                "border": "none",
                                "padding": "1rem 2rem",
                                "margin": "0",
                                "text-decoration": 2,
                                "background": "darkcyan",
                                "color": "#ffffff",
                                "font-family":"sans-serif",
                                "font-size": "1rem",
                                "cursor": "pointer",
                                "text-align": "center",
                                "border-radius":"5px"}
                        )
                    ], style={ "margin":"2%", "padding-left":"38%"}
                ),

                ## DIV PREDICCION MODELO
                html.Div(
                    id ="div_prediccion"
                )
            ],
            id="div_panel_modelo",
            style={
                "background-color":"rgb(255,255,255)",
                "border":"1px solid rgb(204,202,202)",
                "margin":"1%",
                "margin-top":"48px",  # 32 px
                "margin-left":"2.9%",
                "margin-right":"0%",
                "padding-left":"5%",
                "padding-right":"5%",
                "display": "inline-block",
                "border-radius" : "2px",
                "width":"70em",
                "padding-bottom":"1%",
                "height":"55em"
            }
        ),



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

'''
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

        fig.update_layout(height=390)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})
'''

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



# 2.A CALLBACK DE RADIO BUTTON HISTOGRAMAS PARA VER DISTRIBUCION DE NUMERICAS EN GENERAL Y YES/NO HEART DISEASE 
@app.callback(
    Output("histograma_distribucion_numericas_general_yes_no", "figure"),
    Output("histograma_distribucion_numericas_general_yes_no", "style"),
    Input("dropdown_numericas", "value"),
    Input("slider_histograma_numericas", "value"),
    Input("radio_item_dist_var_numerica_selector_general_yes_no", "value")
)

def histograma_distribucion_numericas_dropdown_yes_no(dropdown_numericas,slider_histograma_numericas, radio_item_dist_var_numerica_selector_general_yes_no):
    
    diccionario_variables_numericas = {
        "BMI":"Body Mass Index",
        "MentalHealth":"Mental Health",
        "PhysicalHealth":"Physical Health",
        "SleepTime":"Sleep Time"
    }

    if dropdown_numericas and radio_item_dist_var_numerica_selector_general_yes_no=="Distribución General":
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
        fig.update_layout(height=390)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        
        return (fig,{"display":"block"})
    
    elif dropdown_numericas and radio_item_dist_var_numerica_selector_general_yes_no=="Distribución YES Heart Disease":
        
        df_yes_HeartDisease = df[df['HeartDisease'] == 'Yes' ]
        data = [
        go.Histogram(
            x = df_yes_HeartDisease[dropdown_numericas],
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
        fig.update_layout(height=390)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        return (fig,{"display":"block"})
    elif dropdown_numericas and radio_item_dist_var_numerica_selector_general_yes_no=="Distribución NO Heart Disease":
        
        df_no_HeartDisease = df[df['HeartDisease'] == 'No' ]
        data = [
        go.Histogram(
            x = df_no_HeartDisease[dropdown_numericas],
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

        fig.update_layout(height=390)

        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        return (fig,{"display":"block"})
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
        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        return (fig,{"display":"block", "height":"555px"})
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
        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
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
    
        fig.update_layout(
            font=dict(
                family="Verdana",
                size=11,  # Set the font size here
            )
        )
        return (fig,{"display":"block"})
    else:
        return (go.Figure(data = [], layout = {}), {"display": "none"})

## CALLBACK BOTON
@app.callback(
    Output('div_prediccion', 'children'),
    Input('button', 'n_clicks'),
    State('dropdown_modelo_smoking', 'value'),
    State('dropdown_modelo_alcoholdrinking', 'value'),
    State('dropdown_modelo_stroke', 'value'),
    State('dropdown_modelo_diffwalking', 'value'),
    State('dropdown_modelo_sex', 'value'),
    State('dropdown_modelo_agecategory', 'value'),
    State('dropdown_modelo_race', 'value'),
    State('dropdown_modelo_diabetic', 'value'),
    State('dropdown_modelo_physicalactivity', 'value'),
    State('dropdown_modelo_genhealth', 'value'),
    State('dropdown_modelo_asthma', 'value'),
    State('dropdown_modelo_kidneydisease', 'value'),
    State('dropdown_modelo_skincancer', 'value'),
    State('numeric_input_bmi', 'value'),
    State('numeric_input_sleeptime', 'value'),
    State('slider_modelo_mentalhealth', 'value'),
    State('slider_modelo_physicalhealth', 'value')
)

def update_div_prediccion(n_clicks,smoking_value, alcohol_value, stroke_value, diffwalking_value, 
    sex_value, age_value, race_value, diabetic_value, physicalactivity_value, genhealth_value, asthma_value,
    kidneydisease_value, skincancer_value, bmi_value, sleeptime_value, mentalhealth_value, physicalhealth_value):
    return 'Probability of having heart disease: {}%'.format(round(realizar_prediccion(smoking_value, alcohol_value, stroke_value, diffwalking_value, 
            sex_value, age_value, race_value, diabetic_value, physicalactivity_value, genhealth_value, asthma_value,
            kidneydisease_value, skincancer_value, bmi_value, sleeptime_value, mentalhealth_value, physicalhealth_value)*100,2))


if __name__ == '__main__':
    app.run_server()
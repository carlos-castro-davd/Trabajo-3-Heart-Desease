import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from plotly.graph_objects import Layout
import dash_daq as daq
import pickle as pk


def plot_correlation_matrix_numericals(df):
    '''
    Función para representar una matriz de correlación entre las variables numéricas de un dataset

    Inputs:
    ======
    
    df : pandas DataFrame. Datos sobre los que se computará la correlación.

    Output:
    ======

    fig : plotly figure. Representación gráfica de la matriz de correlaciones.
    '''
    #df_numerical = df[['BMI','PhysicalHealth','MentalHealth', 'SleepTime']]
    # Compute the correlations
    corr_matrix = df.corr()

    # Generate the heatmap
    fig = px.imshow(corr_matrix, color_continuous_scale='Viridis')
    fig.update_layout(title="Correlation Matrix")
    fig.update_layout(height=500,width=400)

    return fig


def radar_chart(df):
    '''
    Función para representar el radar chart para los perfines medios de persona enferma y persona no enferma. Computa el valor 
    medio de varias de las variables para cada uno de estos grupos (incluyendo algunas categoricas orginales convertidas a
    numéricas, así como algunas de las binarias (Yes/No) para obtener una visión general).

    Inputs:
    ======

    df : pandas DataFrame. Datos originales

    Output:
    ======

    fig : plotly figure. Radar chart.
    
    '''
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


def bullet_chart_heartdisease_probability(probability):
    '''
    Función para generar una representación visual de la probabilidad computada por el modelo, usando un bullet chart.

    Inputs:
    ======

    probability : float. Cifra de probabilidad a representar

    Output:
    ======

    fig : plotly figure. Bullet chart representando la probabilidad.

    '''
    fig = go.Figure(go.Indicator(
    mode = "number+gauge", value = probability, number_font_color="black", 
    number = {"suffix": "%"},
    domain = {'x': [0, 1], 'y': [0, 1]},
    delta = {'reference': 50, 'position': "top", 'valueformat':'.2%'},
    title = {'text':"<b>Prob %</b><br><span style='color: gray; font-size:0.8em'></span>", 'font': {"size": 14}},
    gauge = {
        'shape': "bullet",
        'axis': {'range': [None, 100]},
        'threshold': {
            'line': {'color': "red", 'width': 2},
            'thickness': 0.75, 'value': 270},
        'bgcolor': "white",
        'steps': [
            {'range': [0, 50], 'color': "lightskyblue"},
            {'range': [50, 100], 'color': "orangered"}],
        'bar': {'color': "black"}}))
    fig.update_layout(height = 250, width=1000)
    
    return fig


def realizar_prediccion(df_columns, smoking_value, alcohol_value, stroke_value, diffwalking_value, 
    sex_value, age_value, race_value, diabetic_value, physicalactivity_value, genhealth_value, asthma_value,
    kidneydisease_value, skincancer_value, bmi_value, sleeptime_value, mentalhealth_value, physicalhealth_value):

    '''
    Función para leer los modelos (guardados previamente en formato pkl) y realizar la predicción.

    Inputs:
    ======

    df_columns : lista de Strings. Lista de las columnas de df, necesaria para estructurar adecuadamente los valores de entrada.

    smoking_value, alcohol_value, stroke_value, 
    diffwalking_value, sex_value, age_value, 
    race_value, diabetic_value, genhealth_value,
    asthma_value, kidneydisease_value, skincancer_value 
    physicalactivity_value                              : String. Clases elegidas desde el dash para cada una de las categóricas

    bmi_value, sleeptime_value : float. Valor de BMI y duración media del sueño en horas
    
    mentalhealth_value, physicalhealth_value : int. Salud mental y física.

    Output:
    ======

    probability[0][1] : int. Probabilidad estimada por el modelo para dicha observación.

    '''

    # Leer los modelos
    cluster = pk.load(open('../Modelo/cluster.plk', 'rb'))
    scaler = pk.load(open('../Modelo/scaler.plk', 'rb'))
    modelo_heart_disease = pk.load(open('../Modelo/modelo_heart_disease.plk', 'rb'))

    new_obs = pd.DataFrame([[bmi_value, smoking_value, alcohol_value, stroke_value, physicalhealth_value,
                            mentalhealth_value, diffwalking_value, sex_value, age_value, race_value, 
                            diabetic_value, physicalactivity_value, genhealth_value, sleeptime_value,
                            asthma_value, kidneydisease_value, skincancer_value]], index=[0],
                            columns=df_columns.drop('HeartDisease'))
    
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
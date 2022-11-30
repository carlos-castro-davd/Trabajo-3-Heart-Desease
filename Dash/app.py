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


# Para este Dash, vamos a seleccionar un fichero de datos y realizar un dashboard descriptivo
# sobre un conjunto de datos

df = pd.read_csv('../Datos/heart_2020_cleaned.csv')


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

        # 1. Titulo (POSIBLEMENTE EDITAR A MARKDOWN)
        html.H1(
            children = [
                "Heart Disease Analysis"
            ],
            id = "titulo",
            style = {
                "text-align": "center",
                "margin-bottom": "20px",
                "height": "50px"
            }

        ),
        html.P(
            children = [
                "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since " +
                "the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, " +
                "but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset " +
                "sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."
            ],
            id ="intro",
            style = {
                "text-align": "left",
                "margin-bottom": "20px"
            }
        ),



        ## 2. Distribucion variables categoricas

        html.H2(
            children = [
                "Variables Catégoricas: Distribución"
            ],
            id = "distribucion_categoricas",
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
                            id = "dropdown_categoricas",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                dcc.Graph(
                            id = "dropdown_piechart_distribucion_categoricas",
                            style = {
                                "display": "none"
                            }
                        )
            ]

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

        ## 4. Porcentaje Heart Disease en Variables Categoricas
        html.H2(
            children = [
                "Variables Categóricas: Porcentaje de Heart Disease"
            ],
            id = "porcentaje_heart_disease_var_categoricas",
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
                            id = "dropdown_porcentaje_heart_disease_variables_categoricas",
                            style = {
                                "display": "block",
                                "width": "300px",
                                "margin-left": "10px"
                            }
                        ),
                html.Div(

                    children = [

                        #html.Div(
                            #children = [
                                dcc.Graph(
                                    id = "hist_porcentaje_heart_disease_categoricas",  style={'display': 'inline-block'}
                                ),

                            #], className="six columns"

                        #),

                        ##html.Div(
                            #children = [
                                dcc.Graph(
                                    id = "hist_total_heart_disease_categoricas", style={'display': 'inline-block'}
                                )
                            #], className="six columns"
                        #)
                    ],style = { "display": "inline-block"},className="row"
                ),
            

            ], #style = { "display": "inline-block"} 
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
                                "display": "none"
                            }
                        ),
                
                dcc.Graph(
                            id = "comparacion_violinplot_heart_disease_segun_var_cat_y_num",
                            style = {
                                "display": "none"
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
          'margin-right': "5%",
          'margin-left': "5%",
          'border-radius': '10px',
          "font-family": "sans-serif"
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
        
        return (fig,{"display":"block"})
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
                    marker_color = "firebrick",
                    text= ["{0}%".format(round(value*100,1)) for value in y],
                    textposition="auto",
                    opacity=0.9,
                    textangle=0,
                    textfont_size = 20,
                    textfont_color= "white",
                   )

        data = [trace]
        layout = go.Layout(title = "% Heart Disease by " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], xaxis_title = diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], yaxis_title = "% of heart disease")
        fig = go.Figure(data = data, layout = layout)
        

        
        return (fig,{"display":"inline"})
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
                marker_color = "darkorange",
                name = "Heart Disease"
            ),
            go.Histogram(
                x = df[df["HeartDisease"] == 'No'][dropdown_porcentaje_heart_disease_variables_categoricas],
                marker_color = "mediumseagreen",
                name = "No Heart Disease"
            )
        ]
        layout = go.Layout(title = "Heart Disease según " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], xaxis_title =diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas], yaxis_title = "Heart Disease según " + diccionario_columnas_categoricas[dropdown_porcentaje_heart_disease_variables_categoricas])

        fig = go.Figure(data = data, layout = layout)
        

        
        return (fig,{"display":"inline"})
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

  
    
        
        return (fig,{"display":"block"})
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
                            line_color='blue')
                )
        fig.add_trace(go.Violin(x=df[dropdown_cat_comparacion_categorica_numerica][ df['HeartDisease'] == 'No' ],
                            y=df[dropdown_num_comparacion_categorica_numerica][ df['HeartDisease'] == 'No' ],
                            legendgroup='No', scalegroup='No', name='No',
                            side='positive',
                            line_color='orange')
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
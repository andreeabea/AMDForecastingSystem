# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import datetime
import os
import re
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import tensorflow as tf
import dash_table
import pandas as pd
import numpy as np
from dash.dependencies import Output, Input, State
from PIL import Image
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px

from data_processing.db_handler import DbHandler
from data_processing.latent_code_handler import LatentCodeHandler
from data_processing.timeseries_augmentation import TimeSeriesGenerator
from neural_networks.rnn import Rnn

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

datatype = 'numerical'
include_timestamps = True

db_handler = DbHandler(datatype, include_timestamps)
data = db_handler.get_data_from_csv(normalize=False)

features1 = data.columns.values

for i in range(0, 24):
    if 'MinCentralThickness' in features1[i] or 'MaxCentralThickness' in features1[i] \
            or 'CentralThickness' in features1[i] or 'TotalVolume' in features1[i]:
        features1[i] = features1[i][:len(features1[i]) - 1]
    else:
        if '0' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'C0'
        elif '1' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'N1'
        elif '2' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'N2'
        elif '3' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'S1'
        elif '4' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'S2'
        elif '5' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'T1'
        elif '6' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'T2'
        elif '7' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'I1'
        elif '8' in features1[i]:
            features1[i] = features1[i][:len(features1[i]) - 1] + 'I2'

tabs_styles = {
    'height': '44px'
}

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}

app.layout = html.Div(children=[
    html.H6(children='AMD evolution forecasting system'),

    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Train models', value='tab-1', style=tab_style),
        dcc.Tab(label='Forecast visual acuities', value='tab-2', style=tab_style),
    ], style=tabs_styles),
    html.Div(id='tabs-example-content')

])


# @app.callback(Output("folder-files", "children"), Input("dropdown", "value"))
# def list_all_files(folder_name):
#     # This is relative, but you should be able
#     # able to provide the absolute path too
#     file_names = os.listdir(folder_name)
#
#     file_list = html.Ul([html.Li(file) for file in file_names])
#
#     return file_list


@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        algorithms = ["Linear regression", "LASSO regression", "Gradient Boosting Regression",
                      "Random Forest Regression", "Extremely Randomized Trees", "Simple RNN network",
                      "LSTM network", "GRU network"]
        data_transform = ['Monthly time-series resampling', "Original series with timestamps as features"]
        feature_sel = ['All features', "Recursive Feature Elimination", "LASSO feature selection"]

        controls = [
            html.Label(['Machine Learning algorithm', dcc.Dropdown(
                options=[{"label": x, "value": x} for x in algorithms],
                value=algorithms[0]
            )], style={'font-size': '12px'}),
            html.Label(['Data transformation method', dcc.Dropdown(
                options=[{"label": x, "value": x} for x in data_transform],
                value=data_transform[0],
            )], style={'font-size': '12px'})
            ,
            html.Label(['Feature selection method', dcc.Dropdown(
                options=[{"label": x, "value": x} for x in feature_sel],
                value=feature_sel[0],
            )], style={'font-size': '12px'}),
            html.Label(['Time-series size', dcc.Slider(
                min=2,
                max=10,
                value=2,
                marks={
                    2: {'label': '2 visits'},
                    3: {'label': '3 visits'},
                    4: {'label': '4 visits'},
                    5: {'label': '5 visits'},
                    6: {'label': '6 visits'},
                    7: {'label': '7 visits'},
                    8: {'label': '8 visits'},
                    9: {'label': '9 visits'},
                    10: {'label': '10 visits'}
                },
                included=False
            )], style={'font-size': '12px'})
            ,
            html.Button('Train', id='submit-val', style={'left': '30%', 'position': 'relative'})
        ]

        return html.Div([
            html.Div([html.H3(''),
                      html.Div(controls),  # html.Div(id="folder-files")
                      ])
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'})
    elif tab == 'tab-2':
        patients = data.index.get_level_values(0).tolist()
        patients = list(dict.fromkeys(patients))
        # forecast for existing patient or add new patient
        controls = [
            dcc.ConfirmDialog(
                id='confirm',
                message='Danger danger! Are you sure you want to continue?',
            ),
            html.Div(html.Label(['Select patient and eye', dcc.Dropdown(
                id='dropdown',
                options=[{"label": x, "value": x} for x in patients],
                value=patients[0]
            )], style={'font-size': '12px'})),
            # html.Label(['Choose forecasting time span', dcc.Slider(
            #     min=1,
            #     max=3,
            #     value=1,
            #     marks={
            #         1: {'label': '1 month'},
            #         2: {'label': '2 month'},
            #         3: {'label': '3 months'}
            #     },
            #     included=False
            # )], style={'font-size': '12px'})
            # ,
            html.Button('Predict', id='predict', style={'left': '30%', 'position': 'relative'}),
            html.Div(id='prediction-container')
        ]

        return html.Div([
            html.Div([html.H3(''),
                      html.Div(controls),  # html.Div(id="folder-files")
                      ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle',
                                'horizontal-align': 'middle'}),
            html.Div(id='dd-output-container',
                     style={'width': '67%', 'display': 'inline-block', 'vertical-align': 'middle',
                            'horizontal-align': 'middle', 'padding-top': '10px', 'padding-left': '20px'}),
            html.Div([dcc.Graph(
                id='va-evolution-graph',
                style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'middle',
                   }),
                html.Div(id='heatmap-container', title='Fundus image heatmap',
                         style={'width': '17.25%', 'display': 'inline-block', 'vertical-align': 'middle',
                                'horizontal-align': 'middle', 'padding-top': '10px', 'padding-left': '123px'})
            ])
        ])


current_patient = None
patient_data_notres = None


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output(value):
    global current_patient, patient_data_notres
    current_patient = value
    patient_data = data.loc[data.index.get_level_values(0) == value].copy()
    patient_data.drop(patient_data.tail(1).index, inplace=True)
    patient_data_notres = patient_data
    patient_data.columns = features1
    patient_data.insert(0, 'Visit date', patient_data.index.get_level_values(1))
    patient_data['Visit date'] = patient_data['Visit date'].dt.date
    patient_data = patient_data.round(2)

    return html.Div([
        html.Div(dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i, "selectable": True} for i in patient_data.columns],
        data=patient_data.to_dict('records'),
        style_table={
            'overflowY': 'scroll',
            'overflowX': 'scroll'
        }), style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'middle',
                                'horizontal-align': 'middle'}),
        html.Div(id='oct-image-container', style={'width': '25%', 'display': 'inline-block', 'vertical-align': 'middle',
                            'horizontal-align': 'middle', 'padding-top': '10px', 'padding-left': '20px'}, title='OCT fundus scan')
    ])


@app.callback(
    [Output('oct-image-container', 'children'),
     Output('heatmap-container', 'children')],
    Input('table', 'active_cell'),
    State('table', 'data')
)
def getActiveCell(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        if col == 'Visit date':
            row = active_cell['row']
            cellData = data[row][col]
            global img_path
            img_path = DbHandler.get_fundus_img_path(current_patient, cellData)
            code_handler = LatentCodeHandler("./models/autoencoder256-original-sgm.h5")
            img = code_handler.get_image(img_path)
            code_handler.get_gradcam_heatmap(img)
            test_base64 = base64.b64encode(open('heatmap.png', 'rb').read()).decode('ascii')

            with Image.open(img_path) as image:
                try:
                   return [html.Img(src="data:image/png;base64, " + pil_to_b64(image),
                                               style={'height':'90%', 'width':'90%'}),
                                html.Img(src='data:image/png;base64,{}'.format(test_base64),
                                         style={'height': '90%', 'width': '90%'})
                                ]
                except Exception:
                    print("Cannot convert" + img_path)
    return [html.P('No visit selected'), html.P('No visit selected')]


def pil_to_b64(im, enc_format="png", **kwargs):
    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded


@app.callback(
    [dash.dependencies.Output('prediction-container', 'children'),
     dash.dependencies.Output('va-evolution-graph', 'figure')],
    dash.dependencies.Input('predict', 'n_clicks'))
def update_output(n_clicks):
    global nb_clicks
    nb_clicks = n_clicks
    if not n_clicks:
        raise PreventUpdate
    db_handler_all = DbHandler('all', include_timestamps=False)
    all_data = db_handler_all.get_data_from_csv()
    patient_data = all_data.loc[all_data.index.get_level_values(0) == current_patient].copy()
    actual_va = patient_data["VA"]. iloc[-1]
    gen = TimeSeriesGenerator(patient_data)
    dependencies = {
        'root_mean_squared_error': Rnn.root_mean_squared_error
    }

    if len(patient_data.index) > 3:
        X, Y = gen.generate_timeseries(size=3)
        X = X.reshape(-1, 1, X.shape[1])
        if len(X) > 1:
            X = X[len(X)-2:len(X)-1]
        lstm = tf.keras.models.load_model("best_models/lstm-v3-all-res-0.99-best.h5", custom_objects=dependencies)
        lstm_prediction = lstm.predict(X, batch_size=1)[0][0]
        return [html.P(['Model: LSTM network, Performance: 99%', html.Br(),
                        'Actual VA: ' + str(actual_va.round(2)), html.Br(),
                        'Predicted VA: ' + str(lstm_prediction.round(2)), html.Br(),
                        'Prediction error: ' + str(abs(lstm_prediction - actual_va.round(2)).round(2))]),
                go.Figure(data=[go.Scatter(y=patient_data_notres['VA'], x=patient_data_notres.index.get_level_values(1), name='Time-series with previous visits'),
                                go.Scatter(y=[patient_data_notres['VA'][-1], lstm_prediction],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(months=1)], mode='lines',
                                           name='Model forecasting', showlegend=False),
                                go.Scatter(y=[lstm_prediction], x=[patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(months=1)],
                                           name='Model prediction', marker=dict(color=px.colors.qualitative.Plotly[1])),
                                go.Scatter(y=[patient_data_notres['VA'][-1], actual_va],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(months=1)],
                                           mode='lines',
                                           name='Model prediction', showlegend=False, line=dict(color=px.colors.qualitative.Plotly[2])),
                                go.Scatter(y=[actual_va],
                                           x=[patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(months=1)],
                                           name='Actual evolution', marker=dict(color=px.colors.qualitative.Plotly[2]))
                                ],
                          layout={'title': 'Visual acuity evolution'})]
    elif len(patient_data.index) == 3:
        X, Y = gen.generate_timeseries(size=2)
        X = X.reshape(-1, 1, X.shape[1])
        lstm = tf.keras.models.load_model("best_models/lstm-v2-all-res-0.98.h5", custom_objects=dependencies)
        lstm_prediction = lstm.predict(X, batch_size=1)[0][0]
        return [html.P(['Model: LSTM network, Performance: 98%', html.Br(),
                       'Actual VA: ' + str(actual_va.round(2)), html.Br(),
                       'Predicted VA: ' + str(lstm_prediction.round(2)), html.Br(),
                       'Prediction error: ' + str(abs(lstm_prediction - actual_va).round(2))]),
                go.Figure(data=[go.Scatter(y=patient_data_notres['VA'], x=patient_data_notres.index.get_level_values(1),
                                           name='Time-series with previous visits'),
                                go.Scatter(y=[patient_data_notres['VA'][-1], lstm_prediction],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data.index.get_level_values(1)[-1]], mode='lines',
                                           name='Model forecasting', showlegend=False),
                                go.Scatter(y=[lstm_prediction], x=[patient_data.index.get_level_values(1)[-1]],
                                           name='Model prediction', marker=dict(color=px.colors.qualitative.Plotly[1])),
                                go.Scatter(y=[patient_data_notres['VA'][-1], actual_va],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data.index.get_level_values(1)[-1]],
                                           mode='lines',
                                           name='Model prediction', showlegend=False,
                                           line=dict(color=px.colors.qualitative.Plotly[2])),
                                go.Scatter(y=[actual_va],
                                           x=[patient_data.index.get_level_values(1)[-1]],
                                           name='Actual evolution', marker=dict(color=px.colors.qualitative.Plotly[2]))
                                ],
                          layout={'title': 'Visual acuity evolution'})]
    elif len(patient_data.index) == 2:
        X, Y = gen.generate_timeseries(size=1)
        X = X.reshape(-1, 1, X.shape[1])
        lstm = tf.keras.models.load_model("best_models/rnn-v1-all-res-0.984.h5", custom_objects=dependencies)
        lstm_prediction = lstm.predict(X, batch_size=1)[0][0]
        return [html.P(['Model: SimpleRNN network, Performance: 98%', html.Br(),
                       'Actual VA: ' + str(actual_va.round(2)), html.Br(),
                       'Predicted VA: ' + str(lstm_prediction.round(2)), html.Br(),
                       'Prediction error: ' + str(abs(lstm_prediction - actual_va).round(2))]),
                go.Figure(data=[go.Scatter(y=patient_data_notres['VA'], x=patient_data_notres.index.get_level_values(1),
                                           name='Time-series with previous visits'),
                                go.Scatter(y=[patient_data_notres['VA'][-1], lstm_prediction],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(
                                                  months=1)], mode='lines',
                                           name='Model forecasting', showlegend=False),
                                go.Scatter(y=[lstm_prediction], x=[
                                    patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(months=1)],
                                           name='Model prediction', marker=dict(color=px.colors.qualitative.Plotly[1])),
                                go.Scatter(y=[patient_data_notres['VA'][-1], actual_va],
                                           x=[patient_data_notres.index.get_level_values(1)[-1],
                                              patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(
                                                  months=1)],
                                           mode='lines',
                                           name='Model prediction', showlegend=False,
                                           line=dict(color=px.colors.qualitative.Plotly[2])),
                                go.Scatter(y=[actual_va],
                                           x=[patient_data_notres.index.get_level_values(1)[-1] + pd.DateOffset(
                                               months=1)],
                                           name='Actual evolution', marker=dict(color=px.colors.qualitative.Plotly[2]))
                                ],
                          layout={'title': 'Visual acuity evolution'})]
    return html.P('')


@app.callback(Output('confirm', 'displayed'),
              Input('dropdown', 'value'))
def display_confirm(value):
    if value == 'Pacient 1':
        return True
    return False


if __name__ == '__main__':
    app.run_server(debug=True)

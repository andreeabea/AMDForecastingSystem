# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import os
import re
from io import BytesIO

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas as pd
from dash.dependencies import Output, Input, State
from PIL import Image

from data_processing.db_handler import DbHandler

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
            html.Label(['Select patient and eye', dcc.Dropdown(
                id='dropdown',
                options=[{"label": x, "value": x} for x in patients],
                value=patients[0]
            )], style={'font-size': '12px'}),
            html.Label(['Choose forecasting time span', dcc.Slider(
                min=1,
                max=3,
                value=1,
                marks={
                    1: {'label': '1 month'},
                    2: {'label': '2 month'},
                    3: {'label': '3 months'}
                },
                included=False
            )], style={'font-size': '12px'})
            ,
            html.Button('Predict', id='predict', style={'left': '30%', 'position': 'relative'})
        ]

        return html.Div([
            html.Div([html.H3(''),
                      html.Div(controls),  # html.Div(id="folder-files")
                      ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle',
                                'horizontal-align': 'middle'}),
            html.Div(id='dd-output-container',
                     style={'width': '67%', 'display': 'inline-block', 'vertical-align': 'middle',
                            'horizontal-align': 'middle', 'padding-top': '10px', 'padding-left': '20px'})
        ])


current_patient = None


@app.callback(
    dash.dependencies.Output('dd-output-container', 'children'),
    [dash.dependencies.Input('dropdown', 'value')])
def update_output(value):
    global current_patient
    current_patient = value
    patient_data = data.loc[data.index.get_level_values(0) == value].copy()
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
                            'horizontal-align': 'middle', 'padding-top': '10px', 'padding-left': '20px'})
    ])


@app.callback(
    Output('oct-image-container', 'children'),
    Input('table', 'active_cell'),
    State('table', 'data')
)
def getActiveCell(active_cell, data):
    if active_cell:
        col = active_cell['column_id']
        if col == 'Visit date':
            row = active_cell['row']
            cellData = data[row][col]
            img_path = DbHandler.get_fundus_img_path(current_patient, cellData)
            with Image.open(img_path) as image:
                try:
                    return html.Img(src="data:image/png;base64, " + pil_to_b64(image), style={'height':'90%', 'width':'90%'})
                except Exception:
                    print("Cannot convert" + img_path)
    return html.P('No visit selected')


def pil_to_b64(im, enc_format="png", **kwargs):
    buff = BytesIO()
    im.save(buff, format=enc_format, **kwargs)
    encoded = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded


@app.callback(Output('confirm', 'displayed'),
              Input('dropdown', 'value'))
def display_confirm(value):
    if value == 'Pacient 1':
        return True
    return False


if __name__ == '__main__':
    app.run_server(debug=True)

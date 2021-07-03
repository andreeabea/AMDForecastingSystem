# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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

    dcc.Tabs(id='tabs-example', value='tab-0', children=[
            dcc.Tab(label='View patients', value='tab-0', style=tab_style),
            dcc.Tab(label='Train models', value='tab-1', style=tab_style),
            dcc.Tab(label='Forecast visual acuities', value='tab-2', style=tab_style),
        ], style=tabs_styles),
    html.Div(id='tabs-example-content')

])


@app.callback(Output("folder-files", "children"), Input("dropdown", "value"))
def list_all_files(folder_name):
    # This is relative, but you should be able
    # able to provide the absolute path too
    file_names = os.listdir(folder_name)

    file_list = html.Ul([html.Li(file) for file in file_names])

    return file_list


@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-0':
        return html.Div([
            html.H3('Tab content 0'),
        ])
    elif tab == 'tab-1':
        algorithms = ["Linear regression", "LASSO regression", "Gradient Boosting Regression",
                      "Random Forest Regression", "Extremely Randomized Trees", "Simple RNN network",
                      "LSTM network", "GRU network"]
        data_transform = ['Time-series resampling', "Original series with timestamps as features"]
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
                      html.Div(controls), #html.Div(id="folder-files")
                      ])
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'})
    elif tab == 'tab-2':
        patients = os.listdir('data')
        patients.append('Other')
        # forecast for existing patient or add new patient
        controls = [
            html.Label(['Machine Learning algorithm', dcc.Dropdown(
                options=[{"label": x, "value": x} for x in patients],
                value=patients[0]
            )], style={'font-size': '12px'}),
            html.Button('Predict', id='submit-val', style={'left': '30%', 'position': 'relative'})
        ]
        return html.Div([
            html.Div([html.H3(''),
                      html.Div(controls),  # html.Div(id="folder-files")
                      ])
        ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'middle'})


if __name__ == '__main__':
    app.run_server(debug=True)

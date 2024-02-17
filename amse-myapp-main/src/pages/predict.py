import os

import numpy as np
import pandas as pd

import dash
from dash import html
from dash import dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate

import plotly.express as px

import joblib

from .templates.kpi import generate_kpi

dash.register_page(__name__, path='/predict')

df = pd.read_csv('../src/data/1553768847-housing.csv')


options_model = [
    {'label': model, 'value': model}
    for model in os.listdir('../src/models/')
]

dropdown1 = dcc.Dropdown(
    id='input-model-filename',
    options=options_model,
    value=options_model[0]['value']
)

slider1 = dcc.Slider(
    id='input-feature-1',
    min=df['total_rooms'].min(),
    max=df['total_rooms'].max(),
    value=df['total_rooms'].mean()
)
slider2 = dcc.Slider(
    id='input-feature-2',
    min=df['housing_median_age'].min(),
    max=df['housing_median_age'].max(),
    value=df['housing_median_age'].mean()
)

layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                html.H1('Housing Prices in California'),
                html.Div(''),
            ]
        ),
        html.Form(
            className="row g-3 mb-3",
            children=[
                html.Div(
                    className="col-2",
                    children=[
                        html.Label(
                            className="form-label",
                            children="ML Model"
                        ),
                        dropdown1
                    ],
                ),
                html.Div(
                    className="col-2",
                    children=[
                        html.Label(
                            className="form-label",
                            children="total_rooms"
                        ),
                        slider1
                    ],
                ),
                html.Div(
                    className="col-2",
                    children=[
                        html.Label(
                            className="form-label",
                            children="housing_median_age"
                        ),
                        slider2
                    ],
                ),
                html.Button(
                    id='input-predict-go',
                    className='col-1 btn btn-primary',
                    n_clicks=0,
                    children='Update',
                    type='button'
                )
            ]
        ),
        html.Div(
            id='output-predict-kpis',
            className="row",
        ),
    ]
)

@callback(
    Output(component_id='output-predict-kpis', component_property='children'),
    Input(component_id='input-predict-go', component_property='n_clicks'),
    State(component_id='input-model-filename', component_property='value'),
    State(component_id='input-feature-1', component_property='value'),
    State(component_id='input-feature-2', component_property='value')
)
def update_graph(n_clicks, model_filename, input_1, input_2):
    model = joblib.load('../src/models/{}'.format(model_filename))
    y_pred = int(model.predict(np.array([[input_1, input_2]])))
    kpi1 = generate_kpi("Your house is estimated at".format(y_pred), '${:,}'.format(y_pred))
    kpi2 = generate_kpi("total_rooms", '{:,}'.format(int(input_1)))
    kpi3 = generate_kpi("housing_median_age", '{:,}'.format(int(input_2)))

    kpis_output = [kpi1, kpi2, kpi3]
    return kpis_output

import os

import dash
from dash import html
from dash import dcc, Input, Output, State, callback
from dash.exceptions import PreventUpdate

import plotly.express as px

import pandas as pd

from .templates.kpi import generate_kpi
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objs as go
import plotly.express as px


dash.register_page(__name__, path='/sales-price')

df = pd.read_csv('../src/data/1553768847-housing.csv')

label_encoder = LabelEncoder()
df['ocean_proximity'] = label_encoder.fit_transform(df['ocean_proximity'])

df['total_bedrooms'].fillna(0, inplace=True)

cols = [
    'longitude',
    'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households',
    'median_income',
    'ocean_proximity',
]

options = [
    {'label': c, 'value': c}
    for c in cols
]
dropdown1 = dcc.Dropdown(
    id='input-y',
    options=[{'label': 'Median House Value', 'value': 'median_house_value'}],
    value='median_house_value',
    disabled=True
)
dropdown2 = dcc.Dropdown(
    id='input-x',
    options=options,
    value='housing_median_age'
)

dropdown_corr = dcc.Dropdown(
    id='input-corr',
    options=options,
    multi=True,
    value=['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms','population', 'households', 'median_income', 'ocean_proximity', 'median_house_value'],  
)

corr_heatmap = dcc.Graph(
    id='output-corr',
)

scatter_fig = px.scatter(
    df,
    x='longitude',
    y='latitude',
    color='median_house_value',
    size='median_house_value',
    opacity=0.7,
    color_continuous_scale='RdYlBu',
    labels={'median_house_value': 'Median House Value'},
    title='Map of California Housing Values',
    template='plotly',
    width=800,
    height=600
).update_layout(
    coloraxis_colorbar=dict(title='Median House Value'),
    xaxis_title='Longitude',
    yaxis_title='Latitude',
    plot_bgcolor='white'
)


labels = ["<1h ocean", "inland", "near ocean", "near bay", "island"]

fig4 = px.box(
    df,
    x='ocean_proximity',
    y='median_house_value',
    color='ocean_proximity',
    title="Boxplot of Median House Value by Ocean Proximity",
    labels={'ocean_proximity': "Ocean Proximity", 'median_house_value': 'Median House Value'}
).update_traces(marker=dict(color='rgba(0, 123, 255, 0.7)', outliercolor='rgba(219, 64, 82, 0.6)', line=dict(color='rgba(0, 123, 255, 0.7)', width=2))).update_layout(
    xaxis=dict(categoryorder='array', categoryarray=labels)  
)

layout = html.Div(
    children=[
        html.Div(
            className="row",
            children=[
                html.H1('Housing Prices in California'),
                dcc.Graph(
                    id='scatter_fig',
                    figure=scatter_fig,
                    className="col-8",
                    style={'width': '50%'}

                ),
                
                html.Div(""
                ),
            ],
        ),
        html.Form(
            className="row g-3 mb-3",
            children=[
                html.Div(
                    className="col-2",
                    children=[dropdown1],
                ),
                html.Div(
                    className="col-2",
                    children=[dropdown2],
                ),
                html.Button(
                    id='input-go',
                    className='col-1 btn btn-primary',
                    n_clicks=0,
                    children='Update',
                    type='button'
                )
            ]
        ),
        html.Div(
            id='output-kpis',
            className="row",
        ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="col-8",
                    children=[
                        html.Div(
                            className="card",
                            children=[
                                dcc.Graph(
                                    id='output-graph'
                                ),
                            ]
                        )
                    ]
                ),
                html.Div(
                    className="col-4",
                    children=[
                        html.Div(
                            className="card",
                            children=[
                                dcc.Graph(
                                    id='output-graph-hist'
                                )
                            ]
                        )
                    ]
                )
            ]
        ),
        html.Div(
                    className="col-2",
                    children=[dropdown_corr],  # Add the new dropdown for correlation matrix
                ),
        html.Div(
            className="row",
            children=[
                html.Div(
                    className="card",
                    children=[
                        html.H4("Correlation Matrix"),
                        corr_heatmap,
                    ],
                ),
            ],
        ),
                html.Div(
                    className="card",
                    children=[
                        dcc.Graph(
                            id='output-chart4',
                            figure=fig4
                        )
                    ]
                ),
            ]
        )

@callback(
    Output(component_id='output-kpis', component_property='children'),
    Output(component_id='output-graph', component_property='figure'),
    Output(component_id='output-graph-hist', component_property='figure'),
    Input(component_id='input-go', component_property='n_clicks'),
    State(component_id='input-x', component_property='value'),
    State(component_id='input-y', component_property='value')
)
def update_graph(n_clicks, input_x, input_y):
    stats = df[input_x].describe().astype(int)
    kpi1 = generate_kpi("Average ({})".format(input_x), stats.loc['mean'])
    kpi2 = generate_kpi("Stdev ({})".format(input_x), stats.loc['std'])
    graph_ols = px.scatter(df, x=input_x, y=input_y, color_discrete_sequence =['orange']*len(df)).update_layout(plot_bgcolor='white')
    graph_hist = px.histogram(df, x=input_x, nbins=50, color_discrete_sequence =['gold']*len(df)).update_layout(plot_bgcolor='white')

    kpis_output = [kpi1, kpi2]
    return kpis_output, graph_ols, graph_hist

@callback(
    Output(component_id='output-corr', component_property='figure'),
    Input(component_id='input-corr', component_property='value')
)
def update_corr_heatmap(selected_variables):
    corr_matrix = df[selected_variables].corr()
    fig_corr = px.imshow(corr_matrix, color_continuous_scale="YlOrRd").update_layout(title="Correlation Matrix", xaxis=dict(title="Features"), yaxis=dict(title="Features"), autosize=False, width=800, height=600, margin=dict(l=0, r=0, b=0, t=50))
    return fig_corr
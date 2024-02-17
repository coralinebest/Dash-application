import dash
from dash import html
import numpy as np
import pandas as pd
from .templates.kpi import generate_kpi

dash.register_page(__name__, path='/')

df = pd.read_csv('../src/data/1553768847-housing.csv')


layout = html.Div(
    html.Div(
        className="row",
        children=[
            html.H1('Housing Prices in California'),
            html.Div('Some important figures:'),
            generate_kpi("Number of houses", df.shape[0]),
            generate_kpi("Average number of people residing within a block", round(df['population'].mean())),
            generate_kpi("Housing Median Age Range", f"{df['housing_median_age'].min()} - {df['housing_median_age'].max()}"),
            generate_kpi("Average House median value", f"${round(df['median_house_value'].mean()):,.2f}"),
            html.P("In this Dash application, we leverage the California House Price dataset, sourced from the 1990 California census. Comprising 20,640 observations and encompassing 10 numerical features, the dataset offers a comprehensive view. Various pages within the application provide insights through descriptive statistics, facilitating a deeper understanding of the data. Another page features an Interactive Prediction section, enabling users to input values into the model. We offer both a linear model and an XGBoost model for analysis. Users have the autonomy to deliberately choose the model that aligns with their preferences."),
            html.Img(src='/assets/IMG_.jpg', style={'width': '20%'}),
            html.Img(src='/assets/dataset-card.jpg', style={'width': '20%'}),
            html.Img(src='/assets/téléchargement.jpg', style={'width': '20%'}),
            html.Img(src='/assets/téléchargement (1).jpg', style={'width': '20%'}),
            html.Img(src='/assets/ee.jpg', style={'width': '20%'}),
        ]
    )
)
import dash
from dash import Dash, html
import dash_bootstrap_components as dbc

from pages.templates.sidebar import generate_sidebar

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions= "True"
)

app.layout = html.Div(
    id="wrapper",
    children=[
        generate_sidebar(pages=dash.page_registry.values()),
        html.Div(
            id="content-wrapper",
            className="d-flex flex-column",
            children=[
                html.Div(
                    id='content',
                    children=[
                        html.Div(
                            className="container-fluid",
                            children=[
                                dash.page_container
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

if __name__ == '__main__':
    app.run(debug=True)

import dash
from dash import dcc, html, Input, Output

from pages import about, home

app = dash.Dash(__name__)
application = app.server

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Ul(children=[
        html.Li(html.A("Home", href="/"), id="home_href"),
        html.Li(html.A("About", href="/about", id="about_href")),
        # html.Div(html.A(html.Img(src=app.get_asset_url("github.svg"), alt="github"),
        #                href="https://github.com/maze508/ASDI_Hackathon_2022", target="_blank"), className="github")
    ], className="nav-bar"),
    html.Div(id="page-content", className="main"),
    html.Div(
        html.A(html.Img(src=app.get_asset_url("github.svg"), alt="github"),
               href="https://github.com/maze508/ASDI_Hackathon_2022", target="_blank"), className="github"
    )
])


@app.callback([Output("page-content", "children"),
               Output("home_href", "className"),
               Output("about_href", "className")],
              [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/about":
        return [about.layout, "", "active"]
    else:
        return [home.layout, "active", ""]


if __name__ == "__main__":
    application.run(debug=True, port=8000)

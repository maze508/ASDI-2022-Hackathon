from dash import html, dcc

with open(r"README.md", "r") as r:
    content = r.read()

layout = html.Div(children=[
    html.H1(children='About', className="title"),

    dcc.Markdown(content)
], className="markdown")

import pandas as pd
from dash import html, dcc
import plotly.graph_objects as go
from utils.connect_to_s3 import s3

df = pd.read_csv(s3.Bucket('adsi-aws-bucket').Object("model-output.csv").get()['Body'], index_col=0)

fig = go.Figure(data=go.Choropleth(
    locations=df['stateCode'],
    z=df['ensemble_all_3_preds'].astype(float),
    locationmode='USA-states',
    colorscale='Reds',
    autocolorscale=False,
))

fig.update_layout(
    geo=dict(
        scope='usa',
    ),
    margin=dict(l=3, r=3, t=15, b=0),
    plot_bgcolor="beige",
    paper_bgcolor="beige",
)

layout = html.Div(children=[
    html.Div(children=[
        html.H1(children='Eco-Protector'),

        html.H2(children=["Predicting the spread dynamics and probability of invasion of wineberries ",
                          html.I("(Rubus phoenicolasius)"),
                          " by state in USA"]),

        html.H3(children="Amazon Sustainability Data Initiative (ASDI) Global Hackathon"),

        html.P(children=["Done by: ",
                         html.A(children="Ma Ze Xuan", href="https://github.com/maze508", target="_blank"),
                         " and ",
                         html.A(children="Tan Guan Quan", href="https://github.com/guanquann", target="_blank")]),

    ], className="title"),

    html.Div(dcc.Graph(figure=fig), id="choropleth_chart"),

    # html.Video(
    #     controls=True,
    #     id='movie_player',
    #     src=r"C:\Users\tguan\Videos\Captures\localhost_1313_OpenBBTerminal_ - Google Chrome 2022-08-18 19-56-21.mp4",
    #     autoPlay=True
    # ),
])

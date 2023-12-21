from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from categories import *
import pandas as pd

cat_df = pd.DataFrame({
    'subject': subjects,
    'sub_category': [subj_to_subcat[s] for s in subjects],
    'category': [subj_to_cat[s] for s in subjects],
})
# }).sort_values(by=['category', 'sub_category', 'subject'])

label_style = {
    'margin-top': '10px',
    'font-weight': 'bold',
}

layer_label_style = {
    'font-weight': 'bold',
    # 'margin-top': '5px',
    # 'margin-right': '10px',
}

checkbox_style = {
    'margin-right': '5px'
}

controls = dbc.Card([
    html.Div([
        dbc.Label("Layer", style=layer_label_style),
        dcc.Dropdown(
            id='layer-id-dropdown',
            options=list(str(x) for x in range(32)),
            value=str(31),
            clearable=False
        ),
        # ], style={'display': 'inline-flex'}
        ],
    ),
    # html.Div([
    #     dbc.Label("Sub-Categories", style=label_style),
    #     dcc.Checklist(
    #         id='subcats-id-radio',
    #         options=subcat_list,
    #         value=subcat_list,
    #         labelStyle={'display': 'block'},
    #         inputStyle=checkbox_style,
    #     ),
    # ]),
    ],
    body=True
)

pca_text = 'Each data point '
svm_text = 'Each data point '
overview_text = dcc.Markdown('''
Recently, Mistral AI's Mixture-of-Experts model ([Mixtral MoE](https://mistral.ai/news/mixtral-of-experts/))
shows impressive performance
despite only requiring forward capacity of a 13B model. The model dynamically decides which
experts to use for each token, where an expert is the FFN or MLP layer in a traditional
transformer model. Specifically, there are 8 different expert MLPs at each layer and 2 of them
are picked by a module called router to be applied to the embedding of a given token. Given that
the model can now choose which MLP layer to use for each token, unlike attention modules, it is
reasonable to believe that the experts are, well, _experts_ on different topics. This project
attempts to visualize whether this actually happens.

The idea is to forward a sentences/paragraph/tokens from a variety of topics and calculate how many times
each expert was picked during the forward pass. If the experts do specialize for certain topics,
then this 8 dimensional feature vector of expert frequencies should contain all the information needed
to correctly predict the topic of a given paragraph. Alternatively, paragraphs coming from different
topics should be linearly separable in the space of expert picking frequencies. We use the popular
MMLU dataset with 57 different topics here.
''')


graphs = html.Div(
    dbc.Accordion([
        dbc.AccordionItem([
            html.P(overview_text),
        ], title="Overview"),
        dbc.AccordionItem([
            # html.P(pca_text),
            dcc.Graph(figure={}, id='pca-fig')
        ], title="PCA Projection"),
        dbc.AccordionItem([
            # html.P(svm_text),
            dcc.Graph(figure={}, id='clf-fig')
        ], title="SVM Weights"),
    ],
    )
)

app_layout = dbc.Container(
    [
        dbc.Row([
            html.H1('Visualizing Expert Firing Frequencies in Mixtral MoE'),
            html.Hr(),
        ]),
        dbc.Row(
            [
                dbc.Col(controls, md=2),
                dbc.Col(graphs, md=10),
            ],
        ),
    ],
    fluid=True,
)


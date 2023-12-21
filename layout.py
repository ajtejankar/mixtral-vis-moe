from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
from categories import *
import pandas as pd

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
    ]),
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
    className='sticky-top',
    body=True
)

overview_text = dcc.Markdown('''
Recently, Mistral AI's Mixture-of-Experts model
([Mixtral MoE](https://mistral.ai/news/mixtral-of-experts/)) shows impressive performance
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
MMLU dataset with 57 different topics/subjects here. The subjects are grouped in 17 different
sub-categories and 4 different broad level categories. However, before we attempt to classify the
paragraphs, let us try to visualize them.
''')

pca_text = dcc.Markdown('''
While 8 dimensions is relatively small compared to the typical size of embeddings in Deep Learning,
visualizing the data directly is still not possible in it. Hence, we reduce these 8 dimensions to 2
with PCA. Given that there is a lot of overlap, the plots are separated according to their broad
categories. The plots are interactive and has following features. 1) Hovering on each point shows
information related to it. 2) You can click on legend entries to disable its points. 3) You can double
click on legend entries to only keep the plot for that entry. We can see that in the stem category,
math related topics are on the positive side of the y axis while biology is on the negative side.
''')

svm_text = dcc.Markdown('''
Now that we have some intuition about how the expert frequencies encode information about its topic,
let's explore how truly separated these topics are. To do so, we train an SVM which is linear classifier.
If the accuracy of the SVM is high, then it means that paragraphs from each topic can be separated from
others by a simple plane in 8 dimensions, and we see that the accuracy is indeed high. Further, we
can explore if certain experts are important for certain subjects with the following heatmap.
''')


graphs = html.Div(
    dbc.Accordion([
        dbc.AccordionItem([
            html.P(overview_text),
        ], title="Overview", item_id='item-0'),
        dbc.AccordionItem([
            html.P(pca_text),
            dcc.Graph(figure={}, id='pca-fig')
        ], title="PCA Projection", item_id='item-1'),
        dbc.AccordionItem([
            html.P(svm_text),
            dcc.Graph(figure={}, id='clf-fig')
        ], title="SVM Weights", item_id='item-2'),
    ],
    always_open=True,
    active_item=['item-0', 'item-1', 'item-2']
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


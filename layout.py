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
([Mixtral MoE](https://mistral.ai/news/mixtral-of-experts/)) shows impressive performance,
comparable to a 70B model, despite only requiring forward pass capacity of a 13B model.
The model dynamically decides which
expert to use for each token, where an expert is an FFN or an MLP layer in a traditional
transformer model. Specifically, there are 8 different expert MLPs at each layer out of which 2
are picked by a module called router for a given token.
Given that the model can now choose which MLP layers to use for each token, unlike attention modules, it is
reasonable to believe that the experts are, well, _experts_ on different topics. This project
attempts to visualize whether this actually happens.

The idea is to forward a few sentences/paragraphs from different topics and calculate how many times
each expert was picked during the forward pass. If the experts do specialize for certain topics,
then this 8 dimensional feature vector of expert frequencies should contain all the information needed
to correctly predict the topic of a given paragraph. Alternatively, paragraphs coming from different
topics should reside far away from each other in the space of expert frequencies. Doing this needs a dataset
with topic-wise annotations. So, we use the popular MMLU dataset with 57 different topics/subjects.
The subjects are grouped in 17 different sub-categories and 4 different broad level categories.
However, before we attempt to classify the paragraphs, let us try to visualize them.
''')

pca_text = dcc.Markdown('''
While 8 dimensions is relatively small compared to the typical dimension size of embeddings in Deep Learning,
visualizing the data in it is still difficult. Hence, we reduce these 8 dimensions to 2
with PCA. Given that there is a lot of overlap, the plots are separated according to their broad
categories. The plots are interactive and have following features. 1) Hovering on each point shows
information related to it. 2) You can click on legend entries to disable its points. 3) You can double
click on legend entries to only keep the plot for that entry. For the last layer and stem category,
we can see that math related topics are on one side of the y axis while biology is on the other side.
Changing the layer results in a different arrangement of topics which indicates how each layer is
learning topic-wise information.
''')

svm_text = dcc.Markdown('''
Now that we have some intuition about how the expert frequencies encode topic information,
let's explore how truly separated these topics are. To do so, we train an SVM, a linear classifier, to separate
input data by topics. If linear separation is possible, then the classifier should have a high accuracy, and we
indeed see this happening. For instance, the SVM classifier gets about 95% accuracy at the last layer. What
else can the SVM tell us? Well, we can look at the classifier weights for each subject and try to understand
which experts are important/unimportant for it. Blue means the expert fires less frequently than average while
yellow means the expert fires more. Another surprising finding, at least for me, is that the classifier accuracy
stays high for almost all layers. I had expected the accuracy to be bad for initial layers but to progressively
get better at later layers. Since this doesn't happen and the accuracy is always high, we can only conclude
that the topic-wise segregation happens from the very first layer. Quite interesting if you ask me.
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
            dbc.Col(html.H1('Visualizing Expert Firing Frequencies in Mixtral MoE'), md=10),
            dbc.Col(
                html.A('GitHub',
                    href='https://github.com/ajtejankar/mixtral-vis-moe',
                    target='_blank',
                ),
                className='text-center',
                md=2
            ),
            html.Hr(),
        ],
        className='align-items-center',
        ),
        dbc.Row([
            dbc.Col(controls, md=2),
            dbc.Col(graphs, md=10),
        ],
        style={'margin-bottom': '50px !important'},
        ),
        dbc.Row([
            html.Footer([
                '© Copyright 2023 ',
                html.A('Ajinkya Tejankar', href='https://ajtejankar.github.io'),
            ],
            className='footer bg-light py-1 text-center small',
            style={'position': 'fixed', 'bottom': '0'},
            )
        ],
        className='mt-5'),
    ],
    fluid=True,
)


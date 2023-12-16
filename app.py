#!/usr/bin/env python
# coding: utf-8

import os
import shlex
import argparse
from collections import Counter, defaultdict
from functools import partial

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, html, dcc, callback, Output, Input

from tqdm.auto import tqdm
from categories import subcategories, categories

choices = ["A", "B", "C", "D"]

with open('subjects.txt', 'r') as f:
    subjects = [line.strip() for line in f.readlines()]

subcat_to_subj = defaultdict(list)
for i, (subj, subcat) in enumerate(subcategories.items()):
    subcat_to_subj[subcat[0]].append(i)
subj_to_subcat = {subjects[v]: k for k, vv in subcat_to_subj.items() for v in vv}
subcat_list = sorted(subcat_to_subj.keys())

cat_to_subj = defaultdict(list)
for i, (cat, subcats) in enumerate(categories.items()):
    cat_name = cat.split()[0].lower().replace('social', 'social sciences')
    for subcat in subcats:
        cat_to_subj[cat_name].extend(subcat_to_subj[subcat])
subj_to_cat = {subjects[v]: k for k, vv in cat_to_subj.items() for v in vv}
cat_list = sorted(cat_to_subj.keys())

subj_to_frac_vecs = np.load('router_logit_frac_vecs.npy').item()

def get_data(layer_id=31, label_type=0):
    X = []
    y = []
    label_lists = [subjects, subcat_list, cat_list]

    for subj in subjects:
        labels = [subj, subj_to_subcat[subj], subj_to_cat[subj]]
        X.append(subj_to_frac_vecs[subj][:, layer_id])
        y.extend([label_lists[label_type].index(labels[label_type])]*len(X[-1]))

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    y_names = np.array(label_lists[label_type])
    # X_reduced = PCA(n_components=2).fit_transform(X)
    return X, y, y_names


def run_classif(layer_id=31, label_type=0):
    accs = []
    for fold in range(10):
        X, y, y_names = get_data(layer_id, label_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
        clf = make_pipeline(StandardScaler(), LinearSVC(dual="auto", tol=1e-5))
        clf.fit(X_train, y_train)
        acc = clf.score(X_test, y_test)
        # print(f'Fold: {fold:2d} | Acc: {acc:.2f}')
        accs.append(acc)
    accs = np.array(accs)
    # print(f'\nAll Folds Accuracy: Mean {accs.mean():.2f} | Var {accs.var():.4f}')
    return clf, accs


# 1. Visualize Classifier Weights

layer_id = 31
label_type = 0

# 3. Add figures to the page
@callback(
    Output(component_id='clf-fig', component_property='figure'),
    Input(component_id='layer-id-dropdown', component_property='value')
)
def update_clf(layer_id):
    layer_id = int(layer_id)

    clf, _ = run_classif(layer_id, label_type)
    clf.steps[1][1].coef_.shape
    clf_fig = px.imshow(clf.steps[1][1].coef_.T)

    return clf_fig

# 3. Add figures to the page
@callback(
    Output(component_id='pca-fig', component_property='figure'),
    Input(component_id='layer-id-dropdown', component_property='value')
)
def update_pca(layer_id):
    layer_id = int(layer_id)
    _, y1, y1_names = get_data(layer_id, 1)
    _, y0, y0_names = get_data(layer_id, 0)
    X, y2, y2_names = get_data(layer_id, 2)

    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)


    df = pd.DataFrame({'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'category': y2_names[y2], 'sub_category': y1_names[y1], 'subject': y0_names[y0]})
    pca_fig = px.scatter(df, x='x', y='y', facet_col='category', color='sub_category', hover_data='subject')
    pca_fig.update_layout(width=1200, height=400)

    return pca_fig

app = Dash(__name__)
server = app.server
app.layout = html.Div([
    dcc.Dropdown(list(str(x) for x in range(32)), str(layer_id), id='layer-id-dropdown'),
    # dcc.Dropdown(list(str(x) for x in range(3)), str(label_type), id='label-type-dropdown')
    html.Div(children=f'Visualization of Linear SVM Classifier Parameters: Layer {layer_id}'),
    dcc.Graph(figure={}, id='clf-fig'),
    html.Div(children=f'Visualization of PCA: Layer {layer_id}'),
    dcc.Graph(figure={}, id='pca-fig'),
])


if __name__ == '__main__':
    app.run(debug=True)


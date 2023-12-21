#!/usr/bin/env python
# coding: utf-8

import os
import shlex
import argparse
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
from layout import app_layout
from dash import Dash, callback, Output, Input
import dash_bootstrap_components as dbc

from tqdm.auto import tqdm
from categories import *


subj_to_frac_vecs = {}
for file_name in os.listdir('frac_vecs'):
    arr = np.load(os.path.join('frac_vecs', file_name))
    subj = file_name.split('.npy')[0]
    subj_to_frac_vecs[subj] = arr

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
    clf_fig.update_layout(coloraxis=dict(colorbar=dict(orientation='h', y=-0.01)))
    
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

    df = pd.DataFrame({
        'x': X_reduced[:, 0],
        'y': X_reduced[:, 1],
        'category': y2_names[y2],
        'category_idx': y2,
        'sub_category': y1_names[y1],
        'sub_category_idx': y1,
        'subject': y0_names[y0],
        'subject_idx': y0,
    }).sort_values(by=['category_idx', 'sub_category_idx', 'subject_idx'])

    # df['show_subcat'] = df['category'] + ' - ' + df['sub_category']
    df['show_subcat'] =  df['sub_category'] + ' (' + df['category'] + ')'


    pca_fig = px.scatter(df,
            x='x', y='y',
            facet_col='category', color='show_subcat', hover_data='subject')
            # color_discrete_sequence=px.colors.qualitative.G10)

    pca_fig.update_layout(legend=dict(
        title='Sub Categories',
        entrywidth=200,
        orientation="h",
        yanchor="bottom",
        y=1.10,
        # xanchor="right",
        # x=0.0
    ))

    return pca_fig


app = Dash(
    'Visualize Mixtral MoE',
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

server = app.server
app.layout = app_layout


if __name__ == '__main__':
    app.run(debug=True)


import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import argparse
import scipy.sparse as ssp
import pickle
from utils.data_utils import *
from utils.builder import PandasGraphBuilder

if __name__ == '__main__':
    articles = pd.read_csv('data/articles.csv')
    transactions = pd.read_csv('data/transactions_train.csv')[['customer_id', 'article_id', 't_dat']]
    customers = pd.DataFrame({'customer_id': transactions['customer_id'].drop_duplicates()})
    
    articles = articles.drop(columns = ['product_type_name', 'graphical_appearance_name', 'colour_group_name', 'perceived_colour_value_name',
                            'perceived_colour_master_name', 'index_name', 'index_group_name', 'section_name', 
                            'garment_group_name', 'prod_name', 'department_name', 'detail_desc'])

    for col in ['index_code', 'product_group_name']:
        number = LabelEncoder()
        articles[col] = number.fit_transform(articles[col].astype('str'))

    for col in ['product_type_no', 'graphical_appearance_no', 'colour_group_code', 'perceived_colour_value_id', 'perceived_colour_master_id']:
        number = LabelEncoder()
        articles[col] = number.fit_transform(articles[col].astype('int64'))


    transactions['t_dat'] = transactions['t_dat'].values.astype('datetime64[s]').astype('int64')

    graph_builder = PandasGraphBuilder()
    graph_builder.add_entities(customers, 'customer_id', 'customer')
    graph_builder.add_entities(articles, 'article_id', 'article')
    graph_builder.add_binary_relations(transactions, 'customer_id', 'article_id', 'bought')
    graph_builder.add_binary_relations(transactions, 'article_id', 'customer_id', 'bought-by')
    g = graph_builder.build()

    for col in articles.columns:
        if col == 'article_id':
            continue
        else:
            g.nodes['article'].data[col] = torch.LongTensor(articles[col].values)
    g.edges['bought'].data['t_dat'] = torch.LongTensor(transactions['t_dat'].values)
    g.edges['bought-by'].data['t_dat'] = torch.LongTensor(transactions['t_dat'].values)

    train_indices, val_indices, test_indices = np.load('data/train_indices.npy'), np.load('data/val_indices.npy'), np.load('data/test_indices.npy')

    train_g = build_train_graph(g, train_indices, 'customer', 'article', 'bought', 'bought-by')
    val_matrix, test_matrix = build_val_test_matrix(
        g, val_indices, test_indices, 'customer', 'article', 'bought')

    dataset = {
            'train-graph': g,
            'val-matrix': val_matrix,
            'test-matrix': test_matrix,
            'item-texts': {},
            'item-images': None,
            'user-type': 'customer',
            'item-type': 'article',
            'user-to-item-type': 'bought',
            'item-to-user-type': 'bought-by',
            'timestamp-edge-column': 't_dat'}

    with open('data/full_graph', 'wb') as f:
        pickle.dump(dataset, f)
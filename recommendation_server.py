# This file runs a server using python flask librabry.
# This just loads  our model, acepts reuests from the user and based on the playlist submitted by user returns a list recommeded songs and their info.
# This is code isnsimilar to the one in the light GCN model.
# if you are trying to run this make sure you have the model pickel file and the front end code is making request at correct endpoint.



import torch
torch.cuda.is_available()
print(torch.zeros(1).cuda())

import torch_geometric
import torch_geometric.nn as pyg_nn
from torch_sparse import SparseTensor, matmul
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
print(torch_geometric.__version__)

import torch_scatter
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.utils as pyg_utils

from torch import Tensor
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType, OptTensor)

from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

import networkx as nx
from deepsnap.hetero_graph import HeteroGraph
import copy
from copy import deepcopy
import pickle

import deepsnap
from deepsnap.graph import Graph
from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from sklearn.metrics import f1_score, roc_auc_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import json
from pathlib import Path as Data_Path
from os import listdir
from os.path import isfile, join
from itertools import combinations
from tqdm.notebook import tqdm
from flask import Flask, request

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

import graph_tool.all as gt
import json
print("graph-tool version: {}".format(gt.__version__.split(' ')[0]))
import pickle


app = Flask(__name__)


input_dim = 0

with open("/home/asa489/Downloads/updated_fixed_graph.pickle", "rb") as f:
    g_nx = pickle.load(f)
mapping = {old_label: new_label for new_label, old_label in enumerate(sorted(g_nx.nodes()))}

# reindex the nodes in the graph
g_nx = nx.relabel_nodes(g_nx, mapping)

class LGCN_MessagePassing(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize = True,
                 bias = False, **kwargs):  
        super(LGCN_MessagePassing, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

    def forward(self, x, edge_index, size = None):
        out = self.propagate(edge_index, x=(x, x))
        return out

    def message(self, x_j):
        out = x_j
        return out

    def aggregate(self, inputs, index, dim_size = None):
        node_dim = self.node_dim
        out = torch_scatter.scatter(inputs, index, dim=node_dim, reduce='mean')
        return out
        
# This is creates a light gcn along with muliple light gcn message passing and aggregation layers.
 #  Here after calculating final embeddings of nodes after passing it through multiple LIGHT gcn message passing and aggregation layers, we calculate dot product of embeddings of pair of nodes that are defined in edge_label_index, which contains both negative and positive edges.
 # We return this score for each entry in edge_label_index,  and the ground truth that is label for that partiicular pair is stored in edge_label. Both of these are then fed into our loss function so that it can learn to predict whether there exists a edge betweenpair of nodes or not. 
    
class LightGCN(torch.nn.Module):
    def __init__(self, train_data, num_layers, emb_size=16):
        super(LightGCN, self).__init__()
        self.convs = nn.ModuleList()
        for l in range(num_layers):
            self.convs.append(LGCN_MessagePassing(0, 0))
        total_nodes = train_data.node_label_index.size()[0]
        self.embeddings = nn.Embedding(total_nodes, emb_size)
        self.loss_fn = nn.BCELoss()
        self.num_layers = num_layers

    def forward(self, data):
        edge_index, edge_label_index, node_label_index = data.edge_index, data.edge_label_index, data.node_label_index
        x = self.embeddings(node_label_index)
        final_emb = x
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            final_emb += x
        final_emb /= self.num_layers
        n1 = torch.index_select(final_emb, 0, edge_label_index[0,:].long())
        n2 = torch.index_select(final_emb, 0, edge_label_index[1,:].long())
        out = torch.sum(n1 * n2, dim=-1) 
        pred = torch.sigmoid(out)
        return torch.flatten(pred)

    def loss(self, pred, label):
        return self.loss_fn(pred, label)

args = {
    'device' : 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_layers' : 4,
    'emb_size' : 32,
    'weight_decay': 1e-6,
    'lr': 0.2,
    'epochs': 50
}

    



def gettrackname(uri_list):
    # Replace the values below with your own Spotify API credentials
    client_id = 'd5566a60926740f3a8070889731a2d21'
    client_secret = 'eb5fc0638a1241c3a611186ff8d167e3'

    # Initialize the Spotify API client
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    counter=1

    info = []
    for uri in uri_list:
        # Use the track method to get information about the track
        track_info = sp.track(uri)

        # Get the track name from the track information
        track_name = track_info['name']
        track_info = sp.track(uri)
        album=track_info['album']['uri']
        album_info = sp.album(album)
        image_uri=album_info['images'][0]['url']
        preview_url = track_info['preview_url']
        external_url = track_info['external_urls']['spotify']
        # Get the artist name from the track information
        artist_name = track_info['artists'][0]['name']
        info.append((track_name, artist_name, image_uri, preview_url, external_url))
        if(counter>10):
            break
        counter=counter+1
        
    return info


@app.after_request
def add_header(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'GET':
        name = request.args.get('name')
    elif request.method == 'POST':
        name = request.json.get('name')
    else:
        name = 'Throwbacks'
    print('Loading graph')
    ds_graph = Graph(g_nx)
    print('Loading graph done:', ds_graph.edge_label_index)
    node_dict = {v: k for k, v in nx.get_node_attributes(g_nx,'uri').items() if v}
    pl_dict={}
    for k, v in nx.get_node_attributes(g_nx,'name').items():
        if(v):
            if(v in pl_dict):
                continue
            else:
                pl_dict[v]=k

    track_nodes=[n for n, d in g_nx.nodes(data=True) if d.get("uri") != ""]
    playlist_exist=[]

    playlist_node_label=[]
    playlist_edge_label=[[],[]]
    for u,v in g_nx.edges(pl_dict[name]):
        playlist_exist.append(v)

    for  x in track_nodes:
        if(x in playlist_exist):
            continue
        else:
            playlist_edge_label[0].append(pl_dict[name])
            playlist_edge_label[1].append(x)
            playlist_node_label.append(0)
    
    node_label_tensor=torch.tensor(playlist_node_label)
    edge_label_tensor=torch.tensor(playlist_edge_label)
    ds_graph.edge_label_index=edge_label_tensor
    print('Loading graph done 2:', ds_graph.edge_label_index)
    newModel = LightGCN(ds_graph, args['num_layers'], emb_size=32)
    newModel.load_state_dict(torch.load('pm_latest.pt'))
    pred=newModel(ds_graph)
    print('Pred before filtering',pred)
    ls_trUri=[]
    for x in ds_graph.edge_label_index[1]:
        ls_trUri.append(g_nx.nodes[x.item()]['uri'])
    df_dict={'pred':pred.detach().numpy(),'Track_node_label':ds_graph.edge_label_index[1].detach().numpy(),'Track_Uri':ls_trUri}
    df_rec_list=pd.DataFrame(df_dict)
    df_lessthan = df_rec_list[df_rec_list['pred'] > 0.95]
    uri_list=list(df_lessthan['Track_Uri'])
    print(df_lessthan['Track_Uri'].head(30))
    return(gettrackname(uri_list))

app.run()

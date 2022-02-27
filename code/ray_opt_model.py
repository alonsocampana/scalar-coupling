#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch_geometric.data import InMemoryDataset, download_url, Data, Batch
from torch import nn
from torch import functional as F
import os
import pandas as pd
import numpy as np
import pickle
import itertools
import jax
from jax import numpy as jnp
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import MinMaxScaler
import mendeleev
from sklearn.datasets import fetch_california_housing
import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
import os
import gc


# In[2]:


from mendeleev import element


# In[3]:


def get_metaedge_features(incident_edges, coords):
    head2tail = []
    triads_ht = []
    head2head = []
    triads_hh = []
    tail2tail = []
    triads_tt = []
    for x, i in enumerate(incident_edges):
        for y, j in enumerate(incident_edges):
            if i[1] == j[0]:
                head2tail.append((x, y))
                triads_ht.append((i[0], i[1], j[1]))
            if i[0] == j[0] and x != y:
                tail2tail.append((x, y))
                triads_hh.append((i[1], i[0], j[1]))
            if i[1] == j[1] and x != y:
                head2head.append((x, y))
                triads_tt.append((i[0], i[1], j[0]))

    head2tail = np.array(head2tail)
    head2head = np.array(head2head)
    tail2tail = np.array(tail2tail)
    all_triads = triads_ht + triads_hh + triads_tt
    non_empty = [array for array in [head2tail, head2head, tail2tail] if len(array) > 0]
    metaedges = np.concatenate(non_empty)
    angles = []
    for triad in all_triads:
        angles.append(get_angle(coords, triad))
    return metaedges, np.array(angles)

def get_angle(array, triad):
    """
    Array of distances to angles between edges
    """
    i, j, k = triad
    a =  array[i] - array[j]
    b =  array[k] - array[j]
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    return a@b


# In[4]:


def get_edge_features(coords, dipole_moment, eps = 0.000001):
    norm_dipole = np.linalg.norm(dipole_moment) # get the norm to normalize the vector and find angles
    distmat = squareform(pdist(coords)) # get dist_mat to find edges
    np.fill_diagonal(distmat, np.nan) # fill to avoid ranking problem
    rankings = distmat.argsort(axis=0) # order distance matrix to get n-neighborhood of each edge
    G = nx.from_numpy_matrix(distmat, create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G)) # remove to avoid self edges
    edgelist = nx.to_edgelist(G)
    edgelist, edge_features = edge_list_to_numpy(edgelist) #get edges and distances
    G = nx.from_numpy_matrix(rankings+1, create_using=nx.DiGraph())
    G.remove_edges_from(nx.selfloop_edges(G))
    edgelist2 = nx.to_edgelist(G)
    edgelist2, edge_rankings = edge_list_to_numpy(edgelist2) # get rankings
    edge_rankings = edge_rankings - 1
    n_edges = edgelist.shape[0]
    coords_nodes = coords[edgelist] #select nodes in edges
    vectors_edges = coords_nodes.transpose(0, 2, 1)[:, :, 1] - coords_nodes.transpose(0, 2, 1)[:, :, 0] # get vector of each edge
    vectors_edges_normalized = vectors_edges/(np.linalg.norm(vectors_edges, axis=1) + eps)[:,None] 
    dipole_moment_normalized = dipole_moment/(norm_dipole + eps)
    angles_dipole_moment = np.dot(vectors_edges_normalized, dipole_moment) # do dot product to find angles
    ranks = np.zeros([n_edges, 6])
    edge_rankings[edge_rankings > 4] = 5 # replace to impose same dimensionality in all molecules
    ranks[np.arange(n_edges), edge_rankings] = 1
    edge_features = np.concatenate([edge_features[:,None], ranks, angles_dipole_moment[:,None]], axis=1) # concatenate all features
    return edgelist, edge_features


# In[5]:


def edge_list_to_numpy(edgelist):
    tail = []
    head = []
    weight = []
    for edge in list(edgelist):
        tail.append(edge[0])
        head.append(edge[1])
        weight.append(edge[2]["weight"])
    tail = np.array(tail)[:,None]
    head = np.array(head)[:,None]
    weight = np.array(weight)
    return np.concatenate([tail, head], axis=1), weight

def get_dihedral(coords, indices):
    a = coords[indices[0],:] - coords[indices[1],:]
    b = coords[indices[2],:] - coords[indices[1],:]
    return a@b/(np.linalg.norm(b) * np.linalg.norm(a))


# In[6]:


class SCDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, training= True):
        super().__init__(root, transform, pre_transform)
        self.filenames = pd.read_csv("raw/processed_names.csv")
        self.charges = pd.read_csv("raw/mulliken_charges.csv")
        self.magnetic_shieldings = pd.read_csv("raw/magnetic_shielding_tensors.csv")
        self.dipole_moments = pd.read_csv("raw/dipole_moments.csv")
        self.potential_energy = pd.read_csv("raw/potential_energy.csv")
        self.target = pd.read_csv("raw/train.csv")
        self.structures = pd.read_csv("raw/structures.csv")
        self.molecule_names = molecule_names = np.unique(self.potential_energy["molecule_name"])
        if training:
            self.training_mask = np.loadtxt("./raw/training_mask2.csv").astype(bool)
            self.molecule_names = self.molecule_names[self.training_mask]
            
    def len(self) -> int:
        return len(self.molecule_names)
    def standarize(self):
        mms = MinMaxScaler([-4, 4])
        self.charges["mulliken_charge"] = mms.fit_transform(self.charges[["mulliken_charge"]]).squeeze()
        self.magnetic_shieldings.iloc[:, 2:] = mms.fit_transform(self.magnetic_shieldings.iloc[:, 2:])
        self.dipole_moments.iloc[:, 1:] = mms.fit_transform(self.dipole_moments.iloc[:, 1:])
        self.potential_energy["potential_energy"] = mms.fit_transform(self.potential_energy[["potential_energy"]]).squeeze()

    def preprocess(self, k = None):
        charges = self.charges
        magnetic_shieldings = self.magnetic_shieldings
        dipole_moments = self.dipole_moments
        potential_energy = self.potential_energy
        molecule_names = self.molecule_names
        target = self.target
        structures = self.structures
        dfs = [charges, magnetic_shieldings, dipole_moments, potential_energy, target, structures]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].set_index("molecule_name", drop=True)
        charges, magnetic_shieldings, dipole_moments, potential_energy, target, structures = dfs
        atoms = structures["atom"].unique()
        atoms_id = {atoms[i]:i for i in range(len(atoms))}
        training_mask = []
        for x, name in enumerate(list(molecule_names)):
            any_training_edges = True
            coords = structures.loc[name][["x", "y", "z"]].to_numpy()
            n_nodes = coords.shape[0]
            print("{}/{}".format(x + 1, len(molecule_names)), end = "\r")
            # adj_mat
            atom_types = structures.loc[name]["atom"].replace(atoms_id).to_numpy()
            atom_onehot = np.zeros([n_nodes, len(atoms)])
            atom_onehot[np.arange(0, n_nodes), atom_types] = 1
            charge = charges.loc[name]["mulliken_charge"].to_numpy()
            shieldings = magnetic_shieldings.loc[name].iloc[:, 2:].to_numpy()
            node_features = np.concatenate([charge[:, None], shieldings, atom_onehot, coords], axis=1)
            with open("./processed2/{}_node_attr.csv".format(name), "wb") as f:
                np.savetxt(f, node_features)
            try:
                edges_target = target.loc[[name]]
                training_mask.append(True)
            except KeyError:
                training_mask.append(False)
                any_training_edges = False
                
            if any_training_edges:
                edges_target["type"] = edges_target["type"].replace(edge_to_int).astype(np.int64)
                scalar_coupling = edges_target.loc[:, ["atom_index_0", "atom_index_1","type","scalar_coupling_constant"]].to_numpy()
            else:
                scalar_coupling = np.array([-1, -1, -1, 0])
            with open("./processed2/{}_target.csv".format(name), "wb") as f:
                np.savetxt(f, scalar_coupling)
            # Graph features
            dipole_moment = dipole_moments.loc[name]
            norm_dipole = np.array([np.linalg.norm(dipole_moment)])
            potential = potential_energy.loc[name]
            graph_features = (np.concatenate([dipole_moment, norm_dipole, potential, np.array([n_nodes]), atom_onehot.sum(axis=0)]))
            with open("./processed2/{}_graph_features.csv".format(name), "wb") as f:
                np.savetxt(f, graph_features)
            # edge_features
            edgelist, edge_attr = get_edge_features(coords, dipole_moment)
            # metaedge
            second_nhood = edgelist[edge_attr[:, 2] == 1]
            second_nhood = np.unique(np.sort(second_nhood, axis=1), axis=0)
            first_nhood = edgelist[edge_attr[:, 1] == 1]
            first_nhood = np.unique(np.sort(first_nhood, axis=1), axis=0)
            is_in_first = [x in first_nhood.tolist() for x in second_nhood.tolist()]
            second_nhood = second_nhood[~np.array(is_in_first)]
            meta_edges = []
            normal_edges = []
            for i in first_nhood:
                for j in first_nhood:
                    meta_edge = np.unique(np.concatenate([i, j]))
                    if len(meta_edge) == 3:
                        for k in second_nhood:
                            is_in_meta = np.in1d(k, meta_edge)
                            if is_in_meta.all():
                                center_node = (np.setdiff1d(meta_edge, k))
                                meta_edges.append(np.array([k[0], center_node[0], k[1]]))
                                normal_edges.append(k)
            if len(meta_edges) > 0:         
                unique_dihedral_edges, is_unique = np.unique(np.stack(normal_edges), axis=0, return_index=True)
                meta_edges_un = np.stack(meta_edges)[is_unique]
                angles = []
                for me in meta_edges_un.astype(np.int64):
                    angles.append(get_dihedral(coords, me))
                dihedral_angles = np.zeros(edgelist.shape[0])
                for i in range(len(angles)):
                    is_edge = np.where(edgelist == unique_dihedral_edges[i], 1, 0).all(axis=1)
                    dihedral_angles[is_edge] = angles[i]
                    is_edge = np.where(edgelist == unique_dihedral_edges[i][::-1], 1, 0).all(axis=1)
                    dihedral_angles[is_edge] = angles[i]
            else:
                dihedral_angles = np.zeros(edgelist.shape[0])
            with open("./processed2/{}_edge_list.csv".format(name), "wb") as f:
                np.savetxt(f, edgelist)
            with open("./processed2/{}_edgeattr.csv".format(name), "wb") as f:
                np.savetxt(f, np.concatenate([edge_attr, dihedral_angles[:,None]], axis=1))
                
    def mem_load(self):
        self.mem = {}
        for i, molecule in enumerate(self.molecule_names):
            graph_features = pd.read_csv("./processed2/{}_graph_features.csv".format(molecule), sep=" ", header=None).to_numpy()
            node_features = pd.read_csv("./processed2/{}_node_attr.csv".format(molecule), sep=" ", header=None).to_numpy()
            atomtypes = node_features[:,-5:].argmax(axis=1)
            prop_atoms = props[atomtypes,:]
            n_nodes = node_features.shape[0]
            graph_features = np.tile(graph_features, [1, n_nodes]).T
            node_features = np.concatenate([prop_atoms, node_features, graph_features], axis=1)
            target =  pd.read_csv("./processed2/{}_target.csv".format(molecule), sep=" ", header=None).to_numpy()
            edge_type = target[:,2]
            edge_type = np.concatenate([edge_type, edge_type], axis=0)
            edges_target = target[:,0:2]
            target = target[:,3]
            target = np.concatenate([target, target])
            edges_target = np.concatenate([edges_target, edges_target[:,::-1]], axis=0)
            edge_list = pd.read_csv("./processed2/{}_edge_list.csv".format(molecule), sep=" ", header=None).to_numpy()
            edge_attr = pd.read_csv("./processed2/{}_edgeattr.csv".format(molecule), sep=" ", header=None).to_numpy()
            data = Data(x=torch.Tensor(node_features), edge_index = torch.Tensor(edge_list).T, y=torch.Tensor(target), edge_attr = torch.Tensor(edge_attr))
            data.nodes_target = torch.Tensor(edges_target)
            data.nodes = n_nodes
            data.edges = edge_list.shape[0]
            data.types = torch.Tensor(edge_type)
            # data.edge_cross = edgelist
            # data.nodes = node_features.shape[0]
            self.mem[molecule] = data
            print("{}/{}".format(i, len(self.molecule_names)), end = "\r")
            
    def __getitem__(self, idx):
        molecule = self.molecule_names[idx]
        return self.mem[molecule]
        

def get_distance_matrix(X, k=None):
    dist = squareform(pdist(X))
    if k is not None:
        non_k = dist.argsort(axis=1)[:, k+1:]
        dist[np.arange(0, dist.shape[0])[:,None], non_k] = 0
    return dist

def to_batch(list_graphs):
    n_nodes = 0
    for graph in list_graphs:
        graph["nodes_target"] += n_nodes
        n_nodes += graph.nodes
    return Batch.from_data_list(list_graphs) 


# In[7]:


from torch_geometric.nn import GCNConv, GATv2Conv, GATConv, SAGEConv
import torch.nn as nn
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


# In[8]:


class GATv2EncoderGated(nn.Module):
    def __init__(self, num_node_features, hidden_features, n_heads, n_layers, p_dropout):
        super().__init__()
        self.p_dropout = p_dropout
        assert n_layers > 1
        self.init_conv = GATv2Conv(num_node_features, hidden_features, heads=n_heads, dropout=p_dropout,  edge_dim=9, concat=False)
        self.layers = nn.ModuleList([GATv2Conv(hidden_features, hidden_features, heads=n_heads, dropout=p_dropout,  edge_dim=9, concat=False) for i in range(n_layers-1)])
        self.gates = nn.Parameter(torch.Tensor(n_layers))
        self.init_conv.apply(init_weights)
        for conv in self.layers:
            conv.apply(init_weights)
    def forward(self, x, edge_index, edge_attr):
        range_gates = torch.sigmoid(self.gates)
        x = self.init_conv(x, edge_index, edge_attr)
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(x)
            x = (range_gates[i])*layer(x, edge_index, edge_attr) + (1-range_gates[i])*x
        return x
    
class ResNetGated(nn.Module):
    def __init__(self, init_dim, hidden_dim, layers, p_dropout):
        super().__init__()
        self.p_dropout = p_dropout
        assert layers > 1
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(init_dim, hidden_dim),
                             nn.ReLU(),
                             nn.Dropout(p=p_dropout),
                             nn.Linear( hidden_dim, init_dim)) for i in range(layers)])
        self.gates = nn.Parameter(torch.Tensor(layers))
        self.layers.apply(init_weights)
    def forward(self, x):
        range_gates = torch.sigmoid(self.gates)
        for i, layer in enumerate(self.layers):
            x = F.relu(x)
            x = (range_gates[i])*layer(x) + (1-range_gates[i])*x
        return x

    
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, out_features, n_heads, n_layers, n_res, p_dropout):
        super().__init__()
        self.conv = GATv2EncoderGated(num_node_features, out_features, n_heads=n_heads, p_dropout=p_dropout,  n_layers=n_layers)
        self.fcs = nn.ModuleList([nn.Sequential(ResNetGated(out_features*2, out_features*64, n_res, p_dropout),
                               nn.Linear(2 * out_features, 1)) for i in range(8)])
        for fc in self.fcs:
            fc.apply(init_weights)
    def forward(self, x, edge_index, edge_attr, edge_cross, types):
        x = self.conv(x, edge_index, edge_attr)
        x = x[edge_cross]
        shp = x.shape
        x = x.transpose(1, 2).reshape([shp[0], shp[2]*2])
        xs = []
        for i in range(8):
            xs.append(self.fcs[i](x[types == i]))
        x = torch.concat(xs, axis=0)
        return x


# In[9]:


def train(model, device, data ,loss_fn, optimizer, batch_acc):
    model.train()
    train_losses = []
    optimizer.zero_grad()
    for i, batch in enumerate(data):
        x, edge_index, edge_attr, target, edge_cross, types = (batch["x"],
                                                               batch["edge_index"],
                                                               batch["edge_attr"],
                                                               batch["y"],
                                                               batch["nodes_target"],
                                                               batch["types"])
        types_cpu = types.numpy()
        sort_index = torch.Tensor(types.numpy().argsort(kind="stable")).long()
        target = target[sort_index]
        x, edge_index, edge_attr, target, edge_cross, types = x.to(device),                                                             edge_index.long().to(device),                                                             edge_attr.to(device),                                                             target.to(device),                                                            edge_cross.long().to(device),                                                             types.long().to(device)
        logits = model(x, edge_index, edge_attr, edge_cross, types)
        loss = loss_fn()
        output=loss(logits.squeeze(), target.squeeze())
        output.backward()
        if ((i+1)%batch_acc == 0):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
        train_loss = output.data.cpu().numpy()
        train_losses.append(train_loss)
        del x, edge_index, edge_attr, target, edge_cross, types
        gc.collect()
    return np.mean(train_losses)

### Testing function
def test(model, device, data, loss_fn):
    # Set evaluation mode for encoder and decoder
    model.eval()
    test_losses = []
    with torch.no_grad(): # No need to track the gradients
        for i, batch in enumerate(data):
            x, edge_index, edge_attr, target, edge_cross, types = (batch["x"],
                                                               batch["edge_index"],
                                                               batch["edge_attr"],
                                                               batch["y"],
                                                               batch["nodes_target"],
                                                               batch["types"])
            types_cpu = types.numpy()
            sort_index = torch.Tensor(types.numpy().argsort(kind="stable")).long()
            target = target[sort_index]
            x, edge_index, edge_attr, target, edge_cross, types = x.to(device),                                                                 edge_index.long().to(device),                                                                 edge_attr.to(device),                                                                 target.to(device),                                                                edge_cross.long().to(device),                                                                 types.long().to(device)
            logits = model(x, edge_index, edge_attr, edge_cross, types)
            loss = loss_fn()
            output=loss(logits.squeeze(), target.squeeze())
            test_loss = output.data.cpu().numpy()
            test_losses.append(test_loss)
            del x, edge_index, edge_attr, target, edge_cross, types
            gc.collect()
    return np.mean(test_losses)


# In[10]:


def train_GCN(config, checkpoint_dir=None):
    loss_fn = nn.MSELoss
    train_dataloader = ray.get(config["train_data"])
    test_dataloader = ray.get(config["test_data"])
    ### Set the random seed for reproducible results
    torch.manual_seed(0)
    model = GCN(37, config["conv_features"], config["n_heads"], config["n_layers"], config["n_res"], config["dropout"])
    params_to_optimize = [
        {'params': model.parameters()}
    ]

    optim = torch.optim.Adam(params_to_optimize, lr=config["lr"], weight_decay=config["weight_decay"])
    # Check if the GPU is available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # Move both the encoder and the decoder to the selected device
    model.to(device)
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optim.load_state_dict(optimizer_state)

    loss_fn = nn.MSELoss

    for epoch in range(2):
        train_loss = train(model, device, train_dataloader, loss_fn, optim, config["batch_acc"])
        test_loss = test(model, device, test_dataloader, loss_fn)

        # Here we save a checkpoint. It is automatically registered with
        # Ray Tune and will potentially be passed as the `checkpoint_dir`
        # parameter in future iterations.
        gc.collect()
    with tune.checkpoint_dir(step=epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save(
            (model.state_dict(), optim.state_dict()), path)
    tune.report(loss=test_loss)
    print("Finished Training")


# In[11]:


def main(num_samples=15, max_num_epochs=50, gpus_per_trial=1):
    dataset = SCDataset('/media/pedro/data/projects/scalar_coupling')
    with open("./data_new.pkl", "rb") as f:
        dataset.mem = pickle.load(f)
    train_length = int(len(dataset) * 0.85)
    test_length = len(dataset) - train_length
    train_set, val_set = torch.utils.data.random_split(dataset, [train_length, test_length])
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=128, collate_fn=to_batch, shuffle=True, num_workers=12)
    test_dataloader = torch.utils.data.DataLoader(val_set, batch_size=128, collate_fn=to_batch, shuffle=True, num_workers=12)
    train_id = ray.put(train_dataloader)
    test_id = ray.put(test_dataloader)
    config = {
    "lr": tune.loguniform(1e-5, 1e-1),
    "dropout": tune.uniform(0, 0.4),
    "weight_decay": tune.loguniform(1e-8, 1e-1),
    "batch_acc": tune.choice([2, 4, 8]),
    "n_heads": tune.choice([2, 3, 4]),
    "n_layers" : tune.choice([2, 3, 4]),
    "n_res" : tune.choice([2, 3, 4]),
    "conv_features" :tune.choice([32, 64, 128]),
    "train_data":train_id,
    "test_data":test_id
    }
    algo = TuneBOHB(metric="loss", mode="min")
    bohb = HyperBandForBOHB(
        time_attr="training_iteration",
        metric="loss",
        mode="min",
        max_t=10)
    result = tune.run(
        train_GCN,
        resources_per_trial={"cpu": 12, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=bohb,
        search_alg=algo
    )
    gc.collect()
    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))


# In[12]:


main()


# In[ ]:





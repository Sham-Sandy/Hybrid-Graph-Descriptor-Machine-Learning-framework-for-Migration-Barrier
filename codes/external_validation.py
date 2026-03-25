import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool

from pymatgen.core import Structure, Lattice

from sklearn.metrics import mean_absolute_error, r2_score


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

BASE = r"D:\ML\Migration Barrier\migration_barrier_ml"

JSON_FILE = os.path.join(BASE,"data","raw","EM-COMPLETE-DATASET.json")

MODEL_DIR = os.path.join(BASE,"models")
RESULT_DIR = os.path.join(BASE,"results")

os.makedirs(RESULT_DIR,exist_ok=True)


# -------------------------------------------------------
# Load trained model
# -------------------------------------------------------

model = pickle.load(open(os.path.join(MODEL_DIR,"hybrid_xgb.pkl"),"rb"))
scaler = pickle.load(open(os.path.join(MODEL_DIR,"scaler.pkl"),"rb"))


# -------------------------------------------------------
# GNN encoder
# -------------------------------------------------------

class PathGNN(nn.Module):

    def __init__(self):

        super().__init__()

        nn1=nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )

        self.conv1=GINEConv(nn1,edge_dim=1)

        nn2=nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        self.conv2=GINEConv(nn2,edge_dim=1)

        nn3=nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

        self.conv3=GINEConv(nn3,edge_dim=1)


    def forward(self,data):

        x=data.x
        edge_index=data.edge_index
        edge_attr=data.edge_attr

        x=F.relu(self.conv1(x,edge_index,edge_attr))
        x=F.relu(self.conv2(x,edge_index,edge_attr))
        x=F.relu(self.conv3(x,edge_index,edge_attr))

        x=global_mean_pool(x,data.batch)

        return x


encoder=PathGNN()

encoder.load_state_dict(
    torch.load(os.path.join(MODEL_DIR,"path_gnn.pt"))
)

encoder.eval()


# -------------------------------------------------------
# Li hop detection
# -------------------------------------------------------

def detect_li_hop(struct):

    li_sites=[i for i,s in enumerate(struct.species) if s.symbol=="Li"]

    if len(li_sites)<2:
        return None

    pairs=[]

    for i in range(len(li_sites)):
        for j in range(i+1,len(li_sites)):

            d=struct.get_distance(li_sites[i],li_sites[j])
            pairs.append((d,li_sites[i],li_sites[j]))

    pairs.sort()

    return pairs[0]


# -------------------------------------------------------
# Graph builder
# -------------------------------------------------------

def build_graph(struct,li1,li2):

    nodes=[li1,li2]

    neigh=struct.get_neighbors(struct[li1],3)

    for n in neigh:
        nodes.append(n.index)

    nodes=list(set(nodes))

    node_map={old:i for i,old in enumerate(nodes)}

    node_feat=[]
    edges=[]
    edge_attr=[]

    for i in nodes:

        site=struct[i]

        Z=site.specie.Z
        en=site.specie.X if site.specie.X else 0
        r=site.specie.atomic_radius or 1

        node_feat.append([Z,en,r])


    for i in nodes:
        for j in nodes:

            if i>=j:
                continue

            d=struct.get_distance(i,j)

            if d<4:

                edges.append([node_map[i],node_map[j]])
                edges.append([node_map[j],node_map[i]])

                edge_attr.append([d])
                edge_attr.append([d])


    node_feat=torch.tensor(node_feat,dtype=torch.float)
    edge_index=torch.tensor(edges).t().contiguous()
    edge_attr=torch.tensor(edge_attr,dtype=torch.float)

    return Data(x=node_feat,edge_index=edge_index,edge_attr=edge_attr)


# -------------------------------------------------------
# Load external dataset
# -------------------------------------------------------

data=json.load(open(JSON_FILE))

rows=[]

print("Running external validation")


for entry in tqdm(data):

    formula = entry["formula"]

    # Filter Li systems
    if "Li" not in formula:
        continue

    true = entry["target"]

    # Barrier sanity filter
    if true < 0 or true > 1.2:
        continue


    elems=entry["structure_ini"]["elements"]

    struct=Structure(

        Lattice(entry["structure_ini"]["lattice_mat"]),
        elems,
        entry["structure_ini"]["coords"]

    )

    struct=struct.get_reduced_structure()


    hop_data=detect_li_hop(struct)

    if hop_data is None:
        continue


    hop,li1,li2=hop_data


    neigh=struct.get_neighbors(struct[li1],3)

    if len(neigh)==0:
        continue


    graph=build_graph(struct,li1,li2)


    dists=[n.nn_distance for n in neigh]

    coord=len(dists)

    avg_dist=np.mean(dists)
    min_dist=np.min(dists)
    std_dist=np.std(dists)

    bottleneck=min_dist-1.4
    distortion=std_dist/avg_dist


    vol=struct.volume
    nat=len(struct)
    density=vol/nat


    feat=[

        vol,nat,density,
        hop,coord,
        avg_dist,min_dist,std_dist,
        bottleneck,
        distortion

    ]


    g=graph
    g.batch=torch.zeros(g.x.shape[0],dtype=torch.long)

    with torch.no_grad():
        emb=encoder(g).numpy()[0]


    X=np.concatenate([feat,emb]).reshape(1,-1)

    X=scaler.transform(X)

    pred=np.exp(model.predict(X))[0]


    rows.append({

        "formula":formula,
        "true_barrier":true,
        "predicted_barrier":pred,
        "abs_error":abs(true-pred)

    })


df=pd.DataFrame(rows)

df.to_csv(
    os.path.join(RESULT_DIR,"external_predictions.csv"),
    index=False
)


# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

mae=mean_absolute_error(df.true_barrier,df.predicted_barrier)
r2=r2_score(df.true_barrier,df.predicted_barrier)

metrics=pd.DataFrame({

"MAE":[mae],
"R2":[r2]

})

metrics.to_csv(
    os.path.join(RESULT_DIR,"external_metrics.csv"),
    index=False)


print("\nExternal validation results")

print("Samples used:",len(df))
print("MAE:",mae)
print("R2 :",r2)

print("\nSaved:")
print("results/external_predictions.csv")
print("results/external_metrics.csv")
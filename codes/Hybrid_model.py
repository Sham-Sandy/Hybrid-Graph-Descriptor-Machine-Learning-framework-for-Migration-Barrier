import os
import random
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool

from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor


# -------------------------------------------------------
# Reproducibility
# -------------------------------------------------------

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

DATA_DIR = r"D:\ML\Migration Barrier\migration_barrier_ml\data\raw\nebDFT2k"
INDEX_FILE = os.path.join(DATA_DIR,"nebDFT2k_index.csv")

DATASET_FILE = "hybrid_dataset_edge.pkl"

MODEL_DIR="models"
RESULT_DIR="results"
DATASET_DIR="datasets"

os.makedirs(MODEL_DIR,exist_ok=True)
os.makedirs(RESULT_DIR,exist_ok=True)
os.makedirs(DATASET_DIR,exist_ok=True)


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
# Dataset creation
# -------------------------------------------------------

if os.path.exists(DATASET_FILE):

    print("Loading cached dataset")

    graphs,features,targets=pickle.load(open(DATASET_FILE,"rb"))

else:

    print("Building dataset")

    index=pd.read_csv(INDEX_FILE)

    xyz_files=[f for f in os.listdir(DATA_DIR) if f.endswith("_init.xyz")]

    graphs=[]
    features=[]
    targets=[]

    for fname in tqdm(xyz_files):

        atoms=read(os.path.join(DATA_DIR,fname))
        struct=AseAtomsAdaptor.get_structure(atoms)

        material=fname.split("_")[0]

        row=index[index["material_id"]==material]

        if len(row)==0:
            continue

        row=row.iloc[0]

        hop_data=detect_li_hop(struct)

        if hop_data is None:
            continue

        hop,li1,li2=hop_data

        graph=build_graph(struct,li1,li2)

        graphs.append(graph)

        neigh=struct.get_neighbors(struct[li1],3)

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

        features.append(feat)

        targets.append(row["em_dft"])

    pickle.dump((graphs,features,targets),open(DATASET_FILE,"wb"))

    print("Dataset saved")


print("Samples:",len(graphs))


# -------------------------------------------------------
# GNN Encoder
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


train_graphs,test_graphs=train_test_split(graphs,test_size=0.2,random_state=42)

train_loader=DataLoader(train_graphs,batch_size=32,shuffle=True)

encoder=PathGNN()

optimizer=torch.optim.Adam(encoder.parameters(),lr=5e-4)

print("\nTraining Path-GNN encoder")

for epoch in range(40):

    encoder.train()

    for batch in train_loader:

        optimizer.zero_grad()

        emb=encoder(batch)

        loss=(emb**2).mean()

        loss.backward()

        optimizer.step()


torch.save(
    encoder.state_dict(),
    os.path.join(MODEL_DIR,"path_gnn.pt")
)

print("Saved Path-GNN model")


# -------------------------------------------------------
# Generate embeddings
# -------------------------------------------------------

encoder.eval()

embeddings=[]

for g in graphs:

    g.batch=torch.zeros(g.x.shape[0],dtype=torch.long)

    with torch.no_grad():

        emb=encoder(g)

    embeddings.append(emb.numpy()[0])

embeddings=np.array(embeddings)

np.save(os.path.join(DATASET_DIR,"gnn_embeddings.npy"),embeddings)

print("Saved embeddings")


# -------------------------------------------------------
# Hybrid ML
# -------------------------------------------------------

X=np.concatenate([np.array(features),embeddings],axis=1)

y=np.log(np.array(targets)+1e-6)

X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)

scaler=StandardScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


model=XGBRegressor(

    n_estimators=3500,
    max_depth=9,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train,y_train)


pickle.dump(
    model,
    open(os.path.join(MODEL_DIR,"hybrid_xgb.pkl"),"wb")
)

pickle.dump(
    scaler,
    open(os.path.join(MODEL_DIR,"scaler.pkl"),"wb")
)

print("Saved hybrid ML model")


# -------------------------------------------------------
# Predictions
# -------------------------------------------------------

# Train metrics
train_pred=np.exp(model.predict(X_train))
train_true=np.exp(y_train)

train_mae=mean_absolute_error(train_true,train_pred)
train_r2=r2_score(train_true,train_pred)

# Test metrics
test_pred=np.exp(model.predict(X_test))
test_true=np.exp(y_test)

test_mae=mean_absolute_error(test_true,test_pred)
test_r2=r2_score(test_true,test_pred)

print("\nHybrid Migration Barrier Model")

print("\nTrain Performance")
print("Train MAE:",train_mae)
print("Train R2 :",train_r2)

print("\nTest Performance")
print("Test MAE:",test_mae)
print("Test R2 :",test_r2)


# -------------------------------------------------------
# Save metrics
# -------------------------------------------------------

metrics=pd.DataFrame({

"train_MAE":[train_mae],
"train_R2":[train_r2],
"test_MAE":[test_mae],
"test_R2":[test_r2]

})

metrics.to_csv(
    os.path.join(RESULT_DIR,"internal_metrics.csv"),
    index=False
)


# -------------------------------------------------------
# Save predictions
# -------------------------------------------------------

pred_df=pd.DataFrame({

"true_barrier":test_true,
"predicted_barrier":test_pred,
"abs_error":np.abs(test_true-test_pred)

})

pred_df.to_csv(
    os.path.join(RESULT_DIR,"internal_predictions.csv"),
    index=False
)

print("Saved metrics and predictions")
import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from pymatgen.core import Structure
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_mean_pool


# -------------------------------------------------------
# USER INPUT
# -------------------------------------------------------

structure_file = input("\nEnter path to structure file (CIF/POSCAR/XYZ): ").strip()


# -------------------------------------------------------
# PATHS
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models_noembed")


# -------------------------------------------------------
# DEFINE EXACT SAME GNN
# -------------------------------------------------------

class PathGNN(nn.Module):

    def __init__(self):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(),
            nn.Linear(64,64)
        )
        self.conv1 = GINEConv(nn1, edge_dim=1)

        nn2 = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.conv2 = GINEConv(nn2, edge_dim=1)

        nn3 = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.conv3 = GINEConv(nn3, edge_dim=1)

    def forward(self, data):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))

        x = global_mean_pool(x, data.batch)

        return x


# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------

print("\nLoading trained models...")

try:
    model = pickle.load(open(os.path.join(MODEL_DIR, "hybrid_xgb_noembed.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler_noembed.pkl"), "rb"))

    gnn_model = PathGNN()
    gnn_model.load_state_dict(
        torch.load(os.path.join(MODEL_DIR, "path_gnn_noembed.pt"),
                   map_location=torch.device('cpu'))
    )
    gnn_model.eval()

except Exception as e:
    print("❌ Error loading model files:", e)
    exit()


# -------------------------------------------------------
# LOAD STRUCTURE
# -------------------------------------------------------

try:
    structure = Structure.from_file(structure_file)
except Exception as e:
    print("❌ Error loading structure:", e)
    exit()


# -------------------------------------------------------
# Li HOP DETECTION (EXACT SAME)
# -------------------------------------------------------

def detect_li_hop(struct):

    li_sites = [i for i,s in enumerate(struct.species) if s.symbol=="Li"]

    if len(li_sites) < 2:
        return None

    pairs = []

    for i in range(len(li_sites)):
        for j in range(i+1, len(li_sites)):
            d = struct.get_distance(li_sites[i], li_sites[j])
            pairs.append((d, li_sites[i], li_sites[j]))

    pairs.sort()
    return pairs[0]


hop_data = detect_li_hop(structure)

if hop_data is None:
    print("❌ No Li hop found")
    exit()

hop, li1, li2 = hop_data


# -------------------------------------------------------
# GRAPH BUILDER (EXACT SAME)
# -------------------------------------------------------

def build_graph(struct, li1, li2):

    nodes = [li1, li2]

    neigh = struct.get_neighbors(struct[li1], 3)

    for n in neigh:
        nodes.append(n.index)

    nodes = list(set(nodes))

    node_map = {old:i for i,old in enumerate(nodes)}

    node_feat = []
    edges = []
    edge_attr = []

    for i in nodes:
        site = struct[i]
        Z = site.specie.Z
        en = site.specie.X if site.specie.X else 0
        r = site.specie.atomic_radius or 1

        node_feat.append([Z, en, r])

    for i in nodes:
        for j in nodes:

            if i >= j:
                continue

            d = struct.get_distance(i,j)

            if d < 4:
                edges.append([node_map[i], node_map[j]])
                edges.append([node_map[j], node_map[i]])

                edge_attr.append([d])
                edge_attr.append([d])

    node_feat = torch.tensor(node_feat, dtype=torch.float)
    edge_index = torch.tensor(edges).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=node_feat, edge_index=edge_index, edge_attr=edge_attr)


graph = build_graph(structure, li1, li2)

graph.batch = torch.zeros(graph.x.shape[0], dtype=torch.long)


# -------------------------------------------------------
# DESCRIPTORS (EXACT SAME)
# -------------------------------------------------------

neigh = structure.get_neighbors(structure[li1], 3)

dists = [n.nn_distance for n in neigh]

coord = len(dists)

avg_dist = np.mean(dists)
min_dist = np.min(dists)
std_dist = np.std(dists)

bottleneck = min_dist - 1.4
distortion = std_dist / avg_dist

vol = structure.volume
nat = len(structure)
density = vol / nat

desc = np.array([
    vol, nat, density,
    hop, coord,
    avg_dist, min_dist, std_dist,
    bottleneck,
    distortion
])


# -------------------------------------------------------
# GNN EMBEDDING (MATCH TRAINING)
# -------------------------------------------------------

REMOVE_IDX = [6,52,87]

with torch.no_grad():
    emb = gnn_model(graph)

emb = emb.numpy()[0]
emb = np.delete(emb, REMOVE_IDX)


# -------------------------------------------------------
# HYBRID FEATURE
# -------------------------------------------------------

X = np.concatenate([desc, emb])
X = scaler.transform([X])


# -------------------------------------------------------
# PREDICTION
# -------------------------------------------------------

log_pred = model.predict(X)[0]
pred = np.exp(log_pred)


# -------------------------------------------------------
# OUTPUT
# -------------------------------------------------------

print("\n====================================")
print(f"Structure: {structure_file}")
print(f"Li hop distance: {hop:.3f} Å")
print(f"Predicted Migration Barrier: {pred:.4f} eV")
print("====================================\n")
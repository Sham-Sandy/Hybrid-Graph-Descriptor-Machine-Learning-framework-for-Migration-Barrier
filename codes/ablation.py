import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


# -------------------------------------------------------
# PATHS (EDIT IF NEEDED)
# -------------------------------------------------------

DATASET_FILE = "hybrid_dataset_edge_noembed.pkl"
EMBED_FILE = "datasets_noembed/gnn_embeddings_noembed.npy"

RESULT_DIR = "results_ablation"
os.makedirs(RESULT_DIR, exist_ok=True)


# -------------------------------------------------------
# LOAD DATA  
# -------------------------------------------------------

print("Loading dataset...")

graphs, features, targets = pickle.load(open(DATASET_FILE, "rb"))

features = np.array(features)
targets = np.array(targets)

print("Loading embeddings...")
embeddings = np.load(EMBED_FILE)

print("Samples:", len(features))


# -------------------------------------------------------
# FEATURE MATRICES
# -------------------------------------------------------

X_desc = features
X_gnn = embeddings
X_hybrid = np.concatenate([features, embeddings], axis=1)

y = np.log(targets + 1e-6)


# -------------------------------------------------------
# SAME SPLIT FOR ALL  IMPORTANT
# -------------------------------------------------------

idx = np.arange(len(y))

train_idx, test_idx = train_test_split(
    idx, test_size=0.2, random_state=42
)

X_train_d, X_test_d = X_desc[train_idx], X_desc[test_idx]
X_train_g, X_test_g = X_gnn[train_idx], X_gnn[test_idx]
X_train_h, X_test_h = X_hybrid[train_idx], X_hybrid[test_idx]

y_train, y_test = y[train_idx], y[test_idx]


# -------------------------------------------------------
# SCALING
# -------------------------------------------------------

scaler_d = StandardScaler()
scaler_g = StandardScaler()
scaler_h = StandardScaler()

X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)

X_train_g = scaler_g.fit_transform(X_train_g)
X_test_g = scaler_g.transform(X_test_g)

X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)


# -------------------------------------------------------
# EVALUATION FUNCTION
# -------------------------------------------------------

def evaluate(model, X_train, X_test, name):

    model.fit(X_train, y_train)

    train_pred = np.exp(model.predict(X_train))
    train_true = np.exp(y_train)

    test_pred = np.exp(model.predict(X_test))
    test_true = np.exp(y_test)

    train_mae = mean_absolute_error(train_true, train_pred)
    test_mae = mean_absolute_error(test_true, test_pred)

    train_r2 = r2_score(train_true, train_pred)
    test_r2 = r2_score(test_true, test_pred)

    # Save predictions
    df = pd.DataFrame({
        "True": test_true,
        "Predicted": test_pred,
        "Error": np.abs(test_true - test_pred)
    })

    df.to_csv(os.path.join(RESULT_DIR, f"{name}_predictions.csv"), index=False)

    # Parity plot
    plt.figure(figsize=(5,5))
    plt.scatter(test_true, test_pred, alpha=0.6)
    plt.plot([test_true.min(), test_true.max()],
             [test_true.min(), test_true.max()], 'r--')
    plt.xlabel("DFT (eV)")
    plt.ylabel("Predicted (eV)")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, f"{name}_parity.png"), dpi=300)
    plt.close()

    return train_mae, test_mae, train_r2, test_r2


# -------------------------------------------------------
# MODELS
# -------------------------------------------------------

models = {

    "XGB_Descriptor": (XGBRegressor(
        n_estimators=3500,
        max_depth=9,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ), X_train_d, X_test_d),

    "RandomForest": (RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    ), X_train_d, X_test_d),

    "SVR": (SVR(
        kernel="rbf",
        C=10,
        gamma="scale"
    ), X_train_d, X_test_d),

    "XGB_GNN": (XGBRegressor(
        n_estimators=3500,
        max_depth=9,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ), X_train_g, X_test_g),

    "XGB_Hybrid": (XGBRegressor(
        n_estimators=3500,
        max_depth=9,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ), X_train_h, X_test_h),
}


# -------------------------------------------------------
# RUN ALL MODELS
# -------------------------------------------------------

results = []

print("\nRunning ablation study...\n")

for name, (model, Xtr, Xte) in models.items():

    print(f"Training {name}...")

    tr_mae, te_mae, tr_r2, te_r2 = evaluate(model, Xtr, Xte, name)

    results.append({
        "Model": name,
        "Train_MAE": tr_mae,
        "Test_MAE": te_mae,
        "Train_R2": tr_r2,
        "Test_R2": te_r2
    })


# -------------------------------------------------------
# SAVE RESULTS
# -------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nFinal Results:")
print(results_df)

results_df.to_csv(os.path.join(RESULT_DIR, "ablation_all_models.csv"), index=False)


# -------------------------------------------------------
# PLOT
# -------------------------------------------------------

plt.figure(figsize=(7,5))

sns.barplot(x="Model", y="Test_MAE", data=results_df)

plt.xticks(rotation=45)
plt.ylabel("MAE (eV)")
plt.title("Model Comparison")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "model_comparison.png"), dpi=300)

plt.close()


print("\n DONE: All results saved in", RESULT_DIR)

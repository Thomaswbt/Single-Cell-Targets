import anndata as ad
import pandas as pd
import scanpy as sc
import os
import torch
import numpy as np
from preprocess import *
from scETM import scETM, UnsupervisedTrainer, evaluate, prepare_for_transfer
sc.set_figure_params(dpi=120, dpi_save=250, fontsize=10, figsize=(10, 10), facecolor="white")

# Construct mouse pancreas AnnData object
# mp_csvs = ['GSM2230761_mouse1_umifm_counts.csv', 'GSM2230762_mouse2_umifm_counts.csv']

# dataname = 'GBM.csv'
dataname = 'IDH-MUT.csv'
mp_csvs = [dataname]
mp_adatas = []
for fpath in mp_csvs:
    df = pd.read_csv(os.path.join("data",fpath), index_col=0)
    adata = ad.AnnData(X=df.iloc[:, 0:], obs=df.iloc[:, :0])
    mp_adatas.append(adata)
mp = ad.concat(mp_adatas, label="batch_indices")

# Preprocess:
graph_path = os.path.join("graph","graph_"+dataname[:-4]+".npy")
if not os.path.exists(graph_path):
    preprocess(dataname)
###

graph_np = np.load(graph_path)
graph = torch.from_numpy(graph_np)
del graph_np
mp_model = scETM(mp.n_vars, mp.obs.batch_indices.nunique(),guidance_graph = graph)
trainer = UnsupervisedTrainer(mp_model, mp, test_ratio=0)

# Loss = original_loss + reg_weight * L_reg
trainer.train(n_epochs = 12000, eval_every = 3000, eval=False, reg_weight = 0.01, eval_kwargs = dict(cell_type_col = 'assigned_cluster'))


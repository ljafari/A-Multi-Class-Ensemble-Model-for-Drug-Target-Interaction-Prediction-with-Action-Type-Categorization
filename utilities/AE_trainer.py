# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 10:01:21 2025

@author: Leila Jafari Khouzani
"""
# dual_encoder/trainer.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# add to the top with other imports:
from utilities.DatasetBuilder import DatasetBuilder
# ---------------- Autoencoder for tabular features ----------------
class TabularAutoencoder(nn.Module):
    """
    Simple MLP autoencoder:
      - encoder: input_dim -> hidden -> bottleneck
      - decoder: bottleneck -> hidden -> input_dim
    Used per-entity (drugs and targets) with reconstruction loss.
    """
    def __init__(self, input_dim: int, bottleneck: int = 128):
        super().__init__()
        hidden = max(64, input_dim // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, bottleneck)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        xrec = self.decoder(z)
        return xrec, z

def _train_autoencoder_matrix(X_np: np.ndarray,
                              bottleneck: int = 128,
                              epochs: int = 80,
                              batch_size: int = 128,
                              lr: float = 1e-3,
                              device: str = None):
    """
    Train an autoencoder on a single matrix (e.g., all drugs or all targets).
    Returns the trained model and (reconstruction_mse, embeddings).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_np.shape[1]
    model = TabularAutoencoder(input_dim=input_dim, bottleneck=bottleneck).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X = torch.from_numpy(X_np.astype(np.float32))
    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        count = 0
        for (xb,) in dl:
            xb = xb.to(device)
            xrec, z = model(xb)
            loss = loss_fn(xrec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
        if ep == 1 or ep % 10 == 0:
            print(f"[AE] epoch {ep:03d}/{epochs}  recon_mse={total / max(1, count):.6f}")

    # switch to eval, compute embeddings + per-sample reconstruction_mse
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        xrec_all, z_all = model(X_dev)
        rec_mse = ((xrec_all - X_dev) ** 2).mean(dim=1).detach().cpu().numpy()
        Z = z_all.detach().cpu().numpy()

    return model, rec_mse, Z


def train_autoencoder_and_export(builder, cfg, bottleneck: int = 128):
    """
    Full AE pipeline:
      1) Load normalized drug/target matrices from DatasetBuilder
      2) Train separate AEs for drugs and targets
      3) Export embeddings to CSV (with IDs + reconstruction_mse)
    Output directory: <cfg.run_dir>/autoencoder_b{bottleneck}/
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = os.path.join(cfg.run_dir, f"autoencoder_b{bottleneck}")
    os.makedirs(save_dir, exist_ok=True)

    # ----------------- Drugs -----------------
    drug_ids = builder.drugs_dic["ids"]
    drug_feats = builder.drugs_dic["normalized_features"]  # np.ndarray [N_drug, d]
    print(f"[AE] Training on drugs: {drug_feats.shape}")

    _, drug_rec_mse, drug_Z = _train_autoencoder_matrix(
        X_np=drug_feats,
        bottleneck=bottleneck,
        epochs=getattr(cfg.model, "ae_epochs", 80),
        batch_size=getattr(cfg.model, "batch_size", 128),
        lr=getattr(cfg.model, "lr", 1e-3),
        device=device
    )

    drug_df = pd.DataFrame(drug_Z, columns=[f"z_{i}" for i in range(drug_Z.shape[1])])
    drug_df.insert(0, "DrugID", drug_ids)
    drug_df["reconstruction_mse"] = drug_rec_mse
    drug_out = os.path.join(save_dir, "drug_autoenc_embeddings.csv")
    drug_df.to_csv(drug_out, index=False)
    print(f"[AE] Saved drug embeddings → {drug_out}")

    # ----------------- Targets -----------------
    target_ids = builder.targets_dic["ids"]
    target_feats = builder.targets_dic["normalized_features"]  # np.ndarray [N_target, d]
    print(f"[AE] Training on targets: {target_feats.shape}")

    _, target_rec_mse, target_Z = _train_autoencoder_matrix(
        X_np=target_feats,
        bottleneck=bottleneck,
        epochs=getattr(cfg.model, "ae_epochs", 80),
        batch_size=getattr(cfg.model, "batch_size", 128),
        lr=getattr(cfg.model, "lr", 1e-3),
        device=device
    )

    target_df = pd.DataFrame(target_Z, columns=[f"z_{i}" for i in range(target_Z.shape[1])])
    target_df.insert(0, "UniprotID", target_ids)
    target_df["reconstruction_mse"] = target_rec_mse
    target_out = os.path.join(save_dir, "target_autoenc_embeddings.csv")
    target_df.to_csv(target_out, index=False)
    print(f"[AE] Saved target embeddings → {target_out}")

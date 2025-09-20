"""
Created on Wed May  1 15:27:53 2024

@author: Leila Jafari
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as transforms
from utilities.config import cfg


# =============================================================================
#  class to build and manage datasets for drug-target interaction (DTI) prediction tasks.
# ============================================================================= 
class DatasetBuilder:
    def __init__(self):
        self.drugs_dic = {}
        self.targets_dic = {}
        self.test_drugs_dic = {}
        self.test_targets_dic = {}
        self.test_DTI = {}
        self.DTI = {}
        self.hetero_graph = None
        self.in_dir = cfg.in_dir

    # =============================================================================
    def read_features(self, file, sep=','):

        filename = os.path.join(self.in_dir, file)
        df = pd.read_csv(filename, sep=sep, low_memory=False)
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        ids = list(df.iloc[:, 0])
        features = df_numeric.iloc[:, 1:].to_numpy()
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)

        feature_map = {id1: normalized_features[idx] for idx, id1 in enumerate(ids)}

        dic_feats = {
            'df': df,
            'ids': ids,
            'features': features,
            'normalized_features': normalized_features,
            'feature_map': feature_map
        }

        return dic_feats
    # =============================================================================
    def read_interactions(self, file, sep=','):

        filename = os.path.join(self.in_dir, file)
        df = pd.read_csv(filename, sep=sep)

        if not (df['ActionLabel'] == 0).any():
            zero_file = os.path.join(self.in_dir, "zero pairs with thr", cfg.dataset.zero_edges)
            zero_df = pd.read_csv(zero_file, sep=sep)
            filtered_df = zero_df[zero_df.iloc[:, 0].isin(self.drugs_dic['ids']) &
                                  zero_df.iloc[:, 1].isin(self.targets_dic['ids'])]
            zero_df = filtered_df.head(4000)
            df = pd.concat([df, zero_df], ignore_index=True)

        df = df[df.iloc[:, 0].isin(self.drugs_dic['ids']) &
                df.iloc[:, 1].isin(self.targets_dic['ids'])]
        cfg.dataset.drug_label = df.columns[0]
        cfg.dataset.target_label = df.columns[1]
        cfg.dataset.ActionLabel = 'ActionLabel'
        return df

    def drugwise_test_split(self, df):
        """
        Split self.drugs_dic into train and test drug dictionaries
        and re-normalize using only the training drugs to avoid data leakage.
        """
        unique_drugs = df.iloc[:, 0].unique()
        n_total = len(unique_drugs)
        n_heldout = int(0.05 * n_total)

        np.random.seed(42)
        heldout_drugs = np.random.choice(unique_drugs, size=n_heldout, replace=False)

        test_df = df[df.iloc[:, 0].isin(heldout_drugs)].reset_index(drop=True)
        train_df = df[~df.iloc[:, 0].isin(heldout_drugs)].reset_index(drop=True)

        self.DTI = {
            'edges': train_df,
            'y': train_df['ActionLabel'].values,
            'num_samples': len(train_df),
            'drug_label': train_df.columns[0],
            'target_label': train_df.columns[1],
            'DTI_label': train_df.columns[2],
        }
        self.test_DTI = {
            'edges': test_df,
            'y': test_df['ActionLabel'].values,
            'num_samples': len(test_df),
            'drug_label': test_df.columns[0],
            'target_label': test_df.columns[1],
            'DTI_label': test_df.columns[2],

        }
        test_drugs = set(heldout_drugs)
        train_mask = [i for i, id_ in enumerate(self.drugs_dic['ids']) if id_ not in test_drugs]
        test_mask = [i for i, id_ in enumerate(self.drugs_dic['ids']) if id_ in test_drugs]

        # Get raw (unnormalized) features
        all_features = self.drugs_dic['features']
        train_features = all_features[train_mask]
        test_features = all_features[test_mask]

        # Train scaler only on train features
        scaler = StandardScaler()
        norm_train = scaler.fit_transform(train_features)
        norm_test = scaler.transform(test_features)

        train_ids = [self.drugs_dic['ids'][i] for i in train_mask]
        test_ids = [self.drugs_dic['ids'][i] for i in test_mask]

        self.drugs_dic = {
            'df': self.drugs_dic['df'][~self.drugs_dic['df'].iloc[:, 0].isin(test_drugs)].reset_index(drop=True),
            'ids': train_ids,
            'features': train_features,
            'normalized_features': norm_train,
            'feature_map': {id_: vec for id_, vec in zip(train_ids, norm_train)}
        }

        self.test_drugs_dic = {
            'df': self.drugs_dic['df'][self.drugs_dic['df'].iloc[:, 0].isin(test_drugs)].reset_index(drop=True),
            'ids': test_ids,
            'features': test_features,
            'normalized_features': norm_test,
            'feature_map': {id_: vec for id_, vec in zip(test_ids, norm_test)}
        }
        print(f"✅ Train drugs: {len(train_ids)} | Test drugs: {len(test_ids)}")

        self.test_targets_dic = self.targets_dic
    # =============================================================================
    def create_dataset(self):
        self.drugs_dic = self.read_features(cfg.dataset.DrugDiscriptors)
        self.targets_dic = self.read_features(cfg.dataset.ProteinFeatures)
        df = self.read_interactions(cfg.dataset.DTI)
        self.drugwise_test_split(df)

    # =============================================================================
    def build_hetero_graph(self, device='cpu'):
        """
        Build a heterogeneous graph from the given data.
        Args:
            device: Device to store tensors on
        Returns:
            HeteroData: The constructed heterogeneous graph
        """
        data = HeteroData()
        drugs_dic = self.drugs_dic
        targets_dic = self.targets_dic
        edges_df =  self.DTI['edges']

        # -------------------- Assign node features --------------------
        drug_feats = torch.tensor(drugs_dic['normalized_features'], dtype=torch.float, device=device)
        protein_feats = torch.tensor(targets_dic['normalized_features'], dtype=torch.float, device=device)
        data['drug'].x = drug_feats
        data['protein'].x = protein_feats

        # -------------------- Map node IDs to indices --------------------
        drug_id_to_index = {id_: idx for idx, id_ in enumerate(drugs_dic['ids'])}
        prot_id_to_index = {id_: idx for idx, id_ in enumerate(targets_dic['ids'])}

        # -------------------- Build edge list --------------------
        src_list = []  # Drug node indices (source)
        dst_list = []  # Protein node indices (destination)
        label_list = []  # Interaction labels
        d_list, p_list = [], []
        for _, row in edges_df.iterrows():
            d_id = row[cfg.dataset.drug_label]
            p_id = row[cfg.dataset.target_label]
            y = row[cfg.dataset.ActionLabel]

            if d_id in drug_id_to_index and p_id in prot_id_to_index:
                src_list.append(drug_id_to_index[d_id])
                dst_list.append(prot_id_to_index[p_id])
                label_list.append(y)
                d_list.append(d_id)
                p_list.append(p_id)
        # -------------------- Create edge index tensor --------------------
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long, device=device)
        edge_labels = torch.tensor(label_list, dtype=torch.long, device=device)

        # Replace -1 labels with 2 to ensure labels are non-negative for classification
        edge_labels = edge_labels.clone()
        edge_labels[edge_labels == -1] = 2

        # Assign edges and labels to the heterogeneous graph
        data['drug', 'interacts', 'protein'].edge_index = edge_index
        data['drug', 'interacts', 'protein'].edge_label = edge_labels

        data = transforms.ToUndirected()(data)
        data_dic = {'edge_index': edge_index,
                    'edge_labels': edge_labels,
                    'd_list': d_list,
                    'p_list': p_list}
        self.hetero_graph = data
        return data, data_dic

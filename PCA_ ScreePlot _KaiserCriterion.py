# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 22:23:37 2025

@author: Leila Jafari Khouzani
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utilities.config import cfg, set_cfg                                  
import os
def KaiserCriterion(csv_file, datatype):
    
    csv_path = os.path.join(cfg.in_dir, csv_file)
    
    df = pd.read_csv(csv_path)
    sample_names = df.iloc[:, 0] 
    X = df.iloc[:, 1:] #features
    # ======= Step 2: Standardize the features =======
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    # ======= Step 3: Fit PCA without specifying n_components =======
    pca = PCA()
    pca.fit(X_scaled)

    # ======= Step 4: Scree Plot (Explained Variance) =======
    plt.figure(figsize=(10, 6))
    max_components = min(250, len(pca.explained_variance_ratio_))
    plt.plot(range(1, max_components + 1), pca.explained_variance_ratio_[:max_components] * 100, marker='o')
    plt.title('Scree Plot (Explained Variance per Component)')
    plt.xlabel(f'{datatype} Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ======= Step 5: Kaiser Criterion (Eigenvalues > 1) =======
    eigenvalues = pca.explained_variance_
    kaiser_components = np.sum(eigenvalues > 1)
    print(f'✅ According to Kaiser criterion, keep {kaiser_components} components (eigenvalue > 1).')

    # ======= Step 6: Reduce dimensionality using optimal component count =======
    pca_optimal = PCA(n_components=kaiser_components)
    X_reduced = pca_optimal.fit_transform(X_scaled)

    # ======= Step 7: Save reduced features to CSV =======
    reduced_df = pd.DataFrame(X_reduced, columns=[f'PC{i+1}' for i in range(kaiser_components)])
    reduced_df.insert(0, datatype, sample_names)
    # reduced_df.to_csv(f'dataset//{file}_reduced_features.csv', index=False)
    print("✅ Reduced feature matrix saved to 'reduced_features.csv'.")
# =============================================================================   
if __name__ == '__main__':   
    cfg = set_cfg(cfg) 
       
    # ======= Step 1: Load your feature CSV file =======    
    cur_dir = 'dataset//'
    KaiserCriterion(cur_dir+cfg.dataset.DrugDiscriptors, datatype='drug' )
    KaiserCriterion(cur_dir+cfg.dataset.ProteinFeatures, datatype='protein' )
    
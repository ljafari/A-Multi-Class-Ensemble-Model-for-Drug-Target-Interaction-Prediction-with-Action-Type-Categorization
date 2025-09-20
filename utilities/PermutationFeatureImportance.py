# -*- coding: /utf-8 -*-
"""
Created on Fri Jun 27 11:22:26 2025

@author: laila
"""

import os, time
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.inspection import permutation_importance

from utilities.config import cfg, create_params_yaml
from utilities.classification import combine_features, choose_classifier, define_classifiers
from utilities.calculate_metrics import calculate_classification_metrics

#==============================================================================
# class 
#==============================================================================
class PermutationFeatureImportance:
    def __init__(self, builder): 
        self.cur_dir = None
        self.drugs_features = builder.drugs_dic['normalized_features']
        self.targets_features = builder.targets_dic['normalized_features']
        self.df_edges = builder.DTI    
        self.n_component  = cfg.feature.max_n_components       
             
        self.builder = builder
        self.drug_ids = list(self.builder.drugs_dic['ids'])
        self.target_ids = list(self.builder.targets_dic['ids'])   
        self.test_drug_ids =  list(self.builder.test_drugs_dic['ids'])
        self.test_target_ids = list(self.builder.test_targets_dic['ids'])   
        self.k_fold = cfg.model.n_splits   
        self.id_label_columns = [cfg.dataset.drug_label,
                                 cfg.dataset.target_label, 
                                 cfg.dataset.ActionLabel]  
        self.constant_index = 6 #DT because HGB can not be used
        
        self.metrics_columns = ['Permutation_feature_importance', 'AverageLoss', 
                                'Accuracy', 'Precision', 'Recall', 'F1', 'MCC', 
                                'ROC_AUC','0_precision','0_recall',
                                'f1',	'1_precision',
                                'recall', 'f1',	'2_precision',
                                'recall',	'2_f1']
        
        self.run()
    # ============================================================================= 
    def run(self):
        classifiers = define_classifiers()
        clf_name, clf = choose_classifier(classifiers)
        self.cur_dir = create_params_yaml(cfg, f'{clf_name}')           
        os.makedirs(cfg.run_dir+"\\Test results", exist_ok=True)
        train_results = []
        valid_results = []  
        test_results = [] 
        
        print("create test data...") 
        test_drugs_features = self.builder.test_drugs_dic['normalized_features']
        test_targets_features = self.builder.test_targets_dic['normalized_features']
        
        test_processed_features = combine_features(self.builder.test_DTI['edges'],
                                                     test_drugs_features, 
                                                     test_targets_features,
                                                     self.test_drug_ids,
                                                     self.test_target_ids)
        X_test = test_processed_features.drop(columns=test_processed_features.columns[:3])
        y_test = self.builder.test_DTI['y']
        X_test = np.array(X_test)
        y_test = np.array(y_test) 
        
        df_DTI = self.df_edges
        X_df = combine_features(df_DTI['edges'] , 
                                self.drugs_features, 
                                self.targets_features,
                                self.drug_ids,
                                self.target_ids
                                ) 
        y = df_DTI['y']                  
               
        # Convert to numpy arrays
        X = np.array(X_df.drop(columns=self.id_label_columns))
        y = np.array(y)     
        sm = SMOTE(random_state=42)
        X_resampled, y_resampled = sm.fit_resample(X, y)   
        
        feature_names = X_df.drop(columns=self.id_label_columns).columns.to_list()    
        kf = KFold(n_splits=self.k_fold, 
                   shuffle=True, random_state=41)
        
        for fold, (train_index, valid_index) in enumerate(kf.split(X_resampled, y_resampled)):
            start_time = time.time() 
            print(f"\n=== Fold {fold + 1} ===")
            X_train, X_valid = X_resampled[train_index], X_resampled[valid_index]
            y_train, y_valid = y_resampled[train_index], y_resampled[valid_index]   
            model = clf.fit(X_train, y_train)
            importances_init = model.feature_importances_
            top_idx_init = np.argsort(importances_init)[::-1][:300] 
            X_train_subset = X_train[:, top_idx_init]
            X_valid_subset = X_valid[:, top_idx_init]
            
            # model.score(X_valid, y_valid)
            print("train model again with 300 features...")
            model = clf.fit(X_train_subset, y_train)
            r = permutation_importance(model, 
                                       X_valid_subset,
                                       y_valid,
                                       n_repeats=5,
                                       random_state=0)
            top_k = 100
            top_indices = r.importances_mean.argsort()[::-1][:top_k]
            selected_features_idx = top_idx_init[top_indices]
            print(f"\nTop {top_k} important features:")
            for idx in top_indices:
                print(f"{feature_names[idx]:<40} {r.importances_mean[idx]:.4f} +/- {r.importances_std[idx]:.4f}")
            
            print("train model with final selected features...")
            X_train_top = X_train[:, selected_features_idx]
            X_valid_top = X_valid[:, selected_features_idx]
            X_test_top = X_test[:, selected_features_idx]
            clf.fit(X_train_top, y_train)
            train_metrics = calculate_classification_metrics(clf,
                                                          clf_name,
                                                          X_train_top, y_train) 
            valid_metrics = calculate_classification_metrics(clf,
                                                              clf_name,
                                                              X_valid_top,
                                                              y_valid)    
            test_metrics = calculate_classification_metrics(clf,
                                                 clf_name, 
                                                 X_test_top,
                                                 y_test, 
                                                 plot_roc=True,
                                                 label ='Test results')
           
           
            train_results.append(['Permutation'] + list(train_metrics.values()))
            valid_results.append(['Permutation'] + list(valid_metrics.values()))
            test_results.append(['Permutation'] + list(test_metrics.values())) 
            end_time = time.time()           
            elapsed_time = end_time - start_time
            file_name = os.path.join(cfg.run_dir, 'elapsed_time.txt') 
            with open(file_name, "a") as file:              
                file.write(f"The loop took {elapsed_time:.4f} seconds to run {clf_name}.\n")  # Add \n for new line
              
        
        train_results_df = pd.DataFrame(train_results, columns=self.metrics_columns)   
        valid_results_df = pd.DataFrame(valid_results, columns=self.metrics_columns)
        test_results_df = pd.DataFrame(test_results, columns=self.metrics_columns)
        
        train_results_df.to_csv(self.cur_dir +'\\train.csv', index=False)
        valid_results_df.to_csv(self.cur_dir +'\\valid.csv', index=False)         
        test_results_df.to_csv(self.cur_dir +'\\test.csv', index=False)               
 
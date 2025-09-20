# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:10:15 2024

@author: leila Jafari Khouzani
"""
import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from utilities.config import cfg, create_params_yaml
from utilities.classification import combine_features, choose_classifier, define_classifiers
from utilities.calculate_metrics import calculate_classification_metrics
#==============================================================================
# class 
#==============================================================================
class CompareFeatureSelectionMethods:
    def __init__(self, builder, feature_selection_method='Decision Tree'):
        self.data_test = None
        self.data_splits = None
        self.drugs_features = builder.drugs_dic['normalized_features']
        self.targets_features = builder.targets_dic['normalized_features']
        self.df_edges = builder.DTI    
        self.n_component  = cfg.feature.max_n_components
        self.feature_selection_method = feature_selection_method                
        self.builder = builder
        self.drug_ids = list(self.builder.drugs_dic['ids'])
        self.target_ids = list(self.builder.targets_dic['ids'])   
        self.test_drug_ids =  list(self.builder.test_drugs_dic['ids'])
        self.test_target_ids = list(self.builder.test_targets_dic['ids'])
        self.k_fold = cfg.model.n_splits   
        self.id_label_columns = [cfg.dataset.drug_label,
                                 cfg.dataset.target_label,
                                 cfg.dataset.ActionLabel]
        self.metrics_columns = ['Feature Combination Method',
                                'AverageLoss',
                                'Accuracy',
                                'Precision',
                                'Recall',
                                'F1',
                                'MCC',
                                'ROC_AUC',
                                '0_precision',
                                '0_recall',
                                '1_f1',
                                '1_precision',
                                '1_recall',
                                '2_f1',
                                '2_precision',
                                '2_recall',
                                '2_f1']
    # ============================================================================= 
    def create_train_valid_test(self):
        os.makedirs(cfg.run_dir, exist_ok=True)
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

        # Dictionary to store data splits
        self.data_splits = {}
        self.data_test = {}
        print("split builder to train and valid...")
        kf = KFold(n_splits=self.k_fold,
                   shuffle=True, random_state=41)
        sm = SMOTE(random_state=42)
        for fold, (train_index, valid_index) in enumerate(kf.split(X, y)):
            print(f"fold {fold}")
            self.data_splits[fold] = {}
            X_train, X_valid = X[train_index], X[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            X_resampled, y_resampled = sm.fit_resample(X_train, y_train)
            top_indices, feature_importance = feature_selection(X_resampled,
                                                                y_resampled,
                                                                self.n_component,
                                                                method=self.feature_selection_method)

            self.data_splits[fold] = {
                'X_train': X_resampled,
                'y_train': y_resampled,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'feature_importance': feature_importance,
                'top_indices': top_indices
                }
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
            X_resampled, y_resampled = sm.fit_resample(X, y)
            top_indices, feature_importance = feature_selection(X_resampled,
                                                                y_resampled,
                                                                self.n_component,
                                                                method=self.feature_selection_method) 
            self.data_test = {
                'X': X_resampled, 
                'y': y_resampled,
                'X_test': X_test,
                'y_test': y_test,
                'feature_importance': feature_importance,
                'top_indices': top_indices
                }
    # ============================================================================= 
    def train(self):

        train_results = []
        valid_results = []  
        test_results = [] 
        X_test = self.data_test['X_test']
        y_test = self.data_test['y_test']        
        X = self.data_test['X']
        y = self.data_test['y']
        test_top_indices = self.data_test['top_indices']
        test_importances = self.data_test['feature_importance']
        test_weights = test_importances[test_top_indices]
        # for method in self.feat_sel:            
        for fold in range(0, self.k_fold):                 
            X_train = self.data_splits[fold]['X_train']
            y_train =self.data_splits[fold]['y_train']
            X_valid = self.data_splits[fold]['X_valid']
            y_valid = self.data_splits[fold]['y_valid']                
            top_indices = self.data_splits[fold]['top_indices']
            importances = self.data_splits[fold]['feature_importance']

            X_train_top = X_train[:, top_indices]
            X_valid_top = X_valid[:,  top_indices]                
            X_test_top = X_test[:,  top_indices]               
            
            weights = importances[top_indices]  # shape: (top_k,)

            X_train_weighted = X_train_top * weights
            X_valid_weighted = X_valid_top * weights
            X_test_weighted = X_test_top * weights
            # Train and predict on train/validation fold                   
            self.clf.fit(X_train_weighted, y_train)
            train_metrics = calculate_classification_metrics(self.clf, 
                                           self.clf_name,
                                           X_train_weighted, y_train) 
            valid_metrics = calculate_classification_metrics(self.clf, 
                                          self.clf_name,
                                          X_valid_weighted,
                                          y_valid)  
            
            train_results.append([self.feature_selection_method] + list(train_metrics.values()))
            valid_results.append([self.feature_selection_method] + list(valid_metrics.values()))
        
                
            test_metrics = calculate_classification_metrics(self.clf,
                                             self.clf_name, 
                                             X_test_weighted,
                                             y_test, 
                                             plot_roc=True,
                                             label ='test results')
            test_results.append([self.feature_selection_method] + list(test_metrics.values()) )
        train_results_df = pd.DataFrame(train_results, columns=self.metrics_columns)   
        valid_results_df = pd.DataFrame(valid_results, columns=self.metrics_columns)
        test_results_df = pd.DataFrame(test_results, columns=self.metrics_columns)
        train_results_df.to_csv(self.cur_dir +'\\train.csv', index=False)
        valid_results_df.to_csv(self.cur_dir +'\\valid.csv', index=False)         
        test_results_df.to_csv(self.cur_dir +'\\test.csv', index=False)
    # =============================================================================
    def run(self):
        self.classifiers = define_classifiers()
        self.clf_name, self.clf = choose_classifier(self.classifiers)
        self.cur_dir = create_params_yaml(cfg, f'{self.clf_name}')
        os.makedirs(cfg.run_dir + "\\test results", exist_ok=True)
        self.create_train_valid_test()
        self.train()
# =============================================================================        
def feature_selection(X_train, y_train, n_components, method):
    feature_importance = {}
    top_indices = {}
    if method == 'Decision Tree':
        tree = DecisionTreeClassifier()
        tree.fit(X_train, y_train)
        importances = tree.feature_importances_
        feature_importance['Decision Tree']=importances 
        top_indices['Decision Tree'] = np.argsort(importances)[::-1][:n_components]
    elif method == 'Random Forest':        
        forest = RandomForestClassifier()
        forest.fit(X_train, y_train)
        importances = forest.feature_importances_
        feature_importance['Random Forest'] = importances 
        top_indices['Random Forest'] = np.argsort(importances)[::-1][:n_components]
    else:
        raise ValueError(f"Unknown reduce method '{method}' specified.")

    sorted_indices = np.argsort(importances)[::-1]  # indices of features sorted by importance (descending)
    sorted_importances = importances[sorted_indices]
    cumulative_importance = np.cumsum(sorted_importances)
    threshold=0.95
    n_selected = np.searchsorted(cumulative_importance, threshold) + 1
    print(f'number of 0.95 variance features are {n_selected}')
    return sorted_indices[:n_selected], importances
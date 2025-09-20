# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 11:49:44 2025

@author: Leila Jafari Khouzani
"""

import os,  time
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import  KNeighborsClassifier
from sklearn.ensemble import (RandomForestClassifier, 
                             GradientBoostingClassifier,
                             AdaBoostClassifier,
                             VotingClassifier, 
                             StackingClassifier,
                             ExtraTreesClassifier,
                             HistGradientBoostingClassifier)
from utilities.config import cfg, create_params_yaml
from utilities.calculate_metrics import (calculate_classification_metrics,
                                         evaluate_on_test, save_results)
# =============================================================================
#  class 
# =============================================================================  
class ClassifierEvaluator:
    def __init__(self, builder):
        self.builder = builder
        # Dictionary to store data splits and partitions
        self.data_partitions = {}
        self.data_splits = {}
        self.X = None
        self.y = None
        self.X_test = None
        self.y_test = None
        self.drug_ids = None
        self.target_ids = None
        self.test_drug_ids = None
        self.test_target_ids = None
        self.clf = None
        self.clf_name = None
        self.cur_dir = None
        self.test_results = None
        self.classifiers = define_classifiers()
    # =============================================================================
    def process_features(self):
        """
        Processes the features according to the object's configuration (reduce and combine methods).
        Returns:
            X_train, y_train, X_test, y_test as numpy arrays.
        """
        drugs_features = self.builder.drugs_dic['normalized_features']
        targets_features = self.builder.targets_dic['normalized_features']
        test_drugs_features = self.builder.test_drugs_dic['normalized_features']
        test_targets_features = self.builder.test_targets_dic['normalized_features']

        if cfg.feature.reduce_method == 'pca':
            reduced_drugs_features, drug_reducer = pca_reduce_features(drugs_features,
                                                                       cfg.feature.drug_n_components,
                                                                       return_reducer=True)

            reduced_targets_features, target_reducer = pca_reduce_features(targets_features,
                                                                           cfg.feature.target_n_components,
                                                                           return_reducer=True)
            test_drug_reduced_feats = drug_reducer.transform(test_drugs_features)
            test_target_reduced_feats = target_reducer.transform(test_targets_features)
        else:
            reduced_drugs_features = drugs_features
            reduced_targets_features = targets_features
            test_drug_reduced_feats = test_drugs_features
            test_target_reduced_feats = test_targets_features
        # Combine reduced features
        df_train_valid = combine_features(self.builder.DTI['edges'],
                                          reduced_drugs_features,
                                          reduced_targets_features,
                                          self.drug_ids,
                                          self.target_ids)
        X_train_valid = df_train_valid.drop(columns=df_train_valid.columns[:3])
        y_train_valid = self.builder.DTI['y']

        df_test = combine_features(self.builder.test_DTI['edges'],
                                   test_drug_reduced_feats,
                                   test_target_reduced_feats,
                                   self.test_drug_ids,
                                   self.test_target_ids)
        X_test = df_test.drop(columns=df_test.columns[:3])
        y_test = self.builder.test_DTI['y']

        # Convert to numpy arrays
        self.X = np.array(X_train_valid)
        self.y = np.array(y_train_valid)
        self.X_test = np.array(X_test)
        self.y_test = np.array(y_test)
        self.data_partitions = {'X_train_valid': self.X,
                                'y_train_valid': self.y,
                                'X_test': self.X_test,
                                'y_test': self.y_test
                                }
    # =============================================================================
    def generate_data_splits(self):
        # Loop over repetitions
        for rep in range(cfg.model.n_repeats):
            self.data_splits[rep] = {}   
            kf = KFold(n_splits=cfg.model.n_splits, 
                       shuffle=True, random_state=rep)
            
            for fold, (train_index, valid_index) in enumerate(kf.split(self.X, self.y)):
                self.data_splits[rep][fold] = {}                
                X_train, X_valid = self.X[train_index], self.X[valid_index]
                y_train, y_valid = self.y[train_index], self.y[valid_index]
                
                # Save splits
                self.data_splits[rep][fold] = {'X_train': X_train,
                                                'y_train': y_train,
                                                'X_valid': X_valid,
                                                'y_valid': y_valid
                                                }
    # ============================================================================= 
    def evaluate_classifiers(self):
        file_name = os.path.join(cfg.run_dir, 'elapsed_time.txt')        
        if cfg.model.one_clf:
            start_time = time.time() 
            self.clf_name, self.clf = choose_classifier(self.classifiers)
            self.train_classifier()
            
            end_time = time.time()           
            elapsed_time = end_time - start_time
           
            with open(file_name, "a") as file:              
                file.write(f"The loop took {elapsed_time:.4f} seconds to run {self.clf_name}.\n")  # Add \n for new line
                logging.info(f"The loop took {elapsed_time:.4f} seconds to run {self.clf_name}.")
            
            logging.info(f'Finished classifier {self.clf_name}')
        else:            
            for self.clf_name, self.clf in self.classifiers.items():
                start_time = time.time() 
                self.train_classifier()  
                end_time = time.time()           
                elapsed_time = end_time - start_time
                with open(file_name, "a") as file:
                    file.write(f"The loop took {elapsed_time:.4f} seconds to run {self.clf_name}.\n")
                    logging.info(f"The loop took {elapsed_time:.4f} seconds to run {self.clf_name}.")
              
                logging.info(f'Finished classifier {self.clf_name}')     
    # ============================================================================= 
    def train_classifier(self):   
        self.cur_dir = create_params_yaml(cfg, f'{self.clf_name}')           
        os.makedirs(cfg.run_dir + "\\test results", exist_ok=True)
    
        train_results = []
        valid_results = []  
        test_results = []        
        self.test_results = []
        param_dict = {}
        for rep in self.data_splits:
            for fold in self.data_splits[rep]:  
                logging.info(fold)
    
                # Explicit parameter dictionary with clear keys
                param_dict = {
                    'Classifier': self.clf_name,
                    'repetition': rep,
                    'Fold': fold,
                    'drug_n_components': cfg.feature.drug_n_components,
                    'protein_n_components': cfg.feature.target_n_components,
                    'Feature Combination Method': cfg.feature.DTCombine
                }
    
                X_train = self.data_splits[rep][fold]['X_train']
                y_train = self.data_splits[rep][fold]['y_train']
                X_valid = self.data_splits[rep][fold]['X_valid']
                y_valid = self.data_splits[rep][fold]['y_valid']
    
                # Train and evaluate on the training fold
                self.clf.fit(X_train, y_train)
                train_metrics_row = calculate_classification_metrics(
                                    self.clf, self.clf_name,
                                    X_train, y_train
                                    )
                train_results.append({**param_dict, **train_metrics_row})
    
                # Evaluate on the validation fold
                valid_metrics_row = calculate_classification_metrics(
                                                self.clf, self.clf_name,
                                                X_valid, y_valid
                                                )
                valid_results.append({**param_dict, **valid_metrics_row})
    
            print(f'finished {self.clf_name} repetition {rep}')
            logging.info(f'finished {self.clf_name} repetition {rep}')
    
            # Retrain on train+valid for testing
            self.clf.fit(self.data_partitions['X_train_valid'],
                         self.data_partitions['y_train_valid'])
                      
            model_path = os.path.join(self.cur_dir,
                                      f"{self.clf_name}_trained_model.joblib")
            #save trained classifier 
            joblib.dump(self.clf, model_path)
            
            print(f" model {self.clf_name} saved in : {model_path}")
    
            y_pred, test_metrics_row = calculate_classification_metrics(
                                        self.clf,
                                        self.clf_name,
                                        self.data_partitions['X_test'],
                                         self.data_partitions['y_test'],
                                        plot_roc=True,
                                        y_pred_return=True,
                                        label='test results'
                                    )
            test_results.append({**param_dict, **test_metrics_row})
    
            evaluate_on_test(self.builder, y_pred)
       
        save_results(train_results, self.cur_dir, self.clf_name, 'train')
        save_results(valid_results, self.cur_dir, self.clf_name,'valid')
        save_results(test_results, self.cur_dir, self.clf_name,'test')    
        
        logging.info(f'finished general test of {self.clf_name}')
    # =============================================================================
    def run(self):
        """Run the full evaluation pipeline."""
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.drug_ids = list(self.builder.drugs_dic['ids'])
        self.target_ids = list(self.builder.targets_dic['ids'])
        self.test_drug_ids = list(self.builder.test_drugs_dic['ids'])
        self.test_target_ids = list(self.builder.test_targets_dic['ids'])
        self.process_features()
        self.generate_data_splits()
        self.evaluate_classifiers()
# ============================================================================= 
def define_classifiers():
    random_state = cfg.model.random_state
    classifiers =\
        {
          "DecisionTree": DecisionTreeClassifier(random_state=random_state), 
          "KNN":KNeighborsClassifier(n_neighbors=5),  
          "MLP": MLPClassifier(hidden_layer_sizes=(100,),
                              max_iter=cfg.model.max_iter,
                              random_state=random_state),                            
                                           
          "SVM":SVC(kernel='rbf', probability=True, C=1.0, 
                    max_iter=cfg.model.max_iter, random_state=random_state), 
                              
         "ExtraTrees":ExtraTreesClassifier(n_estimators=cfg.model.n_estimators, 
                                  random_state=random_state),
         "RandomForest":RandomForestClassifier(n_estimators=cfg.model.n_estimators,
                                              random_state=random_state),
          "AdaBoost": AdaBoostClassifier(algorithm='SAMME', 
                                        n_estimators=cfg.model.n_estimators, 
                                        random_state=random_state), 
         "GradientBoosting":GradientBoostingClassifier(n_estimators=cfg.model.n_estimators, 
                                                      random_state=random_state),             
                                        
         "HGBoosting":HistGradientBoostingClassifier(max_iter=cfg.model.max_iter, 
                                                    random_state=random_state)                           
       
          }
    estimators = [('dt', classifiers["DecisionTree"]),
                    ('kn', classifiers["KNN"]), 
                    ('mlp', classifiers["MLP"]),
                    ('et', classifiers["ExtraTrees"])
                        ]
    final_estimator = LogisticRegression()     
     
    ensemble_clfs = {
            "Stacking":StackingClassifier(estimators=estimators,
                                       final_estimator=final_estimator),                                  
            "Voting":VotingClassifier(estimators= estimators, voting='soft')
                                       }   
    classifiers = classifiers | ensemble_clfs   
    return classifiers
# ============================================================================
def choose_classifier(classifiers):
    """
    Chooses a classifier from cfg.model.classifier_types.
      Returns:
        str: The chosen classifier.
    """
    print("Please choose a classifier from the list below:")
    for i, clf in enumerate(cfg.model.classifier_types):
        print(f"{i}: {clf}")
    choice = int(input("Enter the number corresponding to your classifier choice: "))
    try:
        chosen_clf = cfg.model.classifier_types[choice]
        print(f"You have chosen: {chosen_clf}")
        clf = classifiers[chosen_clf]
    except ValueError:
        print("Invalid choice. Please enter a valid number from the list.")
        return choose_classifier(classifiers)  # Retry
    return chosen_clf, clf
# ==============================================================================
# Function:  Perform 1D convolution between drug feature and protein feature or concatenation
# ==============================================================================
def combine_features(edges, drugs_features, targets_features,
                     drug_ids, target_ids, combine_method=cfg.feature.DTCombine):
    """
    Combines drug and target features using the specified method.

    drug_features (numpy): Array of drug features.
    target_features (numpy).Array of target features.

    combine_method (str): Method to combine features ('concat', 'convolve').

    Returns:
    Dataframe: Combined feature matrix.        """

    # =============================================================================
    # Create a dictionary that maps the method name to the corresponding function
    methods = {
        'convolve': lambda x, y: np.convolve(x, y, mode=cfg.feature.PCA_mode),

        'concat': lambda x, y: np.concatenate((x, y))
    }
    # =============================================================================
    id_label_columns = [cfg.dataset.drug_label,
                        cfg.dataset.target_label,
                        cfg.dataset.ActionLabel]
    if combine_method in methods:
        interaction_data = []

        for index, row in edges.iterrows():
            cur_drug = row[cfg.dataset.drug_label]
            cur_target = row[cfg.dataset.target_label]
            cur_y = row[cfg.dataset.ActionLabel]

            drug_idx = drug_ids.index(cur_drug)
            target_idx = target_ids.index(cur_target)

            cur_drug_features = drugs_features[drug_idx]
            cur_target_features = targets_features[target_idx]

            # Use the dictionary to call the appropriate function based on the method
            combined_features = methods.get(combine_method)(cur_drug_features, cur_target_features)

            # Add the row to interaction_data list (DrugID, UniprotID, y, and combined features)
            interaction_data.append([cur_drug, cur_target, cur_y] + list(combined_features))

        # Define the column names for the DataFrame
        num_features = len(combined_features)  # Number of features in the combined vector
        feature_columns = [f'Feature_{i + 1}' for i in range(num_features)]
        columns = id_label_columns + feature_columns

        # Create a DataFrame from interaction_data
        interaction_df = pd.DataFrame(interaction_data, columns=columns)

    else:
        raise ValueError("Unknown combination method specified.")
    return interaction_df
# =============================================================================
def pca_reduce_features(x_train, n_components,  return_reducer=False):
    """
    Reduces features using PCA method.
    Parameters:
    - x_train (numpy.ndarray): The feature matrix to reduce.
    - n_components (int): Number of components to retain.
    Returns:
    - numpy.ndarray: Reduced feature matrix.
    """
    reducer = PCA(n_components=n_components)
    if n_components<x_train.shape[1]:
        reduced_features = reducer.fit_transform(x_train)
    else:
        reduced_features = x_train
    if return_reducer:
        return reduced_features, reducer
    return reduced_features
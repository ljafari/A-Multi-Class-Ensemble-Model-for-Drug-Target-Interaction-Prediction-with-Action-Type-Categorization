# -*- coding: utf-8 -*-
"""
End-to-End HeteroGNN DPI Pipeline (Train, Validation, Test, Logging, Plots)
Created on Sat Jun 28 2025
Author: laila jafari (clean structured integration)
"""
# -------------------- CONFIG LOADING --------------------
import os
import platform
import logging
import inspect
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import KFold
import torch.nn.functional as functional
from utilities.config import cfg, load_cfg, dump_cfg, set_cfg, create_params_yaml
from utilities.DatasetBuilder import DatasetBuilder
from utilities.cmd_args import parse_args
from utilities.graph_models import (HeteroGNN,  HeteroGraphSAGE,
                                    HeteroGAT, HeteroHGT, HeteroHAN, 
                                     HeteroGraphTransformer)
from utilities.calculate_metrics import (calculate_metrics, prepare_metrics_for_saving, 
                                         save_results, loss_categorical_cross_entropy)

# =============================================================================
#  class 
# =============================================================================  
class HetroGNNEvaluator:
    def __init__(self, builder):
        # -------------------- SETUP --------------------
        self.model_class = None
        self.model_name = None
        self.fold = 0
        self.rep = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

        self.model_choices = {
            "HeteroGNN": HeteroGNN,
            "HeteroGAT": HeteroGAT,
            "HeteroGraphSAGE": HeteroGraphSAGE,
              "HeteroHGT": HeteroHGT,
              "HeteroHAN": HeteroHAN,
              "HeteroGraphTransformer": HeteroGraphTransformer
        }
        # -------------------- DATASET --------------------
        self.in_dir = cfg.in_dir
        self.builder = builder
        self.data, self.data_dic = self.builder.build_hetero_graph()     
        self.test_data, self.test_data_dic =  self.builder.build_hetero_graph( device=self.device)
      
        self.num_classes = int(self.data_dic['edge_labels'].max().item() - self.data_dic['edge_labels'].min().item() + 1)
        self.num_edges = self.data_dic['edge_index'].shape[1]   
        self.node_feature_dims =  {
                  node_type: self.data.x_dict[node_type].shape[1]
                  for node_type in self.data.x_dict.keys()
              }

       
        self.kf = KFold(n_splits=cfg.model.n_splits, shuffle=True, random_state=42)
        self.num_epochs = cfg.model.epochs
        self.num_reps = cfg.model.n_repeats
        self.test_edges = self.builder.test_DTI
        self.drug_ids = self.builder.test_drugs_dic['ids']
        self.protein_ids = self.builder.test_targets_dic['ids']
        self.drug_id_to_index = {id_: idx for idx, id_ in enumerate(self.drug_ids)}
        self.prot_id_to_index = {id_: idx for idx, id_ in enumerate(self.protein_ids)}
        self.parent_dir= cfg.run_dir   
            
    # =============================================================================      
    def calculate_gnn_metrics(self, edge_index,
                              y_true,
                              test_mode=False,
                               save_results_flag=False,
                              plot_roc=False,
                              plot_label=''):
        """
        Calculate evaluation metrics for GNN predictions.        
        Args:
            edge_index: Edge indices for prediction
            y_true: True labels            
            test_mode: Whether this is test evaluation
            save_results_flag: Whether to save results to files
            plot_label:
            plot_roc:
        Returns:
            Dictionary containing metrics and (optionally) saves results
        """
        # Set up model evaluation
        self.model.eval()
        
        # Get the appropriate data source
        eval_data = self.test_data if test_mode else self.data
        device = next(self.model.parameters()).device
        
        #Move data to device if needed
        if test_mode:
            for key in eval_data.x_dict:
                eval_data.x_dict[key] = eval_data.x_dict[key].to(device)
            for key in eval_data.edge_index_dict:
                eval_data.edge_index_dict[key] = eval_data.edge_index_dict[key].to(device)
        
        with torch.no_grad():
            out = self.model(eval_data.x_dict, eval_data.edge_index_dict, edge_index)
            y_pred_proba = functional.softmax(out, dim=1).cpu().numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true_np = y_true.cpu().numpy()

        loss = loss_categorical_cross_entropy(y_true, y_pred_proba, one_hot=False)
        metrics_dict = calculate_metrics(y_true_np, y_pred, y_pred_proba, 
                                           average=cfg.metrics.average,
                                           loss=loss,
                                           plot_roc=plot_roc, 
                                           plot_suffix=self.model_name, 
                                           plot_label=plot_label)
       
        metrics_flat = prepare_metrics_for_saving(metrics_dict)
        
        # Prepare parameter dictionary
        param_dict = {
            'Classifier': self.model_name,
            'repetition':  self.rep,
            'Fold': self.fold,
            'hidden_dim': 64  # Add other model parameters as needed
        }
        
        row = {**param_dict, **metrics_flat}
        
        # Additional test evaluation steps
        if test_mode and save_results_flag:
            print("\n🔹 Evaluating on test Test Data")
            # Save detailed predictions
            results_df = pd.DataFrame({
                                    "DrugID": self.test_data_dic['d_list'],
                                    "UniprotID": self.test_data_dic['p_list'],
                                    "TrueLabel": y_true_np,
                                    "PredLabel": y_pred
                                })
            results_df.to_csv(os.path.join(cfg.run_dir, 
                                         "test_predictions.csv"),
                                          index=False)
        return row
    # =============================================================================  
    def run(self):       
        
        for self.model_name, self.model_class in self.model_choices.items():
            print(f"\nRunning model: {self.model_name}")
            cfg.model.classifier = self.model_name
            create_params_yaml(cfg, suffix=self.model_name , filename=None)
            print(f"\nResults will be saved in: {cfg.run_dir}")   
            
            logging.basicConfig(filename=os.path.join(cfg.run_dir, "training_log.txt"),
                               level=logging.INFO,
                               format="%(asctime)s [%(levelname)s] %(message)s")
            
            init_args = inspect.signature(self.model_class.__init__).parameters
            model_kwargs = dict(
                metadata=self.data.metadata(),
                hidden_dim=64,
                out_dim=self.num_classes
                )
            
            if 'node_feature_dims' in init_args:
                model_kwargs['node_feature_dims'] = self.node_feature_dims
                      
            train_results = []
            valid_results = []  
            test_results = []
            for self.rep in range(0, self.num_reps):
                self.model = self.model_class(**model_kwargs).to(self.device)
                optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)  
                print(f"\n🔹 rep {self.rep+1}")
                for self.fold, (train_idx, valid_idx) in enumerate(self.kf.split(torch.arange(self.num_edges))):
                    print(f"\n🔹 Fold {self.fold+1}")                    
                    train_edge_index = self.data_dic['edge_index'][:, train_idx]
                    train_labels = self.data_dic['edge_labels'][train_idx]
                    valid_edge_index = self.data_dic['edge_index'][:, valid_idx]
                    valid_labels = self.data_dic['edge_labels'][valid_idx]   
                    # Move data to device
                    for key in self.data.x_dict:
                        self.data.x_dict[key] = self.data.x_dict[key].to(self.device)
                    for key in self.data.edge_index_dict:
                        self.data.edge_index_dict[key] = self.data.edge_index_dict[key].to(self.device)
                    train_edge_index, train_labels = train_edge_index.to(self.device), train_labels.to(self.device)
                    valid_edge_index, valid_labels = valid_edge_index.to(self.device), valid_labels.to(self.device)
                    
                    # ------------------------ TRAINING ------------------------  
                    best_acc, patience, patience_counter = 0, 10, 0
                    for epoch in range(1, self.num_epochs+1):
                        
                        # ----- Training -----
                        self.model.train()
                        optimizer.zero_grad()
                        out = self.model(self.data.x_dict, self.data.edge_index_dict, train_edge_index)
                        loss = functional.cross_entropy(out, train_labels)
                        loss.backward()
                        optimizer.step()
                        
                        # Evaluate
                        self.model.eval()
                        with torch.no_grad():                            
                            train_metrics = self.calculate_gnn_metrics(train_edge_index, train_labels)
                            
                            val_metrics = self.calculate_gnn_metrics(valid_edge_index, valid_labels)
                        # Logging
                        log_msg = (
                                    f"Epoch {epoch:03d} | "
                                    f"Loss: {val_metrics['overall_loss']:.4f} | "
                                    f"Val Acc: {val_metrics['overall_accuracy']:.4f} | "                                   
                                  )

                        logging.info(log_msg)
                        print(log_msg)  
                                  
                        # Early Stopping
                        if val_metrics['overall_accuracy'] > best_acc:
                            best_acc = val_metrics['overall_accuracy']
                            best_val_metrics = val_metrics
                            best_train_metrics = train_metrics
                            patience_counter = 0
                            torch.save(self.model.state_dict(), os.path.join(cfg.run_dir, "best_model.pt"))
                        else:
                            patience_counter += 1
             
                        if patience_counter >= patience:
                            logging.info(f"Early stopping at epoch {epoch}. Best Val Acc: {best_acc:.4f}")
                            print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_acc:.4f}")
                            break
                    train_results.append({**best_train_metrics})
                    valid_results.append({**best_val_metrics})
                    # --------------------  TEST --------------------
                    test_metrics = self.calculate_gnn_metrics(
                                    edge_index= self.test_data_dic['edge_index'],
                                    y_true=self.test_data_dic['edge_labels'],
                                    test_mode=True,
                                    save_results_flag=True
                                )
                    test_results.append({**test_metrics})
                # -------------------- FINAL EVALUATION --------------------
                self.model.load_state_dict(torch.load(os.path.join(cfg.run_dir, "best_model.pt")))
                self.model.eval()
                
                train_final_metrics = self.calculate_gnn_metrics(train_edge_index, train_labels)
                plot_roc = False
                if self.rep==0:
                    plot_roc = True               
                val_final_metrics = self.calculate_gnn_metrics(valid_edge_index, valid_labels,
                                                               plot_roc=plot_roc, plot_label='valid')
                
                print("\n=== Final Metrics ===")
                print("Train:", train_final_metrics)
                print("Validation:", val_final_metrics)   
                self.fold = +1
                train_results.append({**train_final_metrics})
                valid_results.append({**val_final_metrics})
                # --------------------  TEST --------------------
                test_metrics = self.calculate_gnn_metrics(
                                edge_index= self.test_data_dic['edge_index'],
                                y_true=self.test_data_dic['edge_labels'],
                                test_mode=True,
                                save_results_flag=True,
                                plot_roc=plot_roc, 
                                plot_label='test_results'
                            )
                test_results.append({**test_metrics})
                
            # -------------------- SAVE LOGS & PLOTS --------------------   
            save_results(train_results, cfg.run_dir, 
                         clf_name=self.model_name, label='train_gnn')
            save_results(valid_results, cfg.run_dir, 
                         clf_name=self.model_name, label='valid_gnn')  
            save_results(test_results, cfg.run_dir,
                         clf_name=self.model_name, label='test_gnn')
        
# ---------------------------- MAIN ----------------------------
if __name__ == '__main__':
    if platform.system() == "Windows":
        cfg = set_cfg(cfg)
    elif platform.system() == "Linux":
        args = parse_args()
        load_cfg(cfg, args)
    dump_cfg(cfg)
    builder = DatasetBuilder()
    builder.create_dataset()
    gnn_evaluator = HetroGNNEvaluator(builder)
    gnn_evaluator.run()

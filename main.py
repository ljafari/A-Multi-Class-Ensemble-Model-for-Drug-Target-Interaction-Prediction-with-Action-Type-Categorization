"""
Created on Wed May  1 15:27:53 2024

@author: Leila Jafari Khouzani
"""
import logging
import platform
from importlib import reload

from utilities.config import cfg, load_cfg, dump_cfg, set_cfg
from utilities.cmd_args import parse_args
from utilities.DatasetBuilder import DatasetBuilder
from utilities.classification import ClassifierEvaluator
from utilities.compare_feature_selection_methods import CompareFeatureSelectionMethods
from utilities.PermutationFeatureImportance import PermutationFeatureImportance
from utilities.AE_trainer import train_autoencoder_and_export
from utilities.HeteroGNN import HetroGNNEvaluator
# =============================================================================
#                           main:   
# =============================================================================   
if __name__ == '__main__':  
    
    if platform.system() == "Windows":     
        cfg = set_cfg(cfg)       
        
    elif platform.system() == "Linux":         
        # restart logging file
        reload(logging)           
        # Load cmd line args
        args = parse_args()        
        # Load config file
        load_cfg(cfg, args)
    dump_cfg(cfg)
    builder = DatasetBuilder()
    builder.create_dataset()
    cfg.model.task = 'Classifier Evaluation'
    if cfg.model.task == 'Classifier Evaluation': 
         evaluator = ClassifierEvaluator(builder)
         evaluator.run()
    elif cfg.model.task == 'Feature Selection Analysis':
        obj = CompareFeatureSelectionMethods(builder,
                                             cfg.feature.selection_method)
        obj.run()
    elif cfg.model.task == 'Create AutoEncoder Embedding':
        bottleneck = getattr(cfg.model, "embedding_dim", 128)  #
        train_autoencoder_and_export(builder, cfg, bottleneck=bottleneck)
    elif cfg.model.task == 'Permutation Feature Importance':
        PermutationFeatureImportance(builder)
    elif cfg.model.task =="Evaluate GNN methods":
            gnn_evaluator = HetroGNNEvaluator(builder)
            gnn_evaluator.run()

        
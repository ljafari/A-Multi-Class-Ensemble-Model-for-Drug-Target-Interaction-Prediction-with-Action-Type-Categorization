"""
Created on Mon Sep  2 22:11:18 2024

@author: leila Jafari Khouzani
"""
import os, time
import datetime
from yacs.config import CfgNode as CN
import utilities.register as register

# Global config object
cfg = CN()

# =============================================================================
def set_cfg(cfg):   
   
    if cfg is None:
        return cfg

    # ----------------------------------------------------------------------- #
    # Basic options
    # ----------------------------------------------------------------------- #
    # Set print destination: stdout / file / both
    cfg.print = 'both'   
    #*************************************************************************#
    #input directory
    cfg.in_dir = 'dataset'    

    #*************************************************************************#
    # Output directory
    cfg.out_dir = 'results'
  
    #*************************************************************************#
    # Config name (in out_dir)
    cfg.cfg_dest ='config.yaml'
    
    current_time = datetime.datetime.now()   
    
    # Random current time
    cfg.date = os.path.join(str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day))
    
    cfg.time = os.path.join(str(current_time.hour)  + '-' + str(current_time.minute))
    
    # cfg.time = str(current_time.year) + '-' + str(current_time.month) + '-' + str(current_time.day)
    # # Print rounding
    # cfg.round = 4
    
    # ----------------------------------------------------------------------- #
    # Dataset files
    # ----------------------------------------------------------------------- #  
    cfg.dataset = CN()
    cfg.dataset.DrugDiscriptors = 'DrugDiscriptors.csv'
    cfg.dataset.ProteinFeatures = 'ProteinFeatures.csv'
    cfg.dataset.DTI =  'DTI.csv'
    cfg.dataset.DDS = 'DDS.csv'
    cfg.dataset.PPI = 'PPI.csv'
    cfg.dataset.drug_label = 'DrugID'
    cfg.dataset.target_label = 'UniprotID'
    cfg.dataset.ActionLabel = 'ActionLabel'
    cfg.dataset.ActioType = 'ActionType'
    cfg.dataset.class_labels = [0, 1, 2]
    cfg.dataset.zero_edges = 'zero_thr0.5.csv'
    # ----------------------------------------------------------------------- #
    cfg.feature = CN()
    
    cfg.feature.reduce_methods = ['None',#10,
                                  'pca',
                                  'Decision Tree',
                                  'Random Forest'
                                  ]
    cfg.feature.selection_method = 'Decision Tree'
     
    cfg.feature.combination_methods = ['concat',
                                       'convolve'] 
    
    cfg.feature.DTCombine = 'concat'
    
    cfg.feature.reduce_method = 'pca'
    
    cfg.feature.PCA_mode = 'full' #same valid
    
    cfg.feature.max_n_components  = 200
    
    cfg.feature.drug_n_components = 20
    
    cfg.feature.target_n_components = 20
    # ----------------------------------------------------------------------- #
    # Model options
    # ----------------------------------------------------------------------- #
    cfg.model = CN()     
    
    cfg.model.tasks = ['Classifier Evaluation',
                       'Feature Selection Analysis',
                       'Permutation Feature Importance', 
                       'Evaluate GNN methods',
                       'Create AutoEncoder Embedding']
    cfg.model.task = 'Classifier Evaluation'
    
    cfg.model.one_clf = True
    
    cfg.model.classifier_types = ['DecisionTree', #0
                                  'KNN', #1
                                  'MLP', #2
                                  'SVM', #3
                                  'ExtraTrees',#4
                                  'RandomForest',#5
                                  'AdaBoost', #9
                                  'GradientBoosting', #6
                                  'HGBoosting', #7
                                  'Stacking',#8,
                                  'Voting'#9
                                  ]
    
    cfg.model.classifier = cfg.model.classifier_types[4]
    
    cfg.model.max_iter = 2000 
    
    cfg.model.alpha = 0.01 
    
    cfg.model.n_estimators = 100   
    
    cfg.model.knn_neighbors = 5

    cfg.model.output_dim = 3
 
    cfg.model.epochs = 200
    
    cfg.model.n_splits = 5 
    
    cfg.model.test_size = 0.2
    
    cfg.model.valid_size = 0.2
    
    cfg.model.n_repeats = 1
    
    cfg.model.random_state_seed = 'time'# 42       
    
    cfg.model.random_state = 'time'#42
   
    cfg.model.embedding_dim = 128

    cfg.model.lr = 5e-4

    cfg.data_path = "dataset\\preprocessed"
    # ----------------------------------------------------------------------- #
    #  results 
    # ----------------------------------------------------------------------- #
    cfg.metrics = CN()
   
    cfg.metrics.average = 'weighted' #'macro' # weighted
    
    cfg.metrics.columns = ['Classifier', # Name of the classifier
                           'repetition', # Repetition number                
                           'Fold',       # Fold number
                           'drug_n_componenets', 
                           'protein_n_componenets',
                           'Feature Combination Method',
                           'Loss', #
                           'Accuracy',   # Accuracy score
                           'Precision',  # Precision score
                           'Recall',     # Recall score
                           'F1',         # F1 score
                           'MCC',        # MCC score
                           'ROC_AUC'     # ROC-AUC score
                           ]   
    # ----------------------------------------------------------------------- #
    # Set user customized cfgs
    # ----------------------------------------------------------------------- #
    for func in register.config_dict.values():
       func(cfg)

    return(cfg)
# =============================================================================
def dump_cfg(cfg, label=''):
    r"""
    Dumps the config to the output directory specified in
    :obj:`cfg.out_dir`

    Args:
         cfg (CfgNode): Configuration node
        label:
    """
    
    if  cfg.model.random_state_seed == 'time':
        dynamic_random_state = int(time.time())  # Convert current time to an integer seed
        cfg.model.random_state = dynamic_random_state # 
    else:
        cfg.model.random_state = 42
        
    os.makedirs(cfg.out_dir, exist_ok=True)

   
    if  cfg.model.task == 'classifier_evaluation':
        cfg.out_dir = os.path.join(cfg.out_dir, cfg.model.task , cfg.date, label)
        # makedirs_rm_exist(cfg.out_dir)
        os.makedirs(cfg.out_dir, exist_ok=True)
    else: 
        cfg.out_dir = os.path.join(cfg.out_dir, cfg.model.task, label)
    
    cfg.run_dir = cfg.out_dir

# =============================================================================
def create_params_yaml(cfg, suffix , filename=None):
    if filename is None:
        filename = f'{suffix}_{cfg.feature.DTCombine}_{cfg.time}'
    cfg.run_dir = os.path.join(cfg.out_dir,filename)
    os.makedirs(cfg.run_dir, exist_ok=True)
    cfg_file = os.path.join(cfg.run_dir, cfg.cfg_dest)
    try:
        with open(cfg_file, 'w') as f:
            cfg.dump(stream=f)
    except:
        pass
    return cfg.run_dir
# =============================================================================        
def load_cfg(cfg, args):
    r"""
    Load configurations from file system and command line

    Args:
        cfg (CfgNode): Configuration node
        args (ArgumentParser): Command argument parser

    """
    cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

# =============================================================================
set_cfg(cfg)




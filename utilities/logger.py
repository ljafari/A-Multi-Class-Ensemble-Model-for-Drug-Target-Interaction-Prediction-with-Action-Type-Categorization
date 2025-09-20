import logging
import sys
from DPIgraphgym.config import cfg
from DPIgraphgym.utils.io import  makedirs

# ============================================================================= 
def setup_printing():
    """
    Set up printing options

    """
    #cfg.run_dir =  '/media/leila/PHD/my_graph_neural_network/myGraphGym_master/results/Methapath2vec/2024-1-2/test'
    makedirs(cfg.run_dir)  
    
    # Clear existing handlers from the root logger
    logging.root.handlers = []  
    
    # Configure logging
    logging_cfg = {'level': logging.INFO, 'format':'%(message)s %(asctime)s' }
    
    # logging_cfg = {'level': logging.INFO, 'format':' %(message)s'}
    
      
    h_file = logging.FileHandler('{}/logging.log'.format(cfg.run_dir))      
    h_stdout = logging.StreamHandler(sys.stdout)
    
    if cfg.print == 'file':
        logging_cfg['handlers'] = [h_file]
    elif cfg.print == 'stdout':
        logging_cfg['handlers'] = [h_stdout]
    elif cfg.print == 'both':
        logging_cfg['handlers'] = [h_file, h_stdout]
    else:
        raise ValueError('Print option not supported')
    logging.basicConfig(**logging_cfg)
    
    logging.getLogger('gensim.models.word2vec').setLevel(logging.ERROR)
    # setup final results logger
    logger = logging.getLogger('final_results')
    
    # Set the logging level for the logger
    logger.setLevel(logging.INFO)
    
    
    # Create a FileHandler and attach it to the logger
    file_handler = logging.FileHandler('{}/final_results_logging.log'.format(cfg.run_dir))
    
    formatter = logging.Formatter(' %(message)s\n')
    
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger
# ============================================================================= 
def logger_info(logger, msg, print_sharp = 1):
    
    if  print_sharp == 1:
        logger.info("\n###########################################  " 
                    + msg +      
          "   ###########################################")     
    else: 
        logger.info(msg) 
  
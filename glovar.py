"""Global variables."""
import os

def set_dirs(data_dir=os.getcwd(), dataset='multi_feature'):
    global APP_DIR
    global CKPT_DIR
    global PKL_DIR
    global DATA_DIR
    APP_DIR = data_dir
    CKPT_DIR = os.path.join(data_dir, 'weights/')
    PKL_DIR = os.path.join(data_dir, 'hist/')
    
    data_path = 'dataset/' + dataset + '/'
    DATA_DIR = os.path.join(data_dir, data_path)

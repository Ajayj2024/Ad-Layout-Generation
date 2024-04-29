import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import yaml
from dataclasses import dataclass

from utils.logger import logging
from utils.exception import CustomeException

def parameters(config_path):
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    except CustomeException as ce:
        raise ce
    
@dataclass
class CONFIG:
    debug: int = 0
    config_file_path = 'src/config/config.yaml'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    params = parameters(config_file_path)
    
@dataclass
class MAP_CONFIG:
    config_file_path = 'src/config/mapping.yaml'
    mapping = parameters(config_file_path)
import os, sys
import json
from typing import List, Dict
import errno
import random as rn
import numpy as np
import torch
from utils.exception import CustomeException


DATA2LABEL = {
    
}

def assetion_err(file_path: str):
    assert os.path.exists(file_path), f"{file_path} is not existed"

# Reads the jsonl data line wise and returns each line data in list
def read_jsonl(path: str) -> List[Dict]:
    data = []
    assetion_err(path)
    with open(path,'r') as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
            
    return data

# writes the data line wise and store i .jsonl
def write_jsonl(path: str, data) -> None:
    with open(path, 'w', encoding='utf8') as f:
        for item in data:
            f.write("{}\n".format(json.dumps(item)))
            
#reads json         
def read_json(path: str):
    assetion_err(path)
    with open(path, 'r', encoding='utf8') as f:
        return json.loads(f)
    
# writes the data .json
def write_json(path: str, data, indent: int = 2) -> None:
    with open(path, 'w', encoding='utf8') as f:
        f.write(json.dumps(data, indent = indent))
        
# reads .txt
def read_txt(path: str):
    assetion_err(path)
    with open(path, 'r') as f:
        data = f.read()
        data = data.strip().split('\n')
    return data

# write data to .txt
def write_txt(path: str, data_lst: list):
    with open(path, 'w') as f:
        for data in data_lst:
            f.write(data + '\n')
        
def makedirs(path: str):
    try: 
        '''
        normpath: nornalizes the path ex: a//b///c/ to a/b/c
        expanduser: if the path is not exist the it creates the the folder like a/b/c
        '''
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
        
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise e
        
        
class AverageMeter:
    # Keep tracking average values over the time
    def __init__(self) -> None:
        self.count = 0
        self.average = 0
        self.sum = 0
        
    def reset_values(self):
        self.__init__()
        
    def update(self, val, num_samples= 1):
        self.count = num_samples
        self.sum += num_samples * val
        self.average = self.sum / self.count

def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
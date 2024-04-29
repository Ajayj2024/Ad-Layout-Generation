import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))

import torch
import argparse
from torch.utils.data import Dataset
from utils.exception import CustomeException
from utils.logger import logging
from utils.file_utils import *
from ir.executor import ConstraintExecutor


class LayoutPlacementDataset(Dataset):
    def __init__(self, args, data_split):
        self.args = args
        self.data_split = data_split
        assert data_split in ['train', 'val', 'test', 'prediction'], f"{data_split} dooesn't matches any one of [train, val, test, prediction]"
        self.data_dict = {
            # 'train': {
            #     'source': self.args.train_source_file, 'target': self.args.train_target_file
            # },
            # 'val': {
            #     'source': self.args.source_val_file, 'target': self.args.target_val_file
            # },
            # 'test': {
            #     'source': self.args.test_ground_truth_file, 'target': ''# ?????
            # },
            # 'prediction': {}## ???
        }
        
        self.source_path = os.path.join('dataset', 'stage2/pretrain/train_source.txt')
        self.target_path = os.path.join('dataset', 'stage2/pretrain/train_target.txt')

        self.num_train = 4 # self.config.args.num_train if self.data_split == "train" else None
        if self.source_path.endswith('.json'):
            self.build_json_data()
        elif self.source_path.endswith('.txt'):
            self.build_txt_data()
            
   
    def build_txt_data(self):
        self.source_data = read_txt(self.source_path)
        self.target_data = read_txt(self.target_path)

        assert len(self.source_data) == len(self.target_data), "source data and target data should be of same length"
        if self.data_split == 'train':
            self.source_data, self.target_data = self.source_data[:self.num_train], self.target_data[:self.num_train]
            
        
    def build_json_data(self):
        _data = read_json(self.source_path)
        if self.config.args.is_two_stage:
            if self.split == 'prediction':
                self._process_stage_one_prediction(_data)
            else:
                self.source_data = [item['execution'] for item in _data]
                if self.config.args.add_complete_token:
                    self.target_data = [item['layout_seq_with_completion'] for item in _data]
                else:
                    self.target_data = [item['layout_seq_without_completion'] for item in _data]
        else:
            self.source_data = [item['text'] for item in _data]
            if self.config.args.add_complete_token:
                self.target_data = [item['layout_seq_with_completion'] for item in _data]
            else:
                self.target_data = [item['layout_seq_without_completion'] for item in _data]
        if self.num is not None:
            self.source_data = self.source_data[:self.num]
            self.target_data = self.target_data[:self.num]
    

    def _process_stage_one_prediction(self, predictions):
        executor = ConstraintExecutor('./src/ir/grammar_ad.lark')
        self.source_data = []
        self.target_data = []
        for prediction in predictions:
            lf = prediction['pred_lf'] if self.config.args.test_prediction_ir else prediction['gold_lf']
            try:
                execution = executor(lf)[0].input
                self.source_data.append(execution)
            except:
                self.source_data.append('')
            self.target_data.append('')
    
    def __len__(self):
        return len(self.source_data)
    
    def __getitem__(self, index):
        src_txt = self.source_data[index]
        trg_txt = self.target_data[index]
        print({
            'source_text': src_txt,
            'target_text': trg_txt
        })
        return {
            'source_text': src_txt,
            'target_text': trg_txt
        }
        
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument('--data_root_dir', default= 'dataset')
#     parser.add_argument('--source_train_file', default= 'stage2/pretrain/train_source.txt')
#     parser.add_argument('--target_train_file', default= 'stage2/pretrain/train_target.txt')
#     args_parse = parser.parse_args()
#     dataset = LayoutPlacementDataset(args_parse, data_split= 'train')
#     print(dataset.source_data)
#     print(dataset.target_data)
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import List, Dict
from collections import defaultdict
import torch
import torch.nn.functional as F  
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, AutoTokenizer

from utils import file_utils
from parse_stage.dataset.preprocessor import TextPreprocessor, IRProcessor
from utils.exception import CustomeException
from utils.logger import logging
from config.config import CONFIG

# Gives infomation of each samples containing tokenizing info of both text and IR in dictionary format
class DatasetLoad(Dataset):
    def __init__(self, root_dir: str, 
                 split: str,
                 text_processor: TextPreprocessor, 
                 ir_processor: IRProcessor,
                 tokenizer: PreTrainedTokenizer):
        """Train, val, test dataset Loading. If train, val, test.jsonl are not present then they are created using 
            all.jsonl. A list of of sample info in dictionary is given with tokenizing infomation

        Args:
            root_dir (str): root directory of data which contains train.jsonl, val.jsonl, or in json format
            split (str): value can be of : train, val, test
            text_processor (TextPreprocessor): This preprocesses the prompt (i.e. implicit contraints)
            ir_processor (IRProcessor): This preprocesses the IR
            
        Dictionary contains keys: 'sample_id', 'type', 'text', 'logical_form', 'value_map', 
                                'text_ids', 'text_attention_mask', 'ir_ids', 'ir_attention_mask'
        """
        super().__init__()
        self.root_dir = root_dir
        
        # check file is jsonl
        if os.path.exists(os.path.join(self.root_dir, f"{split}.jsonl")):
            self.split_path = os.path.join(self.root_dir, f"{split}.jsonl")
        
        # checks file if it is json   
        elif os.path.exists(os.path.join(self.root_dir, f"{split}.json")):
            self.split_path = os.path.join(self.root_dir, f"{split}.json")
            
        # if there is no such file named with $split
        else:
            self.create_dataset()
            self.split_path = os.path.join(self.root_dir, f"{split}.jsonl")
        
        self.text_processor = text_processor
        self.ir_processor = ir_processor
        self.tokenizer = tokenizer
        
        if self.split_path.endswith('.jsonl'):
            logging.info(f"Reading {self.split_path}.jsonl")
            temp_data = file_utils.read_jsonl(self.split_path) # type: List[Dict]
            
        else:
            temp_data = file_utils.read_json(self.split_path)   # type: List[Dict]
            
            
        self.data = list()
        try:
            logging.info(f"Preprocessing and tokenizing of {split} started")
            for temp in temp_data:
                _temp = self.preprocess_temp(temp)
                if _temp is None: continue
                
                self.data.append(_temp)
            if CONFIG.debug:
                print(self.data[0])
        except CustomeException as ce:
            raise ce
        
        logging.info("Dataset was loaded and preprocessed and tokenizing")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def preprocess_temp(self, _temp: dict) -> dict:
        """This function preprocess the prompt and IR and also tokenizes

        Args:
            _temp (dict): Contains region, text, ir, id information of one sample

        Returns:
            dict: returns a dictionary conataining text, ir, text_ids, text_attention_mask, ir_ids, ...... 
        """
        if 'ir' not in _temp:
            print("No Intermediate Representation")
            return None     
        
        _temp_id, region_type = _temp['region_id'], _temp['region_type']
        
        # Preprocessing of prompt and IR
        ## preprocessing the prompt
        logging.info("Preprocessing of IR and text started")
        text, value_map = self.text_processor.preprocessor(_temp['text'])
        
        ## preprocessing the IR
        _ir = None
        # check if it is str
        if isinstance(_temp['ir'], str): _ir = _temp['ir']
            
        # if it is not str, if it contains multiple ir then consider the shortest ir
        else:
            _ir = _temp['ir'][0]
            min_length = len(_ir)
            for ir in _temp['ir']:
                if len(ir) < min_length:
                    _ir = ir
                    min_length = len(ir)
                    
        ir = self.ir_processor.preprocess(_ir, value_map)
        
        results = {
            'sample_id': _temp_id, 'type': region_type, 
            'text': text, 'logical_form': ir,
            'value_map': value_map
        }
        
        # Tokenization
        logging.info("Tokenization of IR and text started")
        ## Prompt tokenization
        text_tokenization = self.tokenizer(text, return_tensors='pt')
        text_ids, text_attention_mask = text_tokenization.input_ids[0], text_tokenization.attention_mask[0]
        
        ## IR tokenization
        ir_tokenization = self.tokenizer(ir, return_tensors='pt')
        ir_ids, ir_attention_mask = ir_tokenization.input_ids[0], ir_tokenization.attention_mask[0]
        ir_ids[ir_ids == self.tokenizer.pad_token_id] = -100    # ???????????
        
        results.update({
            'text_ids': text_ids, 'text_attention_mask': text_attention_mask,
            'ir_ids': ir_ids, 'ir_attention_mask': ir_attention_mask,
        })
        
        return results
        
    def create_dataset(self):
        """
        If the there is no train, val, test jsonl files for forming data then this function is called to divide the 
        whole json file to train, val, test. If divides the data in .jsonl format
        """
        base_jsonl = 'all_data.jsonl'
        if not os.path.exists(os.path.join(self.root_dir, base_jsonl)):
            raise FileExistsError(f"No such {base_jsonl} file exists")
        split_file_names = ['train.jsonl', 'val.jsonl', 'test.jsonl']
        samples = file_utils.read_jsonl(os.path.join(self.root_dir, base_jsonl))
        sample_by_type = defaultdict(list)
        
        # Dividing data in ratio 80:10:10
        train_data, val_data, test_data = list(), list(), list()
        # seperating the different type of samples, ex: singleinfo, multiinfo etc..
        for sam in samples:
            sample_by_type[sam['type']].append(sam)
            
        for sam_of_type in sample_by_type.values():
            # divide samples by region_ids
            samples_by_region_id = defaultdict(list)
            for sam in sam_of_type:
                rid = sam['region_id']
                samples_by_region_id[rid].append(sam)
                
            region_ids = list(samples_by_region_id.keys())
            generator = torch.Generator().manual_seed(10)
            shuffled_region_ids = torch.randperm(region_ids, generator= generator)
            
            N = len(shuffled_region_ids)
            ratio = [N * 8, N * 9]
            
            # train data
            for rid in shuffled_region_ids[:ratio[0]]:
                train_data.extend(samples_by_region_id[rid])
            # val data
            for rid in shuffled_region_ids[ratio[0]:ratio[1]]:
                val_data.extend(samples_by_region_id[rid])
            # test data
            for rid in shuffled_region_ids[ratio[1]:]:
                test_data.extend(samples_by_region_id[rid])
        
        # Saving train.jsonl, val.jsonl, test.jsonl     
        for split_data, jsonl_name in zip([train_data, val_data, test_data], split_file_names):
            file_utils.write_jsonl(os.path.join(self.root_dir, jsonl_name), split_data)
            
        logging.info('Train, val, test data was splited and saved')
        
# Stack of all samples
class CollateFn:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id
        
    def __call__(self, samples: List[Dict]):
        batch = defaultdict(list)
        max_input_len, max_label_len = 0, 0
        for sam in samples:
            for key, value in sam.items():
                batch[key].append(value)
                
                if key == 'text_ids':
                    max_input_len = max(max_input_len, len(value))
                    
                elif key == 'ir_ids':
                    max_label_len = max(max_label_len, len(value))
        # Stacking text parameter          
        for id, (input_ids, attention_mask) in enumerate(zip(batch['text_ids'], batch['text_attention_mask'])):
            diff = max_input_len - len(input_ids)
            if diff > 0:
                batch['text_ids'][id] = F.pad(input_ids, (0, diff,), 'constant', self.pad_id)
                batch['text_attention_mask'][id] = F.pad(attention_mask, (0, diff,), 'constant', 0)
                    
        batch['text_ids'] = torch.stack(batch['text_ids'], dim=0)
        batch['text_attention_mask'] = torch.stack(batch['text_attention_mask'], dim=0)
        
             
        # stacking IR parameters
        if 'ir_ids' in batch:
            for idx, (input_ids, attention_mask,) in enumerate(zip(batch['ir_ids'], batch['ir_attention_mask'])):
                diff = max_label_len - len(input_ids)
                if diff > 0:
                    batch['ir_ids'][idx] = F.pad(input_ids, (0, diff,), 'constant', self.pad_id)
                    batch['ir_attention_mask'][idx] = F.pad(attention_mask, (0, diff,), 'constant', 0)
            batch['ir_ids'] = torch.stack(batch['ir_ids'], dim=0)
            batch['ir_attention_mask'] = torch.stack(batch['ir_attention_mask'], dim=0)
        
        logging.info("Collating of data completed")
        return batch


# if __name__ == "__main__":
#     if CONFIG.debug: print("Dataset Loading")
#     tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small', use_fast=False)
#     dataset = DatasetLoad(root_dir= './dataset/stage1',
#                           split= 'train',
#                           text_processor= TextPreprocessor(),
#                           ir_processor= IRProcessor(),
#                           tokenizer= tokenizer
#                           )
#     print("Tokenizer pad_id:", tokenizer.pad_token_id)
#     collate_fn = CollateFn(pad_id= tokenizer.pad_token_id)
#     print(collate_fn(dataset.data))
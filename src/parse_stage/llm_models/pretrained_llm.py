import sys, os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import Callable
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM


from utils.exception import CustomeException
from utils.logger import logging

class PreTrainedLM(nn.Module):
    def __init__(self, model_name: str, training_from_scratch: bool = False) -> None:
        super().__init__()
        
        try:
            logging.info("PLM is loading") 
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda')
            if training_from_scratch:
                config = self.model.config
                config.num_layers = 2
                config.num_decoder_layers = 2
                self.model = AutoModelForSeq2SeqLM.from_config(config)
                
        except CustomeException as ce:
            logging.info('Error in loading PLM')
            raise CustomeException(ce)
            
    
    def forward(self, input_ids, attention_mask, labels=None, pad_token_id=None,
                do_generation=False, generation_max_length: int = 512, prefix_allowed_tokens_fn: Callable = None):
        if do_generation:
            try:
                outputs = self.model.generate(input_ids, attention_mask=attention_mask,
                                            max_length=generation_max_length, pad_token_id=pad_token_id,
                                            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn)
                
            except CustomeException as ce:
                raise ce
        else:
            try:
                outputs = self.model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels)
                
            except CustomeException as ce:
                raise ce
        return outputs
    
# if __name__ == "__main__":
#     plm = PreTrainedLM(model_name= 'google-t5/t5-small')
        
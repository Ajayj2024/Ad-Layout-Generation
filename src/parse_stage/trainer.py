import sys, os, re

import torch.distributed

sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import Callable, List
from collections import Counter
import shutil
import torch
from tqdm import tqdm
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoTokenizer
# import deepspeed as ds
import wandb
import numpy as np
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from config.dictionary import LABELS, ELEMENTS
from config.config import CONFIG
from ir.executor import ConstraintExecutor
from parse_stage.llm_models.pretrained_llm import PreTrainedLM
from parse_stage.dataset.dataset import DatasetLoad, CollateFn
from metrics import AccuracyMetrics,EleAccuracyMetrics, SetAccuracyMetrics, LossMetrics, Metrics
from parse_stage.dataset.preprocessor import IRProcessor, TextPreprocessor
from utils import file_utils
from utils.exception import CustomeException
from utils.logger import logging

config = CONFIG()
is_debug = config.debug
        
class BasicTrainer:
    def __init__(self,
                 args,
                 model,
                 train_dataset: Dataset,
                 val_dataset: Dataset,
                 collatefn: Callable,
                 metric: List[Metrics],
                 ckpt_metric: str = 'loss',
                 optimizer = None):
        self.args = args
        self.plm = model
        self.model = self.plm.model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.collatefn = collatefn
        self.output_dir =self.args.output_dir
        self.optimizer = optimizer
        self.ckpt_metric = ckpt_metric
        self.metric = metric
        
        for met in self.metric:
            if met.METRICS_NAME == self.ckpt_metric:
                self.is_metric_positive = met.is_positive
                
        self.ckpt_path = self.args.output_dir
        file_utils.makedirs(self.ckpt_path)
        self.device = 'cuda'
        
        self._create_dataset()
        
    def _aggreate_metrics(self, result: dict) -> None:
        for metric in self.metric:
            metric.aggregate(result)
            
    def _collect_metrics(self, reset: bool = False) -> dict:
        results = dict()
        for metric in self.metric:
            results.update(metric.compute())
            if reset:
                metric.reset()
        return results 
    
    def _create_dataset(self):
        self.batch_size = self.args.eval_micro_batch_size
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size= self.batch_size,
                                           collate_fn=self.collatefn)
        self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                         batch_size= self.batch_size,
                                         collate_fn=self.collatefn)
        
    def __call__(self, training_step, eval_step):
        train_steps_per_epoch, val_steps_per_epoch = len(self.train_dataloader), len(self.val_dataloader)
        ckpt_metric_best_value = -1e+8 if self.is_metric_positive else np.inf
        epochs = self.args.num_epochs
        for epoch in range(1,epochs + 1):
            self.model.train()
            with torch.enable_grad(), tqdm(total= train_steps_per_epoch) as progress_bar:
                for idx,batch in enumerate(self.train_dataloader):
                    loss = training_step(self.plm, batch, self.device)
                    loss.backward()
                    self.optimizer.step()
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    progress_bar.update(1)
                    progress_bar.set_postfix(epoch=epoch, loss=loss.item())
            
            
            self.model.eval()
            with torch.no_grad(), tqdm(total= val_steps_per_epoch) as progress_bar:
                for data in self.val_dataloader:
                    batch_metrics, _ = eval_step(self.plm, data, self.device, epoch)
                    self._aggreate_metrics(batch_metrics)
                    progress_bar.update(1)
                    progress_bar.set_postfix(epoch=epoch, loss=loss.item())
                    
            metrics = self._collect_metrics(reset=True)
        if is_debug: print(metrics)
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, "finetune_parse_stage_model.bin"))


class TrainerBatchLoss:
    def __init__(self, label_smoothing_factor: float = 0.0) -> None:
        self.label_smoother = None
        if label_smoothing_factor > 0:
            self.label_smoother = LabelSmoother(epsilon=label_smoothing_factor)

    def __call__(self, model, batch, device):
        input_ids, attention_mask = batch["text_ids"].to(device), batch["text_attention_mask"].to(device)
        labels = batch["ir_ids"].to(device)
        outputs = model(input_ids, attention_mask, labels)
        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs.loss

        return loss.mean()
class EvaluateFn:
    def __init__(self, tokenizer: AutoTokenizer, 
                 text_processor: TextPreprocessor, 
                 ir_processor: IRProcessor,
                 do_predict: bool = False,
                 generation_max_length: int = 512):
        
        self.tokenizer = tokenizer
        self.text_processor = text_processor
        self.ir_processor = ir_processor
        self.do_predict, self.generation_max_length = do_predict, generation_max_length
        self.executor = ConstraintExecutor('src/ir/grammar_ad.lark')
        
    def __call__(self, model, batch, device, epoch_num):
        text_ids, text_attention_mask = batch['text_ids'].to(device), batch['text_attention_mask'].to(device)
        label_ids = batch['ir_ids'].to(device)
        predictions = None
        
        outputs = model(input_ids= text_ids, attention_mask= text_attention_mask, labels= label_ids)
        eval_loss = outputs.loss
        metrics = {
            "loss": eval_loss.mean()
        }
        
        if self.do_predict:
            predictions, n_correct, n_ele_correct, n_set_correct = self.predict(model, batch, device)
            metrics.update({
                "num_correct": n_correct,
                "num_element_correct": n_ele_correct,
                "num_set_correct": n_set_correct,
                "num_examples": len(predictions),
            })
        print('Metrics:',metrics)
        if epoch_num % 10 == 0:
            print("Predictions:", predictions)
        return metrics, predictions
    
    def _is_set_accuracy(self, gold_lf, pred_lf):
        _gold_seq = self.executor.get_constraints(gold_lf)
        try:
            _pred_seq = self.executor.get_constraints(pred_lf)
        except:
            return False
        label_set = list(ELEMENTS.values())
        _pattern = f"((?:{'|'.join(label_set)}) [^ ]+ [^ ]+)"
        _gold_constraint = Counter(re.findall(_pattern, _gold_seq))
        _pred_constraint = Counter(re.findall(_pattern, _pred_seq))

        return _gold_constraint.items() == _pred_constraint.items()

    def predict(self, model, batch, device):
        input_ids, attention_mask = batch['text_ids'].to(device), batch['text_attention_mask'].to(device)
        output_sequences = model(input_ids, attention_mask, generation_max_length=self.generation_max_length,
                                 do_generation=True, pad_token_id=self.tokenizer.pad_token_id)
        out_str = self.tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
        
        predictions = list()
        num_correct = 0
        num_element_correct = 0
        num_set_correct = 0
        
        for idx, ostr in enumerate(out_str):
            gold_lf = batch['logical_form'][idx]
            is_correct = (self.ir_processor.postprocess(ostr) == self.ir_processor.postprocess(gold_lf))
            is_element_correct = self.ir_processor.postprocess(ostr, remove_attrs=True) == \
                self.ir_processor.postprocess(gold_lf, remove_attrs=True)
            is_set_correct = self._is_set_accuracy(
                self.ir_processor.postprocess(gold_lf, recover_labels=True),
                self.ir_processor.postprocess(ostr, recover_labels=True)
            )

            predictions.append({
                "pred_ir": self.ir_processor.postprocess(ostr, recover_labels=True),
                "gold_ir": self.ir_processor.postprocess(gold_lf, recover_labels=True),
                "is_set_correct": is_set_correct
            })
            
            meta_info = {key: batch[key][idx] for key in ['text', 'sample_id', 'type']}
            predictions[-1].update(meta_info)

            if is_correct: num_correct += 1
            if is_element_correct: num_element_correct += 1
            if is_set_correct: num_set_correct += 1
       
        return predictions, num_correct, num_element_correct, num_set_correct

# def metrics(gold_ir, pred_ir):
#     ir_processor = IRProcessor()
#     print(ir_processor.postprocess(gold_ir, remove_attrs= True))
#     print(ir_processor.postprocess(pred_ir))
    
# if __name__ == "__main__":
#     gold_ir = "[region: ElectronicDevice [el:text [attr:position'right']] [el:image [attr:position'left'] [attr:position'large']] [el:text [attr:position'right']] [el:price [attr:position'right']] [el:contact]]" 
#     pred_ir = "[region: ElectronicDevice [el:text [attr:position'right']] [el:image [attr:position'left'] [attr:position'large']] [el:text [attr:position'right']] [el:price [attr:position'right']] [el:contact]]" 
#     metrics(gold_ir, pred_ir)
    
# if __name__ == "__main__":
#     print("Step 1")
#     text_processor = TextPreprocessor()
#     ir_processsor = IRProcessor()
#     tokenizer = AutoTokenizer.from_pretrained('google-t5/t5-small', use_fast=False)
#     print("Step 2")
#     collate_fn = CollateFn(pad_id=tokenizer.pad_token_id)
#     print("Step 3")
#     plm = PreTrainedLM(model_name= 'google-t5/t5-small')
#     optimizer = torch.optim.AdamW(plm.model.parameters(),lr= 0.001)
#     train_dataset = DatasetLoad(root_dir= './dataset/stage1',
#                           split= 'train',
#                           text_processor= TextPreprocessor(),
#                           ir_processor= IRProcessor(),
#                           tokenizer= tokenizer
#                           )
#     val_dataset = DatasetLoad(root_dir= './dataset/stage1',
#                           split= 'val',
#                           text_processor= TextPreprocessor(),
#                           ir_processor= IRProcessor(),
#                           tokenizer= tokenizer
#                           )
#     print("Step 4")
#     trainer = BasicTrainer(output_dir= 'temp',
#                            model= plm,
#                            train_dataset= train_dataset,
#                            val_dataset= val_dataset,
#                            collatefn= collate_fn,
#                            metric=[AccuracyMetrics(), LossMetrics(), EleAccuracyMetrics(), SetAccuracyMetrics()], 
#                            ckpt_metric=AccuracyMetrics.METRICS_NAME,
#                            optimizer=optimizer)
#     trainer_fn = TrainerBatchLoss()
#     evaluatefn = EvaluateFn(tokenizer= tokenizer,
#                             text_processor= text_processor,
#                             ir_processor= ir_processsor,
#                             do_predict= True)
#     print("step 5")
#     trainer(trainer_fn, evaluatefn)

# class DSTrainer:
#     TRAINER = 'deepspeed'
#     def __init__(self,
#                  args,
#                  project_name: str,
#                  experiment_config: dict,
#                  model,
#                  train_dataset: Dataset,
#                  val_dataset: Dataset,
#                  collatefn: Callable,
#                  ckpt_metric: List[Metrics],
#                  metric: str = 'loss',
#                  optimizer = None):
#         self.args = args
#         self.model = model
#         self.train_dataset = train_dataset
#         self.val_dataset = val_dataset
#         self.collatefn = collatefn
#         self.output_dir = args.output_dir
        
#         self.ckpt_metric = ckpt_metric
#         self.metric = metric
#         self.is_metric_positive = True
#         for met in self.ckpt_metric:
#             if met.METRIC_NAME == self.metric:
#                 self.is_metric_positive = met.is_positive
                
#         self.ckpt_path = os.path.join(self.output_dir, 'checkpoints')
#         self.eval_interval = self.args.eval_delay   # ???
        
#         logging.info("Setup experiment")
        
#     @property
#     def _is_main_process(self):
#         return self._local_rank in [-1, 0]

#     def _setup_model(self, optimizer = None):
#         if optimizer is not None:
#             logger.info("Overwrite optimizer specified in config.json")
#             self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=self.args, model=self.model,
#                                                                            optimizer=optimizer)
#         else:
#             parameters = self.model.parameters()
#             if self.args.use_adafactor:
#                 optimizer = Adafactor([{'params': parameters}], scale_parameter=False, relative_step=False,
#                                     warmup_init=False, lr=self.args.adafactor_lr, weight_decay=1e-5)
#             else:
#                 optimizer = None
#             self.model_engine, self.optimizer, _, _ = deepspeed.initialize(args=self.args, model=self.model,
#                                                                            model_parameters=parameters,
#                                                                            optimizer=optimizer)
#         self.device = self.model_engine.local_rank
#     def _setup_experiment(self, project_name: str, experiment_config):
#         deepspeed.init_distributed(dist_backend=self.args.backend)
#         self._local_rank = int(os.environ['LOCAL_RANK'])
#         self._world_size = int(os.environ['WORLD_SIZE'])
#         logging.info(f'World size: {self._world_size}, Local rank: {self._local_rank}, Main Process: {self._is_main_process}')
#         self._ds_config = file_utils.read_json(self.args.deepscale_config)

#         # setup
#         if self._is_main_process:
#             if not os.path.exists(self.output_dir):
#                 file_utils.makedirs(self.output_dir)
#             hypara_config = self._log_hyperparameters()
#             hypara_config.update({
#                 'trainer': self.TRAINER_NAME
#             })
#             hypara_config.update(experiment_config)
            # wandb.init(project=project_name, config=hypara_config)
#             shutil.copy(self.args.deepscale_config, os.path.join(self.output_dir, 'ds_config.json'))
#         torch.distributed.barrier()
        
#     def _log_hyperparameters(self):
#         micro_batch_size = self._ds_config['train_micro_batch_size_per_gpu']
#         gradient_accumulation = self._ds_config.get('gradient_accumulation_steps', 1)
#         real_batch_size = micro_batch_size * gradient_accumulation * self._world_size
#         config = {
#             'seed': self.args.seed,
#             'epoch': self.args.num_epochs,
#             'gradient_accumulation_steps': gradient_accumulation,
#             'micro_batch_size': micro_batch_size,
#             'batch_size': real_batch_size,
#             'gradient_clip': self._ds_config.get('gradient_clipping', 0.0),
#             'label_smoothing': self.args.label_smoothing_factor
#         }
#         if self.args.use_adafactor:
#             config['optimizer'] = 'AdaFactor'
#             config['learning_rate'] = self.args.adafactor_lr
#         else:
#             config.update({
#                 'optimizer': self._ds_config['optimizer']['type'],
#                 'learning_rate': self._ds_config['optimizer']['params']['lr'],
#                 'scheduler': self._ds_config['scheduler']['type'],
#                 'warmup_steps': self._ds_config['scheduler']['params']['warmup_num_steps']
#             })
#         return config
    
#     def _setup_dataset(self):
#         micro_batch_size = self._ds_config['train_micro_batch_size_per_gpu']
#         eval_micro_batch_size = self.args.eval_micro_batch_size
#         self.train_sampler = DistributedSampler(dataset=self.train_dataset, num_replicas=self._world_size)
#         self.train_dataloader = DataLoader(dataset=self.train_dataset,
#                                            batch_size=micro_batch_size,
#                                            collate_fn=self.collate_fn,
#                                            sampler=self.train_sampler)
#         self.val_dataloader = DataLoader(dataset=self.val_dataset,
#                                          batch_size=eval_micro_batch_size,
#                                          collate_fn=self.collate_fn)
#     def __call__(self, train_step, eval_step):
#         ckpt_metric_best_value = -1e+8 if self.is_ckpt_metric_positive else np.inf
#         global_step = 0
#         for epoch in range(self.args.num_epochs):
#             self.train_sampler.set_epoch(epoch)
#             self.model_engine.train()
#             for batch_idx, batch in enumerate(self.train_dataloader):
#                 loss = train_step(self.model_engine, batch, self.device)
#                 self.model_engine.backward(loss)
#                 self.model_engine.step()

#                 if global_step % self.args.train_log_step == 0 and self._is_main_process:
#                     wandb.log({'training_loss': loss}, step=global_step)
#                     logger.info('\t'.join([
#                         f'[{epoch}/{self.args.num_epochs}][{batch_idx}/{len(self.train_dataloader)}]',
#                         f'Loss: {loss.item():.3f}'
#                     ]))
#                 global_step += 1

#             # Special checkpoint in Deepspeed
#             if epoch == (self.args.num_epochs - 1) or (epoch + 1) % self.args.save_interval == 0:
#                 self.model_engine.save_checkpoint(save_dir=self._normal_ckpt_path,
#                                               client_state={'epoch': epoch + 1})

#             if self._is_main_process and (epoch == (self.args.num_epochs - 1) or (epoch + 1) % self._eval_interval == 0):
#                 self.model_engine.eval()
#                 with torch.no_grad():
#                     for data in self.val_dataloader:
#                         batch_metrics, _ = eval_step(self.model_engine, data, self.device)
#                         self._aggreate_metrics(batch_metrics)

#                 metrics = self._collect_metrics(reset=True)
#                 wandb.log(metrics, step=global_step)
#                 logger.info('\t'.join([f"{k}: {v:.3f}" for k, v in metrics.items()]))

#                 is_best = False
#                 ckpt_metric_value = metrics[self.ckpt_metric]
#                 if self.is_ckpt_metric_positive and ckpt_metric_value > ckpt_metric_best_value:
#                     is_best = True
#                 elif not self.is_ckpt_metric_positive and ckpt_metric_value < ckpt_metric_best_value:
#                     is_best = True
#                 self.do_checkpointing(epoch, is_best)

#             torch.distributed.barrier()

#     def do_checkpointing(self, epoch, is_best):
#         if is_best:
#             with open(os.path.join(self.output_dir, 'best_epoch'), 'w') as f:
#                 f.write(f"{epoch}")
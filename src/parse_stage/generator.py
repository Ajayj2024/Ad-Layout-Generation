import sys, os

import torch.distributed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import Callable, Dict, List
# import deepspeed
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from parse_stage.dataset.dataset import DatasetLoad,  CollateFn
from parse_stage.dataset.preprocessor import TextPreprocessor, IRProcessor
from parse_stage.llm_models.pretrained_llm import PreTrainedLM
from parse_stage.metrics import Metrics, AccuracyMetrics, EleAccuracyMetrics, SetAccuracyMetrics, LossMetrics
from utils import file_utils
from utils.exception import CustomeException
from utils.logger import logging
from parse_stage.trainer import EvaluateFn
# from parse_stage.run_parser import TrainArguments



class Generator:

    MODEL_BIN_NAME = "finetune_parse_stage_model.bin"

    def __init__(self,args, plm, dataset: Dataset, collate_fn: Callable,
                 metrics: List[Metrics], save_prefix: str = 'test') -> None:
        self.args = args
        self.plm = plm
        # print(self.model)
        self.device = 'cuda'
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.metrics = metrics
        self.ckpt_path = args.output_dir
        self.batch_size = args.eval_micro_batch_size
        self.prediction_dir = args.prediction_dir
        # self.ds_ckpt_tag = self.args.ds_ckpt_tag
        file_utils.makedirs(self.prediction_dir)
        self.metrics_save_path = os.path.join(self.prediction_dir, f'{save_prefix}_metrics.json')
        self.prediction_save_path = os.path.join(self.prediction_dir, f'{save_prefix}_predictions.json')
        self._setup()

    # @property
    # def _is_main_process(self):
    #     return self._local_rank in {-1, 0}

    
    def _set_basic_model(self):
        pass
    def _setup(self):
        self.use_deepspeed = False
        # self._local_rank = int(os.environ['LOCAL_RANK'])
        # self.device = torch.device("cuda:{}".format(self._local_rank))
        # eval_micro_batch_size = self.args.eval_micro_batch_size
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=self.batch_size,
                                     collate_fn=self.collate_fn)

        if os.path.exists(os.path.join(self.ckpt_path, self.MODEL_BIN_NAME)):
            self.use_deepspeed = False
            state = torch.load(os.path.join(self.ckpt_path, self.MODEL_BIN_NAME))
            # print(state)
            self.plm.model.load_state_dict(state, strict=False)
            self.plm.model.to(self.device)
        # else:
        #     logging.info("Inference using deepspeed")
        #     deepspeed.init_distributed(dist_backend=self.args.backend)
        #     logging.info(f"Local Rank: {self._local_rank}, Main Process: {self._is_main_process}")

        #     # load with deepspeed
        #     params = filter(lambda p: p.requires_grad, self.model.parameters())
        #     self.model, _, _, _ = deepspeed.initialize(args=self.args, model=self.model,
        #                                                model_parameters=params)
        #     if self.ckpt_path is not None:
        #         _, client_state = self.model.load_checkpoint(self.ckpt_path, tag=self.ds_ckpt_tag,
        #                                                     load_module_only=True,
        #                                                     load_optimizer_states=False,
        #                                                     load_lr_scheduler_states=False) # model engine
        #         logging.info(client_state)
        #     torch.distributed.barrier()

    def _aggreate_metrics(self, result: Dict) -> None:
        for metric in self.metrics:
            metric.aggregate(result)

    def _collect_metrics(self, reset: bool = False) -> Dict:
        results = dict()
        for metric in self.metrics:
            results.update(metric.compute())
            if reset:
                metric.reset()
        return results

    def __call__(self, eval_step: Callable):
        # if self._is_main_process:
        self.plm.model.eval()
        predictions = list()
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.dataloader):
                batch_metrics, batch_predictions = eval_step(self.plm, batch, self.device)
                self._aggreate_metrics(batch_metrics)
                predictions.extend(batch_predictions)
            metrics = self._collect_metrics(reset=True)
            for key, value in metrics.items():
                logging.info(f"{key}: {value:.3f}")

            file_utils.write_json(self.metrics_save_path, metrics)
            file_utils.write_json(self.prediction_save_path, predictions)

        if self.use_deepspeed:
            torch.distributed.barrier()


# if __name__ == "__main__":
    # text_processor = TextPreprocessor()
    # ir_processor = IRProcessor()
    # tokenizer = AutoTokenizer.from_pretrained('google/t5-v1_1-small', use_fast=False)
    # dataset = DatasetLoad(root_dir= 'dataset/stage1',split= 'val', text_processor= text_processor, ir_processor= ir_processor,
    #                       tokenizer= tokenizer)
    # plm = PreTrainedLM(model_name= 'google/t5-v1_1-small')
    
    # collatefn = CollateFn(pad_id=tokenizer.pad_token_id)
    # print("Step")
    # generator = Generator(plm = plm, dataset= dataset, collate_fn= collatefn, 
    #                       metrics=[AccuracyMetrics(), LossMetrics(), EleAccuracyMetrics(), SetAccuracyMetrics()], save_prefix= 'val')
    # eval_fn = EvaluateFn(tokenizer, text_processor= text_processor,ir_processor= ir_processor, do_predict=True)
    # generator(eval_fn)
    # state = torch.load('checkpoints/parse_stage/finetune_parse_stage_model.bin')
    # print(state)
    # plm.model.load_state_dict(state, strict=False)
    
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'src'))

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_pt_utils import LabelSmoother
import torch

from parse_stage.dataset.preprocessor import IRProcessor, TextPreprocessor
from parse_stage.dataset.dataset import DatasetLoad, CollateFn
from parse_stage.llm_models.pretrained_llm import PreTrainedLM
from parse_stage.trainer import BasicTrainer, TrainerBatchLoss, EvaluateFn
from parse_stage.metrics import Metrics, AccuracyMetrics, EleAccuracyMetrics, SetAccuracyMetrics, LossMetrics
from parse_stage.generator import Generator
from utils.logger import logging
from utils.exception import CustomeException
# from utils import logger
from config.config import CONFIG

config = CONFIG()
is_debug = config.debug

@dataclass
class DataArguments:
    """
    Arguments for dataset
    """
    data_arguments = config.params['DATASET']
    data_root_dir: str = field(default = data_arguments['data_root_dir'],
                               metadata={"help": "data root dir"})
    
    ir_remove_value: bool = field(default=False, 
                                  metadata={"help": "Whether to run remove value attributes in ir."})
    replace_explicit_value: bool = field(
        default=False,
        metadata={"help": "Whether to replace explicitly value with placeholder."}
    )
    eval_split: Optional[str] = field(
        default='test',
        metadata={
            "help": "split to evaluate"
        }
    )
@dataclass
class ModelArguments:
    """
    Arguments for model
    """
    model_arguments = config.params['MODEL_ARGUMENTS']
    model_name: str = field(
        default= model_arguments['base_model'],
        metadata= {'help': "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    
    generation_max_length: Optional[int] = field(
        default= model_arguments['generation_max_length'],
        metadata={
            "help": "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        },
    )
    tuning_method: str = field(
        default= model_arguments['tuning_method'],
        metadata={
            "help": "Tune model using finetuning or prompt tuning",
            "choices": ['finetune', 'prompt_tuning', 'adapter']
        }
    )
    num_prompt_tokens: int = field(
        default= model_arguments['num_prompt_tokens'],
        metadata={"help": "number of prompt tokens"}
    )
    prompt_init_method: str = field(
        default= model_arguments['prompt_init_method'],
        metadata={"help": "ways to initialize embedding of prompt tokens"}
    )

    ADAPTER = 'adapter'
    PROMPT_TUNING = 'prompt_tuning'
    
@dataclass
class TrainArguments:
    """
    Arguments for training
    """
    train_arguments = config.params['TRAIN_ARGUMENTS']
    output_dir: str = field(default= train_arguments['output_dir'],
        metadata={"help": "The output directory where the model checkpoints will be written."},
    )
    # deepscale_config: str = field(
    #     metadata={
    #         "help": "Enable deepspeed and pass the path to deepspeed json config file (e.g. ds_config.json) or an already loaded json file as a dict"
    #     },
    # )
    prediction_dir: str = field(
        default= train_arguments['prediction_dir'],
        metadata={"help": "The output directory where the predictions will be written."},
    )

    seed: int = field(default=train_arguments['seed'], metadata={"help": "Random seed that will be set at the beginning of training."})
    local_rank: int = field(default= train_arguments['local_rank'], metadata={"help": "For distributed training: local_rank"})

    num_epochs: int = field(default= train_arguments['epochs'], metadata={"help": "Total number of training epochs to perform."})
    eval_micro_batch_size: int = field(
        default= train_arguments['eval_micro_batch_size'], metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    eval_delay: Optional[float] = field(
        default=0,
        metadata={
            "help": "Number of epochs or steps to wait for before the first evaluation can be performed, depending on the evaluation_strategy."
        },
    )
    save_interval: Optional[int] = field(
        default=1,
        metadata={
            "help": "interval to save checkpoint."
        }
    )
    train_log_step: Optional[int] = field(
        default=50,
        metadata={
            "help": "logging steps"
        }
    )

    backend: Optional[str] = field(
        default="nccl",
        metadata={"help": 'distributed backend'}
    )


    do_train: bool = field(default= False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default= False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    training_from_scratch: bool = field(default=False, metadata={"help": "Whether to train from scratch."})

    # log_level: Optional[str] = field(
    #     default="info",
    #     metadata={
    #         "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'.",
    #         "choices": logger.log_levels.keys(),
    #     },
    # )

    ds_ckpt_tag: Optional[str] = field(
        default=None,
        metadata={"help": "deepspeed checkpoint tag (load for evaluation)"}
    )

    use_adafactor: bool = field(
        default=False, metadata={"help": "Whether to use adafactor"}
    )
    learning_rate: float = field(
        default= train_arguments['learning_rate'],
        metadata={
            "help": "learning rate for adafactor"
        }
    )
    weight_decay: Optional[float] = field(
        default= train_arguments['weight_decay'],
        metadata={
            "help": "weight decay"
        }
    )
    adam_beta1: Optional[float] = field(
        default= train_arguments['adam_beta1'],
        metadata={
            "help": "adam_beta1"
        }
    )
    adam_beta2: Optional[float] = field(
        default= train_arguments['adam_beta2'],
        metadata={
            "help": "adam_beta2"
        }
    )
    adam_epsilon: Optional[float] = field(
        default= train_arguments['adam_epsilon'],
        metadata={
            "help": "adam_epsilon"
        }
    )
    label_smoothing_factor: float = field(
        default= train_arguments['label_smoothing_factor'], metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )


def get_args():
    hr_args = HfArgumentParser((DataArguments, ModelArguments, TrainArguments))
    data_args, model_args, train_args = hr_args.parse_args_into_dataclasses()
    return data_args, model_args, train_args


        
def main():
    data_args, model_args, training_args = get_args()

    # config_logger(training_args)
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name, use_fast=False)
    print("Loading Tokenizer")
    logging.info("Tokenizer Loaded")
    # if model_args.tuning_method == ModelArguments.PROMPT_TUNING:
    #     logger.info("Use prompt tuning")
    #     model = PromptTuningModel(model_args.model_name, num_prompt_tokens=model_args.num_prompt_tokens,
    #                               initialization_option=model_args.prompt_init_method)
    # elif model_args.tuning_method == ModelArguments.ADAPTER:
    #     logger.info("Use adapter tuning")
    #     model = AdapterPretrainedLM(model_args.model_name, adapter_args=model_args)
    if model_args.tuning_method == 'finetune':
        logging.info("Model loading started")
        plm = PreTrainedLM(model_args.model_name, training_args.training_from_scratch)
        logging.info("Model was loaded")
        if is_debug == 1: print("Model Loaded") 
        
    logging.info("Dataset Loading started")
    logging.info(f"Replace explicit values: {data_args.replace_explicit_value}")
    text_processor = TextPreprocessor(replace_value=data_args.replace_explicit_value)
    ir_processor = IRProcessor(remove_value=data_args.ir_remove_value,
                               replace_value=data_args.replace_explicit_value)
    train_dataset, eval_dataset = None, None
    if training_args.do_train:
        train_dataset = DatasetLoad(split='train', tokenizer= tokenizer, 
                                    ir_processor=ir_processor, text_processor=text_processor)
        eval_dataset = DatasetLoad(split='val', tokenizer= tokenizer, 
                                   ir_processor=ir_processor, text_processor=text_processor)
        logging.info("Datset Loaded for training")
        print("Train and val Data Loded")
    else:
        eval_dataset = DatasetLoad(split= data_args.eval_split, tokenizer=tokenizer, 
                                   ir_processor= ir_processor,
                                   text_processor= text_processor)
        print("Eval data loaded")
    collate_fn = CollateFn(pad_id=tokenizer.pad_token_id)

    eval_fn = EvaluateFn(tokenizer, text_processor, ir_processor, do_predict=True,
                         generation_max_length=model_args.generation_max_length)

    if training_args.do_train:
        logging.info("Training Started...")
        experiment_config = {
            'model_name': model_args.model_name,
            'generation_max_length': model_args.generation_max_length
        }
        # if model_args.tuning_method == ModelArguments.ADAPTER:
        #     optimizer = create_adapter_optimizer(model, training_args)
        # else:
        optimizer = torch.optim.AdamW(plm.model.parameters(),lr= 0.001)
        # if trainer == 'deepseed':
        #     trainer = DSTrainer(
        #         project_name='nl2web-sp', args=training_args, model=model,
        #         train_dataset=train_dataset, val_dataset=eval_dataset,
        #         collate_fn=collate_fn, experiment_config=experiment_config,
        #         metrics=[AccMetric(), LossMetric(), ElementAccMetric(), SetAccMetric()], ckpt_metric=AccMetric.METRIC_NAME,
        #         optimizer=optimizer
        #     )
        # else:
        print("Training Started...........")
        trainer = BasicTrainer(args= training_args,
                               model= plm, 
                               train_dataset= train_dataset, 
                               val_dataset= eval_dataset,
                               collatefn= collate_fn,
                               optimizer= optimizer,
                               metric= [AccuracyMetrics(), LossMetrics(), EleAccuracyMetrics(), SetAccuracyMetrics()])
        train_fn = TrainerBatchLoss(training_args.label_smoothing_factor)
        trainer(train_fn, eval_fn)
        print("Training Ended...........")
    elif training_args.do_eval:
        logging.info("Inference...")
        generator = Generator(args=training_args, plm=plm, dataset=eval_dataset,
                              collate_fn=collate_fn, metrics=[AccuracyMetrics(), LossMetrics(), 
                                                              EleAccuracyMetrics(), SetAccuracyMetrics()],
                              save_prefix=data_args.eval_split)
        generator(eval_fn)


if __name__ == "__main__":
    main()


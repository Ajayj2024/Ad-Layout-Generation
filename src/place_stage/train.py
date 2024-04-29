import os, sys

import torch.distributed
sys.path.append(os.path.join(os.getcwd(), 'src'))

from utils.logger import logging
from utils.exception import CustomeException
from place_stage.place_stage_config.config import Config
from layout_dataset import LayoutPlacementDataset
from trainer.trainer import Trainer
from utils.file_utils import *

import torch
# import wandb
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


if __name__ == "__main__":
    config = Config()
    
    # if config.args.use_wandb and (config.args.local_rank == -1 or config.args.local_rank == 0):
    #     wandb.init(project='placement-model', name=config.args.run_description)
        
    logging.info(f"start training placement model!!!! local_rank={config.args.local_rank}")
    set_seed(config.args.seed)
    
    # if config.args.distribute and config.args.local_rank != -1:
    #     assert torch.cuda.device_count() > config.args.local_rank
    #     torch.cuda.set_device(config.args.local_rank)
    #     device = torch.device("cuda", index=config.args.local_rank)
    #     world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    #     torch.distributed.init_process_group(
    #         backend="nccl",
    #         init_method="env://",
    #         world_size=world_size,
    #         rank=config.args.local_rank
    #     )
    # else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size = 1

    train_set = LayoutPlacementDataset(config, data_split="train")
    valid_set = LayoutPlacementDataset(config, data_split="val")

    # prepare data sampler
    if config.args.distribute and config.args.local_rank != -1:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=config.args.local_rank)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_set,
        batch_size=config.args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.args.num_workers,
        pin_memory=True,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=config.args.batch_size,
        shuffle=False,
        num_workers=config.args.num_workers
    )

    if config.args.local_rank == 0 or config.args.local_rank == -1:
        logging.info(f"datasets loaded, train: {len(train_set)}, valid: {len(valid_set)}")

    trainer = Trainer(config, device, world_size)
    trainer.train(train_loader, valid_loader)
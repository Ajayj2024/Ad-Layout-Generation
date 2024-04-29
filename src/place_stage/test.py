
from config.config import Config
from utils.logger import logging
from layout_dataset import LayoutPlacementDataset

import torch
from torch.utils.data import DataLoader
from trainer.trainer import Trainer
if __name__ == "__main__":
    config = Config()

    logging.info("start testing placement model!!!!")
    set_seed(config.args.seed)

    test_set = LayoutPlacementDataset(config, split="prediction")
    test_loader = DataLoader(
        test_set,
        batch_size=config.args.batch_size,
        shuffle=False,
        num_workers=config.args.num_workers
    )
    logging.info(f"datasets loaded, test: {len(test_set)}")

    trainer = Trainer(config, device='cuda')
    trainer.test(test_loader)
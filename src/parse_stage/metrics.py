import os, sys
sys.path.append(os.path.join(os.getcwd(), 'src'))

from typing import List, Dict
from abc import ABC, abstractmethod
from utils.logger import logging


class Metrics(ABC):
    def __init__(self):
        self._is_positive = True
        
    @abstractmethod
    def aggregate(self, metrics: dict):
        pass
    
    @abstractmethod
    def compute(self):
        pass
    
    @abstractmethod
    def reset(self):
        pass
    
    @property
    def is_positive(self):
        return self._is_positive
    
# Loss Metrics
class LossMetrics(Metrics):
    METRICS_NAME = 'loss'
    
    def __init__(self):
        super().__init__()
        self.n_batch = 0
        self.total_loss = 0.0
        self._is_positive = False
        
    def aggregate(self, metrics: dict):
        self.n_batch += 1
        self.total_loss += metrics['loss'].item()
        
    def compute(self) -> dict:
        if self.n_batch == 0:
            return {self.METRICS_NAME: 0.0}
        
        return {
            self.METRICS_NAME: self.total_loss / self.n_batch
        }
        
    def reset(self) -> None:
        self.n_batch = 0
        self.total_loss = 0.0
        
# Accuracy Metrics
class AccuracyMetrics(Metrics):
    METRICS_NAME = 'accuracy'
    
    def __init__(self):
        super().__init__()
        self.n_correct = 0
        self.n_sample = 0
        
    def aggregate(self, metrics: Dict):
        self.n_correct = metrics.get('n_correct', 0)
        self.n_sample = metrics.get('n_sample', 0)
        
    def compute(self) -> dict:
        if self.n_sample == 0:
            return {self.METRICS_NAME: 0.0}
        return {
            self.METRICS_NAME: self.n_correct / self.n_sample
        }
        
    def reset(self):
        self.n_correct = 0
        self.n_sample = 0
        
# Element Accuracy Metrics: Number of elements are correct
class EleAccuracyMetrics(Metrics):
    METRICS_NAME = 'element_accuracy'
    
    def __init__(self):
        super().__init__()
        self.n_correct = 0
        self.n_sample = 0
        
    def aggregate(self, metrics: Dict):
        self.n_correct = metrics.get('n_ele_correct', 0)
        self.n_sample = metrics.get('n_sample', 0)
        
    def compute(self) -> dict:
        if self.n_sample == 0:
            return {self.METRICS_NAME: 0.0}
        return {
            self.METRICS_NAME: self.n_correct / self.n_sample
        }
        
    def reset(self):
        self.n_correct = 0
        self.n_sample = 0
        
# Set Accuracy Metrics
class SetAccuracyMetrics(Metrics):
    METRICS_NAME = 'set_accuracy'
    
    def __init__(self):
        super().__init__()
        self.n_correct = 0
        self.n_sample = 0
        
    def aggregate(self, metrics: Dict):
        self.n_correct = metrics.get('n_set_correct', 0)
        self.n_sample = metrics.get('n_sample', 0)
        
    def compute(self) -> dict:
        if self.n_sample == 0:
            return {self.METRICS_NAME: 0.0}
        return {
            self.METRICS_NAME: self.n_correct / self.n_sample
        }
        
    def reset(self):
        self.n_correct = 0
        self.n_sample = 0
        
# if __name__ == "__main__":
#     set_acc = SetAccuracyMetrics()
#     metrics = {
#         'n_set_correct': 56,
#         'n_sample': 78
#     }
#     print(set_acc.METRICS_NAME)
#     set_acc.aggregate(metrics)
#     print(set_acc.compute())
#     set_acc.reset()
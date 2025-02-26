"""
Metrics Utility Functions
Current Date and Time (UTC): 2025-02-26 13:32:20
Current User: zlbbbb
"""

import numpy as np
import torch
from typing import Dict, List, Union, Optional, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
import logging

logger = logging.getLogger(__name__)

def convert_to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """将数据转换为numpy数组"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class MetricsCalculator:
    """度量指标计算器"""
    
    @staticmethod
    def mse(y_true: Union[np.ndarray, torch.Tensor],
            y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """均方误差"""
        y_true = convert_to_numpy(y_true)
        y_pred = convert_to_numpy(y_pred)
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def rmse(y_true: Union[np.ndarray, torch.Tensor],
             y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """均方根误差"""
        return np.sqrt(MetricsCalculator.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: Union[np.ndarray, torch.Tensor],
            y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """平均绝对误差"""
        y_true = convert_to_numpy(y_true)
        y_pred = convert_to_numpy(y_pred)
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: Union[np.ndarray, torch.Tensor],
             y_pred: Union[np.ndarray, torch.Tensor],
             epsilon: float = 1e-10) -> float:
        """平均绝对百分比误差"""
        y_true = convert_to_numpy(y_true)
        y_pred = convert_to_numpy(y_pred)
        if np.any(y_true == 0):
            logger.warning("MAPE: 真实值中存在零值，将使用epsilon避免除零错误")
            return mean_absolute_percentage_error(y_true + epsilon, y_pred + epsilon) * 100
        return mean_absolute_percentage_error(y_true, y_pred) * 100
    
    @staticmethod
    def r2(y_true: Union[np.ndarray, torch.Tensor],
           y_pred: Union[np.ndarray, torch.Tensor]) -> float:
        """决定系数"""
        y_true = convert_to_numpy(y_true)
        y_pred = convert_to_numpy(y_pred)
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def smape(y_true: Union[np.ndarray, torch.Tensor],
              y_pred: Union[np.ndarray, torch.Tensor],
              epsilon: float = 1e-10) -> float:
        """对称平均绝对百分比误差"""
        y_true = convert_to_numpy(y_true)
        y_pred = convert_to_numpy(y_pred)
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
        return np.mean(np.abs(y_pred - y_true) / denominator) * 100

class ExperimentMetrics:
    """实验指标管理器"""
    
    def __init__(self):
        """初始化实验指标管理器"""
        self.metrics = {}
        self.current_epoch = 0
        self.best_metrics = {}
        self.calculator = MetricsCalculator()
    
    def update(self, metrics_dict: Dict[str, float], epoch: Optional[int] = None):
        """更新指标"""
        if epoch is not None:
            self.current_epoch = epoch
        
        for name, value in metrics_dict.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((self.current_epoch, value))
            
            # 更新最佳指标（对于验证指标）
            if name.startswith('val_'):
                if name not in self.best_metrics or value > self.best_metrics[name][1]:
                    self.best_metrics[name] = (self.current_epoch, value)
    
    def get_current(self, name: str) -> Optional[float]:
        """获取当前指标值"""
        if name in self.metrics and self.metrics[name]:
            return self.metrics[name][-1][1]
        return None
    
    def get_best(self, name: str) -> Optional[Tuple[int, float]]:
        """获取最佳指标值"""
        return self.best_metrics.get(name)
    
    def compute_all(self,
                   y_true: Union[np.ndarray, torch.Tensor],
                   y_pred: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'mse': self.calculator.mse(y_true, y_pred),
            'rmse': self.calculator.rmse(y_true, y_pred),
            'mae': self.calculator.mae(y_true, y_pred),
            'mape': self.calculator.mape(y_true, y_pred),
            'smape': self.calculator.smape(y_true, y_pred),
            'r2': self.calculator.r2(y_true, y_pred)
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取指标总结"""
        summary = {
            'current': {},
            'best': {}
        }
        
        for name in self.metrics.keys():
            current = self.get_current(name)
            best = self.get_best(name)
            
            if current is not None:
                summary['current'][name] = current
            if best is not None:
                summary['best'][name] = best[1]
        
        return summary

class MovingAverageMetric:
    """移动平均指标"""
    
    def __init__(self, window_size: int = 100):
        """
        初始化移动平均指标
        
        Args:
            window_size: 窗口大小
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0
    
    def update(self, value: float):
        """更新指标值"""
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
    
    def get(self) -> float:
        """获取当前移动平均值"""
        
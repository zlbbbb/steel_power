"""
Visualization Utility Functions
Current Date and Time (UTC): 2025-02-27 03:21:49
Current User: zlbbbb
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

class TrainingVisualizer:
    def __init__(self, log_dir: str, config: Dict[str, Any]):
        """
        初始化训练可视化器
        
        Args:
            log_dir (str): 日志目录
            config (Dict[str, Any]): 配置字典
        """
        self.log_dir = Path(log_dir)
        self.config = config
        self.vis_config = config['evaluation']['visualization']
        
        if not self.vis_config['enabled']:
            return
            
        # 创建日志目录
        self.plots_dir = self.log_dir / 'plots'
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use(self.vis_config.get('style', 'seaborn'))
        
        # 初始化 TensorBoard writer（如果启用）
        if config['training']['logging'].get('tensorboard', True):
            self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        else:
            self.writer = None
        
        # 初始化指标记录器（根据配置的指标）
        self.metrics_history = {
            metric: [] for metric in config['training']['logging']['metrics']
        }
        
    def update_metrics(self, episode: int, metrics: Dict[str, float]):
        """
        更新训练指标并记录到 TensorBoard
        
        Args:
            episode (int): 当前回合数
            metrics (Dict[str, float]): 指标字典
        """
        if not self.vis_config['enabled']:
            return
            
        # 记录到 TensorBoard
        if self.writer:
            for name, value in metrics.items():
                if name in self.metrics_history:
                    self.writer.add_scalar(f'training/{name}', value, episode)
            
        # 更新历史记录
        for name, value in metrics.items():
            if name in self.metrics_history:
                self.metrics_history[name].append(value)
                
    def plot_training_curves(self, save: bool = True):
        """绘制训练曲线"""
        if not self.vis_config['enabled'] or not self.vis_config['plots']['training_curves']['enabled']:
            return
            
        metrics_to_plot = self.vis_config['plots']['training_curves']['metrics']
        n_metrics = len(metrics_to_plot)
        
        if n_metrics == 0:
            return
            
        fig, axes = plt.subplots(n_metrics, 1, 
                                figsize=(12, 4*n_metrics),
                                dpi=self.vis_config['dpi'])
        
        if n_metrics == 1:
            axes = [axes]
            
        window = self.vis_config['plots']['training_curves']['rolling_window']
        
        for ax, metric_name in zip(axes, metrics_to_plot):
            if metric_name not in self.metrics_history:
                continue
                
            data = self.metrics_history[metric_name]
            self._plot_metric(ax, data, metric_name.replace('_', ' ').title(),
                            'Episode', metric_name.replace('_', ' ').title(),
                            window=window)
            
        plt.tight_layout()
        if save and self.vis_config['save_plots']:
            save_path = self.plots_dir / self.vis_config['plots']['training_curves']['filename']
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
            
        if self.vis_config['show_plots']:
            plt.show()
        plt.close()
        
    def plot_power_prediction(self, save: bool = True):
        """绘制电力预测对比图"""
        if (not self.vis_config['enabled'] or 
            not self.vis_config['plots']['predictions']['enabled'] or
            'predicted_power' not in self.metrics_history or
            'actual_power' not in self.metrics_history):
            return
            
        plt.figure(figsize=(12, 6), dpi=self.vis_config['dpi'])
        
        predicted = np.array(self.metrics_history['predicted_power'])
        actual = np.array(self.metrics_history['actual_power'])
        
        plt.plot(actual, label='Actual Power', alpha=0.7)
        plt.plot(predicted, label='Predicted Power', alpha=0.7)
        
        if self.vis_config['plots']['predictions']['confidence_interval']:
            error = np.abs(predicted - actual)
            std_error = np.std(error)
            plt.fill_between(range(len(predicted)),
                           predicted - 2*std_error,
                           predicted + 2*std_error,
                           alpha=0.2)
        
        plt.title('Power Consumption: Prediction vs Actual')
        plt.xlabel('Time Step')
        plt.ylabel('Power (kWh)')
        plt.legend()
        plt.grid(True)
        
        if save and self.vis_config['save_plots']:
            save_path = self.plots_dir / self.vis_config['plots']['predictions']['filename']
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
            
        if self.vis_config['show_plots']:
            plt.show()
        plt.close()
        
    def plot_error_distribution(self, save: bool = True):
        """绘制预测误差分布"""
        if (not self.vis_config['enabled'] or 
            not self.vis_config['plots']['error_dist']['enabled'] or
            'prediction_error' not in self.metrics_history):
            return
            
        plt.figure(figsize=(10, 6), dpi=self.vis_config['dpi'])
        
        errors = np.array(self.metrics_history['prediction_error'])
        bins = self.vis_config['plots']['error_dist']['bins']
        
        sns.histplot(errors, kde=True, bins=bins)
        plt.title('Prediction Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        
        if save and self.vis_config['save_plots']:
            save_path = self.plots_dir / self.vis_config['plots']['error_dist']['filename']
            plt.savefig(save_path, dpi=self.vis_config['dpi'], bbox_inches='tight')
            
        if self.vis_config['show_plots']:
            plt.show()
        plt.close()
        
    def _plot_metric(self, ax, data: List[float], title: str, 
                    xlabel: str, ylabel: str, window: int = 10):
        """辅助函数：绘制平滑的指标曲线"""
        if not data:
            return
            
        y = np.array(data)
        if window > 1:
            y_smooth = np.convolve(y, np.ones(window)/window, mode='valid')
            x_smooth = np.arange(len(y_smooth))
            ax.plot(x_smooth, y_smooth, 'b-', alpha=0.7, label=f'Moving Average (w={window})')
            
        x = np.arange(len(y))
        ax.plot(x, y, 'k.', alpha=0.3, label='Raw Data')
        
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
    def close(self):
        """关闭 TensorBoard writer"""
        if self.writer:
            self.writer.close()
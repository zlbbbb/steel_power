"""
Model Analysis Module
Current Date and Time (UTC): 2025-02-27 05:55:29
Current User: zlbbbb
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
from scipy import stats
import torch
from datetime import datetime

class ModelAnalyzer:
    def __init__(self, save_dir: Path):
        """
        初始化模型分析器
        
        Args:
            save_dir: 保存分析结果的目录
        """
        self.save_dir = save_dir
        self.plots_dir = save_dir / "analysis_plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置绘图样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def plot_training_history(self, 
                            losses: List[float], 
                            rewards: List[float], 
                            smoothing_window: int = 50):
        """
        绘制训练损失和回报历史
        
        Args:
            losses: 训练损失历史
            rewards: 回报历史
            smoothing_window: 平滑窗口大小
        """
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 计算移动平均
        smoothed_losses = self._smooth_curve(losses, smoothing_window)
        smoothed_rewards = self._smooth_curve(rewards, smoothing_window)
        
        # 绘制损失曲线
        episodes = range(len(losses))
        ax1.plot(episodes, losses, 'lightgray', alpha=0.3, label='Raw Loss')
        ax1.plot(episodes, smoothed_losses, 'b', label=f'Smoothed Loss (window={smoothing_window})')
        ax1.set_title('Training Loss History')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制回报曲线
        ax2.plot(episodes, rewards, 'lightgray', alpha=0.3, label='Raw Reward')
        ax2.plot(episodes, smoothed_rewards, 'g', label=f'Smoothed Reward (window={smoothing_window})')
        ax2.set_title('Training Reward History')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def analyze_prediction_errors(self, 
                                predictions: np.ndarray, 
                                actual_values: np.ndarray) -> Dict[str, Any]:
        """
        分析预测误差
        
        Args:
            predictions: 模型预测值
            actual_values: 实际值
            
        Returns:
            包含误差分析结果的字典
        """
        errors = predictions - actual_values
        abs_errors = np.abs(errors)
        rel_errors = np.abs(errors / actual_values) * 100  # 相对误差百分比
        
        # 计算基本统计量
        error_stats = {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'mean_abs_error': np.mean(abs_errors),
            'median_abs_error': np.median(abs_errors),
            'mean_rel_error': np.mean(rel_errors),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'rmse': np.sqrt(np.mean(np.square(errors))),
            'mape': np.mean(rel_errors)
        }
        
        # 绘制误差分析图
        self._plot_error_analysis(errors, predictions, actual_values)
        
        return error_stats
        
    def _plot_error_analysis(self, 
                           errors: np.ndarray, 
                           predictions: np.ndarray, 
                           actual_values: np.ndarray):
        """
        绘制详细的误差分析图
        """
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 误差分布直方图
        ax1 = plt.subplot(321)
        sns.histplot(errors, kde=True, ax=ax1)
        ax1.set_title('Error Distribution')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        
        # 2. Q-Q图
        ax2 = plt.subplot(322)
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        # 3. 误差随时间变化
        ax3 = plt.subplot(323)
        ax3.plot(errors, 'b.', alpha=0.5)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_title('Error vs. Time')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Error')
        
        # 4. 预测值vs实际值散点图
        ax4 = plt.subplot(324)
        ax4.scatter(actual_values, predictions, alpha=0.5)
        min_val = min(actual_values.min(), predictions.min())
        max_val = max(actual_values.max(), predictions.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
        ax4.set_title('Predicted vs Actual Values')
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        
        # 5. 相对误差百分比分布
        ax5 = plt.subplot(325)
        rel_errors = np.abs(errors / actual_values) * 100
        sns.histplot(rel_errors, kde=True, ax=ax5)
        ax5.set_title('Relative Error Distribution (%)')
        ax5.set_xlabel('Relative Error (%)')
        ax5.set_ylabel('Frequency')
        
        # 6. 误差与预测值关系
        ax6 = plt.subplot(326)
        ax6.scatter(predictions, errors, alpha=0.5)
        ax6.axhline(y=0, color='r', linestyle='--')
        ax6.set_title('Error vs Predicted Values')
        ax6.set_xlabel('Predicted Values')
        ax6.set_ylabel('Error')
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _smooth_curve(self, data: List[float], window: int) -> np.ndarray:
        """平滑曲线"""
        kernel = np.ones(window) / window
        return np.convolve(data, kernel, mode='same')
        
    def generate_analysis_report(self, 
                               error_stats: Dict[str, float], 
                               training_time: float,
                               model_params: int):
        """
        生成分析报告
        
        Args:
            error_stats: 误差统计信息
            training_time: 训练时间（秒）
            model_params: 模型参数数量
        """
        report = f"""
        模型分析报告
        生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        1. 模型性能指标
        ---------------
        RMSE: {error_stats['rmse']:.4f}
        MAE: {error_stats['mean_abs_error']:.4f}
        MAPE: {error_stats['mape']:.2f}%
        
        2. 误差统计
        ---------------
        平均误差: {error_stats['mean_error']:.4f}
        误差标准差: {error_stats['std_error']:.4f}
        最大绝对误差: {error_stats['max_error']:.4f}
        最小绝对误差: {error_stats['min_error']:.4f}
        中位数绝对误差: {error_stats['median_abs_error']:.4f}
        
        3. 模型信息
        ---------------
        参数数量: {model_params:,}
        训练时间: {training_time:.2f} 秒
        
        4. 可视化结果
        ---------------
        训练历史图表: {self.plots_dir / 'training_history.png'}
        误差分析图表: {self.plots_dir / 'error_analysis.png'}
        """
        
        # 保存报告
        with open(self.save_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
            
        return report
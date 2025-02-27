"""
Training Analysis Tools for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 11:00:07
Current User: zlbbbb
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

class ModelAnalyzer:
    """模型分析器类，用于生成训练分析报告和可视化"""
    
    def __init__(self, output_dir: Union[str, Path]):
        """
        初始化模型分析器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(__name__)
        
        # 设置可视化样式
        plt.style.use('seaborn')
        sns.set_theme(style="whitegrid")
        
    def generate_analysis_report(self,
                               metrics_history: Dict[str, List[float]],
                               config: Dict[str, Any],
                               hyperparameters: Dict[str, Any],
                               training_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成训练分析报告
        
        Args:
            metrics_history: 训练过程中的指标历史记录
            config: 训练配置
            hyperparameters: 超参数设置
            training_info: 训练信息（如训练时间、轮次等）
            
        Returns:
            包含分析结果的字典
        """
        try:
            # 创建报告目录
            report_dir = self.output_dir / f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 分析训练指标
            metrics_analysis = self._analyze_metrics(metrics_history)
            
            # 生成训练曲线
            plot_paths = self._generate_training_plots(
                metrics_history,
                save_dir=report_dir
            )
            
            # 编写报告
            report = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metrics_analysis': metrics_analysis,
                'plots': plot_paths,
                'training_config': config,
                'hyperparameters': hyperparameters,
                'training_info': training_info
            }
            
            # 保存报告
            report_path = report_dir / 'analysis_report.json'
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
                
            self.logger.info(f"分析报告已生成: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"生成分析报告时出错: {str(e)}")
            raise
            
    def _analyze_metrics(self, metrics_history: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        分析训练指标
        
        Args:
            metrics_history: 训练指标历史记录
            
        Returns:
            指标分析结果
        """
        analysis = {}
        
        for metric_name, values in metrics_history.items():
            values = np.array(values)
            analysis[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'final': float(values[-1]),
                'improvement': float(values[-1] - values[0])
            }
            
        return analysis
        
    def _generate_training_plots(self,
                               metrics_history: Dict[str, List[float]],
                               save_dir: Path) -> Dict[str, str]:
        """
        生成训练过程的可视化图表
        
        Args:
            metrics_history: 训练指标历史记录
            save_dir: 图表保存目录
            
        Returns:
            图表文件路径字典
        """
        plot_paths = {}
        
        # 创建图表保存目录
        plots_dir = save_dir / 'plots'
        plots_dir.mkdir(exist_ok=True)
        
        # 为每个指标生成曲线图
        for metric_name, values in metrics_history.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制训练曲线
            ax.plot(values, label=metric_name)
            
            # 添加滑动平均线
            window_size = min(len(values) // 10, 100)
            if window_size > 1:
                smoothed = pd.Series(values).rolling(window=window_size).mean()
                ax.plot(smoothed, label=f'{metric_name} (滑动平均)', alpha=0.7)
            
            ax.set_title(f'{metric_name} 训练曲线')
            ax.set_xlabel('训练步数')
            ax.set_ylabel(metric_name)
            ax.legend()
            ax.grid(True)
            
            # 保存图表
            plot_path = plots_dir / f'{metric_name}_curve.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            plot_paths[metric_name] = str(plot_path)
            
        return plot_paths
        
    def analyze_predictions(self,
                          predictions: np.ndarray,
                          actual_values: np.ndarray,
                          save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析预测结果
        
        Args:
            predictions: 模型预测值
            actual_values: 实际值
            save_dir: 结果保存目录
            
        Returns:
            预测分析结果
        """
        if save_dir is None:
            save_dir = self.output_dir / 'prediction_analysis'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算误差指标
        errors = predictions - actual_values
        mse = np.mean(errors ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(errors))
        mape = np.mean(np.abs(errors / actual_values)) * 100
        
        # 生成预测对比图
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_values, label='实际值')
        ax.plot(predictions, label='预测值')
        ax.set_title('预测值与实际值对比')
        ax.set_xlabel('样本')
        ax.set_ylabel('值')
        ax.legend()
        ax.grid(True)
        
        plot_path = save_dir / 'predictions_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 生成误差分布图
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title('预测误差分布')
        ax.set_xlabel('误差')
        ax.set_ylabel('频次')
        
        error_plot_path = save_dir / 'error_distribution.png'
        plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        return {
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape)
            },
            'plots': {
                'comparison': str(plot_path),
                'error_distribution': str(error_plot_path)
            }
        }
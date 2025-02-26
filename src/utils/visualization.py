"""
Visualization Utility Functions
Current Date and Time (UTC): 2025-02-26 13:38:10
Current User: zlbbbb
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def set_style():
    """设置绘图样式"""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False

def plot_training_curves(
    metrics: Dict[str, List[Tuple[int, float]]],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True
):
    """
    绘制训练曲线
    
    Args:
        metrics: 指标字典，格式为 {metric_name: [(step, value), ...]}
        save_path: 保存路径
        show: 是否显示图像
    """
    set_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4*n_metrics))
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        steps, vals = zip(*values)
        ax.plot(steps, vals, label=name, marker='.')
        ax.set_xlabel('Steps')
        ax.set_ylabel(name)
        ax.legend()
        
        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 标注最佳值
        best_idx = np.argmax(vals) if 'reward' in name.lower() else np.argmin(vals)
        best_step, best_val = steps[best_idx], vals[best_idx]
        ax.scatter(best_step, best_val, color='red', s=100, zorder=5)
        ax.annotate(f'Best: {best_val:.4f}',
                   (best_step, best_val),
                   xytext=(10, 10),
                   textcoords='offset points')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"训练曲线已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()

def plot_prediction_results(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    title: Optional[str] = None
):
    """
    绘制预测结果对比图
    
    Args:
        true_values: 真实值
        pred_values: 预测值
        save_path: 保存路径
        show: 是否显示图像
        title: 图表标题
    """
    set_style()
    
    fig = plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2, figure=fig)
    
    # 时序对比图
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(true_values, label='True Values', alpha=0.7)
    ax1.plot(pred_values, label='Predictions', alpha=0.7)
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.set_title('Prediction vs True Values')
    
    # 散点图
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(true_values, pred_values, alpha=0.5)
    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    ax2.set_xlabel('True Values')
    ax2.set_ylabel('Predictions')
    ax2.legend()
    ax2.set_title('Prediction Scatter Plot')
    
    # 误差分布图
    ax3 = fig.add_subplot(gs[1, 1])
    errors = pred_values - true_values
    sns.histplot(errors, kde=True, ax=ax3)
    ax3.set_xlabel('Prediction Error')
    ax3.set_ylabel('Count')
    ax3.set_title('Error Distribution')
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"预测结果图已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()

def plot_feature_importance(
    feature_names: List[str],
    importance_scores: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    title: str = 'Feature Importance'
):
    """
    绘制特征重要性图
    
    Args:
        feature_names: 特征名列表
        importance_scores: 重要性得分
        save_path: 保存路径
        show: 是否显示图像
        title: 图表标题
    """
    set_style()
    
    # 排序特征重要性
    idx = np.argsort(importance_scores)
    names = np.array(feature_names)[idx]
    scores = importance_scores[idx]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), scores, align='center')
    plt.yticks(range(len(names)), names)
    plt.xlabel('Importance Score')
    plt.title(title)
    
    # 添加数值标签
    for i, score in enumerate(scores):
        plt.text(score, i, f'{score:.4f}',
                va='center',
                ha='left',
                fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"特征重要性图已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()

def plot_error_distribution(
    errors: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    bins: int = 50
):
    """
    绘制误差分布图
    
    Args:
        errors: 预测误差数组
        save_path: 保存路径
        show: 是否显示图像
        bins: 直方图箱数
    """
    set_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 误差直方图
    sns.histplot(errors, kde=True, ax=ax1, bins=bins)
    ax1.set_xlabel('Prediction Error')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    
    # 误差Q-Q图
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    
    # 添加统计信息
    stats_text = (
        f'Mean: {np.mean(errors):.4f}\n'
        f'Std: {np.std(errors):.4f}\n'
        f'Median: {np.median(errors):.4f}\n'
        f'Skewness: {stats.skew(errors):.4f}\n'
        f'Kurtosis: {stats.kurtosis(errors):.4f}'
    )
    ax1.text(0.95, 0.95, stats_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"误差分布图已保存至: {save_path}")
    
    if show:
        plt.show()
    plt.close()
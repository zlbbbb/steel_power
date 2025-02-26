"""
Model Evaluator Implementation
Current Date and Time (UTC): 2025-02-26 08:31:36
Current User: zlbbbb
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import sys
import os
# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# 使用绝对导入
from src.models.agent import DQNAgent
from src.models.environment import SteelPowerEnv


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, config: Dict):
        """
        初始化评估器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 创建评估结果保存目录
        self.results_dir = Path(config['training']['log_dir']) / 'evaluation'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化环境和智能体
        self.env = SteelPowerEnv(config)
        self.agent = DQNAgent(config)
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def evaluate(self, 
                model_path: Path,
                num_episodes: Optional[int] = None,
                save_predictions: bool = True,
                plot_results: bool = True) -> Dict:
        """
        评估模型性能
        
        Args:
            model_path: 模型检查点路径
            num_episodes: 评估轮数
            save_predictions: 是否保存预测结果
            plot_results: 是否绘制评估图表
            
        Returns:
            评估指标字典
        """
        try:
            # 加载模型
            self.agent.load(model_path)
            self.logger.info(f"模型已从 {model_path} 加载")
            
            num_episodes = num_episodes or 100
            all_metrics = []
            all_predictions = []
            
            # 评估循环
            for episode in tqdm(range(num_episodes), desc="评估进度"):
                episode_metrics = self._evaluate_episode()
                all_metrics.append(episode_metrics)
                all_predictions.extend(episode_metrics['predictions'])
            
            # 计算总体指标
            final_metrics = self._calculate_final_metrics(all_metrics)
            
            # 保存结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_base_path = self.results_dir / f'evaluation_{timestamp}'
            
            # 保存指标
            self._save_metrics(final_metrics, results_base_path)
            
            # 保存预测结果
            if save_predictions:
                self._save_predictions(all_predictions, results_base_path)
            
            # 绘制评估图表
            if plot_results:
                self._plot_evaluation_results(all_predictions, results_base_path)
            
            return final_metrics
            
        except Exception as e:
            self.logger.error(f"评估过程出错: {str(e)}")
            raise
    
    def _evaluate_episode(self) -> Dict:
        """
        评估单个episode
        
        Returns:
            episode评估指标
        """
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        predictions = []
        
        while not done:
            # 使用评估模式选择动作
            action = self.agent.select_action(state, evaluate=True)
            next_state, reward, done, _, info = self.env.step(action)
            
            episode_reward += reward
            predictions.append({
                'step': info['step'],
                'predicted_value': info['predicted_value'],
                'actual_value': info['actual_value'],
                'error': info['error'],
                'relative_error': info['relative_error']
            })
            
            state = next_state
        
        return {
            'episode_reward': episode_reward,
            'predictions': predictions
        }
    
    def _calculate_final_metrics(self, all_metrics: List[Dict]) -> Dict:
        """
        计算最终评估指标
        
        Args:
            all_metrics: 所有episode的指标列表
            
        Returns:
            最终指标字典
        """
        # 提取所有预测结果
        all_predictions = []
        episode_rewards = []
        
        for metrics in all_metrics:
            episode_rewards.append(metrics['episode_reward'])
            all_predictions.extend(metrics['predictions'])
        
        predictions_df = pd.DataFrame(all_predictions)
        
        # 计算整体指标
        mse = np.mean((predictions_df['predicted_value'] - predictions_df['actual_value']) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions_df['predicted_value'] - predictions_df['actual_value']))
        mape = np.mean(predictions_df['relative_error']) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'avg_episode_reward': float(np.mean(episode_rewards)),
            'std_episode_reward': float(np.std(episode_rewards)),
            'min_episode_reward': float(np.min(episode_rewards)),
            'max_episode_reward': float(np.max(episode_rewards)),
            'total_episodes': len(all_metrics)
        }
    
    def _save_metrics(self, metrics: Dict, base_path: Path) -> None:
        """
        保存评估指标
        
        Args:
            metrics: 评估指标字典
            base_path: 基础保存路径
        """
        metrics_path = base_path.with_suffix('.json')
        with open(metrics_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'config': self.config
            }, f, indent=4)
        
        self.logger.info(f"评估指标已保存至: {metrics_path}")
    
    def _save_predictions(self, predictions: List[Dict], base_path: Path) -> None:
        """
        保存预测结果
        
        Args:
            predictions: 预测结果列表
            base_path: 基础保存路径
        """
        predictions_df = pd.DataFrame(predictions)
        csv_path = base_path.with_suffix('.csv')
        predictions_df.to_csv(csv_path, index=False)
        
        self.logger.info(f"预测结果已保存至: {csv_path}")
    
    def _plot_evaluation_results(self, predictions: List[Dict], base_path: Path) -> None:
        """
        绘制评估结果图表
        
        Args:
            predictions: 预测结果列表
            base_path: 基础保存路径
        """
        predictions_df = pd.DataFrame(predictions)
        
        # 设置绘图风格
        plt.style.use('seaborn')
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 预测值vs实际值散点图
        ax = axes[0, 0]
        sns.scatterplot(
            data=predictions_df,
            x='actual_value',
            y='predicted_value',
            alpha=0.5,
            ax=ax
        )
        ax.plot([predictions_df['actual_value'].min(), predictions_df['actual_value'].max()],
                [predictions_df['actual_value'].min(), predictions_df['actual_value'].max()],
                'r--', label='Perfect Prediction')
        ax.set_title('Predicted vs Actual Values')
        ax.set_xlabel('Actual Power Consumption')
        ax.set_ylabel('Predicted Power Consumption')
        ax.legend()
        
        # 2. 预测误差分布
        ax = axes[0, 1]
        sns.histplot(predictions_df['relative_error'] * 100, bins=50, ax=ax)
        ax.set_title('Prediction Error Distribution')
        ax.set_xlabel('Relative Error (%)')
        ax.set_ylabel('Count')
        
        # 3. 误差随时间变化
        ax = axes[1, 0]
        sns.lineplot(data=predictions_df, x='step', y='relative_error', ax=ax)
        ax.set_title('Error Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Relative Error')
        
        # 4. 预测值和实际值随时间变化
        ax = axes[1, 1]
        sns.lineplot(data=predictions_df, x='step', y='actual_value', label='Actual', ax=ax)
        sns.lineplot(data=predictions_df, x='step', y='predicted_value', label='Predicted', ax=ax)
        ax.set_title('Power Consumption Over Time')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Power Consumption')
        ax.legend()
        
        # 保存图表
        plt.tight_layout()
        plot_path = base_path.with_suffix('.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"评估图表已保存至: {plot_path}")


if __name__ == "__main__":
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    
    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logging.error(f"加载配置文件失败: {str(e)}")
        raise
    
    # 创建评估器并运行评估
    evaluator = ModelEvaluator(config)
    
    # 指定模型路径
    model_path = Path(config['training']['checkpoint_dir']) / 'best_model.pth'
    
    # 运行评估
    results = evaluator.evaluate(
        model_path=model_path,
        num_episodes=10,
        save_predictions=True,
        plot_results=True
    )
    
    logging.info("评估完成")
    logging.info(f"RMSE: {results['rmse']:.2f}")
    logging.info(f"MAPE: {results['mape']:.2f}%")
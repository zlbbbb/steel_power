"""
Steel Power Prediction Training Script
Current Date and Time (UTC): 2025-02-26 13:28:36
Current User: zlbbbb

This script combines the functionality of the original trainer.py and train.py,
providing a unified interface for model training.
"""

import os
import sys
from pathlib import Path
import logging
import yaml
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from typing import Dict, Optional, Union, Tuple

# 添加项目根目录到系统路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.agent import DQNAgent
from src.models.environment import SteelPowerEnv
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import setup_logger, ExperimentLogger
from src.utils.metrics import ExperimentMetrics, MetricsCalculator
from src.utils.visualization import plot_training_curves, plot_prediction_results
from src.utils.time_utils import Timer

class Trainer:
    """训练器类，整合了原trainer.py的功能"""
    
    def __init__(self, config: Dict, exp_logger: Optional[ExperimentLogger] = None):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            exp_logger: 实验日志记录器
        """
        self.config = config
        self.logger = exp_logger or setup_logger(__name__)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"使用设备: {self.device}")
        
        # 初始化目录
        self.setup_directories()
        
        # 初始化环境和智能体
        self.env = SteelPowerEnv(config)
        self.agent = DQNAgent(config, self.device)
        
        # 初始化指标记录器
        self.metrics = ExperimentMetrics()
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
        # 记录配置
        self.logger.info("配置初始化完成")
        self.exp_logger = exp_logger
        if exp_logger:
            exp_logger.log_config(config)
    
    def setup_directories(self):
        """设置必要的目录"""
        # 设置检查点目录
        self.checkpoint_dir = Path(self.config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志目录
        self.log_dir = Path(self.config['training']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置tensorboard目录
        if self.config['training'].get('tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(self.log_dir)
                self.use_tensorboard = True
            except ImportError:
                self.logger.warning("Tensorboard未安装，将不会记录详细训练过程")
                self.use_tensorboard = False
        else:
            self.use_tensorboard = False
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch编号
            
        Returns:
            包含训练指标的字典
        """
        self.agent.train()
        epoch_metrics = {
            'reward': [],
            'loss': [],
            'epsilon': []
        }
        
        state, _ = self.env.reset()
        episode_reward = 0
        
        for step in range(self.config['training']['max_steps']):
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _, info = self.env.step(action)
            
            # 存储经验
            self.agent.memory.push(state, action, reward, next_state, done)
            
            # 更新智能体
            if len(self.agent.memory) > self.config['model']['batch_size']:
                loss = self.agent.update()
                epoch_metrics['loss'].append(loss)
            
            episode_reward += reward
            epoch_metrics['epsilon'].append(self.agent.epsilon)
            
            if done:
                break
                
            state = next_state
        
        epoch_metrics['reward'].append(episode_reward)
        
        # 计算平均指标
        metrics = {
            'train_reward': np.mean(epoch_metrics['reward']),
            'train_loss': np.mean(epoch_metrics['loss']) if epoch_metrics['loss'] else 0.0,
            'epsilon': self.agent.epsilon
        }
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        验证模型性能
        
        Returns:
            包含验证指标的字典
        """
        self.agent.eval()
        val_rewards = []
        
        with torch.no_grad():
            for _ in range(self.config['training'].get('val_episodes', 5)):
                state, _ = self.env.reset()
                episode_reward = 0
                
                for _ in range(self.config['training']['max_steps']):
                    action = self.agent.select_action(state, evaluate=True)
                    next_state, reward, done, _, _ = self.env.step(action)
                    episode_reward += reward
                    
                    if done:
                        break
                    
                    state = next_state
                
                val_rewards.append(episode_reward)
        
        return {
            'val_reward': np.mean(val_rewards)
        }
    
    def train(self) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
            训练结果字典
        """
        self.logger.info("开始训练...")
        timer = Timer()
        
        try:
            timer.start()
            
            for epoch in range(self.config['training']['epochs']):
                # 训练一个epoch
                train_metrics = self.train_epoch(epoch)
                
                # 验证
                if (epoch + 1) % self.config['training'].get('val_frequency', 1) == 0:
                    val_metrics = self.validate()
                    train_metrics.update(val_metrics)
                
                # 更新和记录指标
                self.metrics.update(train_metrics, epoch)
                
                # 记录到tensorboard
                if self.use_tensorboard:
                    for name, value in train_metrics.items():
                        self.writer.add_scalar(f'Training/{name}', value, epoch)
                
                # 打印训练进度
                if (epoch + 1) % self.config['training']['eval_frequency'] == 0:
                    self.log_progress(epoch, train_metrics)
                
                # 检查早停和保存模型
                if self.check_early_stopping(train_metrics['val_reward']):
                    self.logger.info(f"触发早停，在epoch {epoch + 1}")
                    break
                
                # 更新探索率
                self.agent.update_epsilon()
            
            training_time = timer.stop()
            
            # 保存最终结果
            results = self.save_results(training_time)
            
            return results
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}", exc_info=True)
            raise
            
        finally:
            if self.use_tensorboard:
                self.writer.close()
    
    def log_progress(self, epoch: int, metrics: Dict[str, float]):
        """记录训练进度"""
        progress_msg = (
            f"Epoch {epoch + 1}/{self.config['training']['epochs']} - "
            f"Train Reward: {metrics['train_reward']:.2f}, "
            f"Val Reward: {metrics.get('val_reward', 0):.2f}, "
            f"Loss: {metrics.get('train_loss', 0):.4f}, "
            f"Epsilon: {metrics['epsilon']:.4f}"
        )
        self.logger.info(progress_msg)
    
    def check_early_stopping(self, val_reward: float) -> bool:
        """
        检查是否需要早停
        
        Args:
            val_reward: 验证奖励
            
        Returns:
            是否需要早停
        """
        if val_reward > self.best_reward:
            self.best_reward = val_reward
            self.no_improvement_count = 0
            self._save_checkpoint('best')
            return False
        
        self.no_improvement_count += 1
        if self.no_improvement_count >= self.config['training']['early_stopping_patience']:
            return True
        
        return False
    
    def _save_checkpoint(self, tag: str):
        """
        保存检查点
        
        Args:
            tag: 检查点标签
        """
        checkpoint = {
            'model_state_dict': self.agent.qnet.state_dict(),
            'target_state_dict': self.agent.target_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config,
            'metrics': self.metrics.get_summary(),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.checkpoint_dir / f'{tag}_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"保存检查点到: {checkpoint_path}")
    
    def save_results(self, training_time: float) -> Dict[str, Any]:
        """
        保存训练结果
        
        Args:
            training_time: 训练时间
            
        Returns:
            结果字典
        """
        results = {
            'metrics': self.metrics.get_summary(),
            'training_time': training_time,
            'early_stopped': self.no_improvement_count >= self.config['training']['early_stopping_patience'],
            'final_epsilon': self.agent.epsilon,
            'total_steps': self.agent.total_steps
        }
        
        # 保存结果
        results_path = self.log_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 绘制训练曲线
        plot_training_curves(
            self.metrics.metrics,
            save_path=self.log_dir / 'training_curves.png'
        )
        
        return results

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Steel Power Prediction Training Script')
    
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                      help='加载检查点路径')
    parser.add_argument('--exp_name', type=str, default=None,
                      help='实验名称')
    parser.add_argument('--debug', action='store_true',
                      help='是否启用调试模式')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    log_level = logging.DEBUG if args.debug else logging.INFO
    
    try:
        # 加载配置
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        # 设置实验名称
        exp_name = args.exp_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建实验日志记录器
        exp_logger = ExperimentLogger(config['training']['log_dir'], exp_name)
        
        # 创建训练器
        trainer = Trainer(config, exp_logger)
        
        # 如果指定了检查点，加载模型
        if args.checkpoint:
            trainer.agent.load_checkpoint(args.checkpoint)
        
        # 开始训练
        results = trainer.train()
        
        # 输出训练结果摘要
        exp_logger.logger.info("训练完成！")
        exp_logger.logger.info(f"最佳验证奖励: {results['metrics']['best']['val_reward']:.4f}")
        exp_logger.logger.info(f"训练轮数: {trainer.agent.total_steps}")
        exp_logger.logger.info(f"训练时长: {results['training_time']:.2f} 秒")
        
        if results['early_stopped']:
            exp_logger.logger.info("训练由于早停策略而终止")
        
    except Exception as e:
        logging.error(f"训练过程出错: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
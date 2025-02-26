"""
Steel Power Prediction Training Script
Current Date and Time (UTC): 2025-02-26 14:45:54
Current User: zlbbbb

This script implements the training loop for the steel power prediction model,
following the configuration defined in config/config.yaml.
"""

import yaml
import torch
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import shutil
import json
import psutil
import time
from typing import Dict, Any, Optional, Tuple

from models.environment import SteelPowerEnv
from models.agent import DQNAgent
from utils.logger import ExperimentLogger, setup_logger

class Trainer:
    """训练器类，管理整个训练过程"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.setup_environment()
        self.setup_logging()
        self.setup_training_components()
        
    def setup_environment(self) -> None:
        """设置训练环境"""
        # 设置随机种子
        torch.manual_seed(self.config['training']['seed'])
        np.random.seed(self.config['training']['seed'])
        
        # 设置设备
        self.device = torch.device(self.config['training']['device'] 
                                 if torch.cuda.is_available() and self.config['training']['device'] == 'cuda'
                                 else 'cpu')
        
        # 创建必要的目录
        self.checkpoint_dir = Path(self.config['training']['checkpoint']['dir'])
        self.log_dir = Path(self.config['training']['logging']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def setup_logging(self) -> None:
        """设置日志系统"""
        # 创建实验名称
        self.exp_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置实验记录器
        self.logger = ExperimentLogger(
            exp_dir=self.log_dir,
            exp_name=self.exp_name,
            config=self.config
        )
        
        # 设置TensorBoard
        if self.config['training']['logging']['tensorboard']:
            self.writer = SummaryWriter(self.log_dir / self.exp_name / 'tensorboard')
        else:
            self.writer = None
            
        # 设置调试日志
        if self.config['debug']['enabled']:
            self.debug_log = setup_logger(
                'debug',
                level=logging.DEBUG,
                log_file=self.log_dir / self.exp_name / 'debug.log'
            )
            
    def setup_training_components(self) -> None:
        """设置训练组件"""
        # 创建环境
        self.env = SteelPowerEnv(self.config['environment'])
        
        # 创建智能体
        self.agent = DQNAgent(
            state_size=self.config['environment']['state_size'],
            action_size=self.config['environment']['action_size'],
            hidden_sizes=self.config['model']['network']['hidden_size'],
            learning_rate=self.config['model']['dqn']['learning_rate'],
            gamma=self.config['model']['dqn']['gamma'],
            epsilon_start=self.config['model']['dqn']['epsilon_start'],
            epsilon_end=self.config['model']['dqn']['epsilon_end'],
            epsilon_decay=self.config['model']['dqn']['epsilon_decay'],
            memory_size=self.config['model']['dqn']['memory_size'],
            batch_size=self.config['model']['dqn']['batch_size'],
            target_update=self.config['model']['dqn']['target_update'],
            use_double=self.config['model']['dqn']['use_double'],
            use_dueling=self.config['model']['dqn']['use_dueling'],
            use_priority=self.config['model']['dqn']['use_priority'],
            device=self.device
        ).to(self.device)
        
        # 设置优化器
        self.setup_optimizer()
        
        # 加载检查点（如果需要）
        if self.config['training']['checkpoint']['load']:
            self.load_checkpoint()
            
    def setup_optimizer(self) -> None:
        """设置优化器和学习率调度器"""
        optimizer_config = self.config['model']['dqn']
        
        # 设置优化器
        if optimizer_config['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(
                self.agent.parameters(),
                lr=optimizer_config['learning_rate']
            )
        else:
            self.optimizer = torch.optim.RMSprop(
                self.agent.parameters(),
                lr=optimizer_config['learning_rate']
            )
            
        # 设置学习率调度器
        scheduler_config = optimizer_config['scheduler']
        if scheduler_config['type'] == 'reduce_on_plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                patience=scheduler_config['patience'],
                factor=scheduler_config['factor'],
                min_lr=scheduler_config['min_lr']
            )
            
    def load_checkpoint(self) -> None:
        """加载检查点"""
        checkpoint_path = self.checkpoint_dir / 'last_model.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.logger.info(f"Loaded checkpoint from {checkpoint_path}")
            
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        保存检查点
        
        Args:
            epoch: 当前轮数
            is_best: 是否是最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # 保存最后的模型
        if self.config['training']['checkpoint']['save_last']:
            last_path = self.checkpoint_dir / 'last_model.pth'
            torch.save(checkpoint, last_path)
            
        # 保存定期检查点
        if epoch % self.config['training']['checkpoint']['save_frequency'] == 0:
            checkpoint_path = self.checkpoint_dir / f'checkpoint_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            
            # 删除旧的检查点（如果超过最大保存数量）
            self.cleanup_checkpoints()
            
        # 保存最佳模型
        if is_best and self.config['training']['checkpoint']['save_best']:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            
    def cleanup_checkpoints(self) -> None:
        """清理旧的检查点文件"""
        max_keep = self.config['training']['checkpoint']['max_keep']
        checkpoints = sorted(
            [f for f in self.checkpoint_dir.glob('checkpoint_*.pth')],
            key=lambda x: int(x.stem.split('_')[1])
        )
        
        while len(checkpoints) > max_keep:
            checkpoints[0].unlink()
            checkpoints = checkpoints[1:]
            
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train') -> None:
        """
        记录训练指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            phase: 训练阶段（train/eval）
        """
        # 更新日志
        self.logger.log_metrics(metrics, step)
        
        # 更新TensorBoard
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{name}', value, step)
                
        # 记录内存使用（如果启用调试）
        if self.config['debug']['save_memory_usage']:
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            if self.writer:
                self.writer.add_scalar('system/memory_usage_mb', memory_usage, step)
                
    def train_episode(self, epoch: int) -> Dict[str, float]:
        """
        训练一个回合
        
        Args:
            epoch: 当前轮数
            
        Returns:
            Dict[str, float]: 训练指标
        """
        state, _ = self.env.reset()
        episode_reward = 0
        episode_loss = []
        step_times = []
        
        for step in range(self.config['training']['max_steps']):
            step_start = time.time()
            
            # 选择动作
            action = self.agent.choose_action(state)
            
            # 执行动作
            next_state, reward, done, truncated, info = self.env.step(action)
            
            # 存储经验
            self.agent.remember(state, action, reward, next_state, done)
            
            # 训练智能体
            if len(self.agent.memory) > self.config['model']['dqn']['batch_size']:
                loss = self.agent.train()
                episode_loss.append(loss)
            
            episode_reward += reward
            state = next_state
            
            # 记录每步执行时间
            step_times.append(time.time() - step_start)
            
            if done or truncated:
                break
                
        # 计算指标
        metrics = {
            'episode_reward': episode_reward,
            'average_loss': np.mean(episode_loss) if episode_loss else 0,
            'epsilon': self.agent.epsilon,
            'episode_length': step + 1,
            'average_step_time': np.mean(step_times)
        }
        
        return metrics
        
    def train(self) -> None:
        """执行训练循环"""
        best_reward = float('-inf')
        patience_counter = 0
        training_start_time = time.time()
        
        try:
            for epoch in range(self.config['training']['epochs']):
                # 训练一个回合
                metrics = self.train_episode(epoch)
                
                # 记录指标
                self.log_metrics(metrics, epoch)
                
                # 更新学习率
                self.scheduler.step(metrics['episode_reward'])
                
                # 检查是否是最佳模型
                is_best = metrics['episode_reward'] > best_reward
                if is_best:
                    best_reward = metrics['episode_reward']
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # 保存检查点
                self.save_checkpoint(epoch, is_best)
                
                # 更新探索率
                self.agent.update_epsilon()
                
                # 打印训练进度
                self.logger.logger.info(
                    f"Epoch {epoch}/{self.config['training']['epochs']} - "
                    f"Reward: {metrics['episode_reward']:.2f}, "
                    f"Loss: {metrics['average_loss']:.4f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )
                
                # 检查早停
                if self.check_early_stopping(patience_counter):
                    break
                    
        except KeyboardInterrupt:
            self.logger.logger.info("Training interrupted by user")
        finally:
            # 清理并保存最终状态
            self.cleanup()
            training_time = time.time() - training_start_time
            self.logger.logger.info(f"Training completed. Total time: {training_time:.2f}s")
            
    def check_early_stopping(self, patience_counter: int) -> bool:
        """
        检查是否需要早停
        
        Args:
            patience_counter: 当前耐心计数器值
            
        Returns:
            bool: 是否需要早停
        """
        if not self.config['training']['early_stopping']['enabled']:
            return False
            
        if patience_counter >= self.config['training']['early_stopping']['patience']:
            self.logger.logger.info(
                f"Early stopping triggered after {patience_counter} epochs "
                f"without improvement"
            )
            return True
        return False
        
    def cleanup(self) -> None:
        """清理资源"""
        if self.writer:
            self.writer.close()
            
        # 保存最终模型
        if self.config['training']['checkpoint']['save_last']:
            self.save_checkpoint(
                self.config['training']['epochs'],
                is_best=False
            )
            
def main():
    """主函数"""
    # 加载配置
    with open("config/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
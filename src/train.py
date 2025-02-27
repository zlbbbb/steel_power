"""
Training Script for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 11:41:23
Current User: zlbbbb
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List, Optional
import json
from tqdm import tqdm
import time
import psutil
import GPUtil
from torch.utils.tensorboard import SummaryWriter

# 添加项目根目录到 PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 本地导入
from src.models.environment import SteelPowerEnv
from src.models.agent import DQNAgent
from src.utils.logger import setup_logger
from src.utils.visualizer import TrainingVisualizer
from src.utils.analysis import ModelAnalyzer

class Trainer:
    """训练管理器类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.setup_training()
        
    def setup_training(self):
        """设置训练环境和组件"""
        # 设置日志
        log_config = self.config['training']['logging']
        self.log_dir = Path(log_config['dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger(
            name="Trainer",
            level=log_config['level'],
            log_file=self.log_dir / "training.log"
        )
        
        # 设置TensorBoard
        if log_config['tensorboard']:
            self.writer = SummaryWriter(self.log_dir / 'tensorboard')
        
        # 设置设备
        self.setup_device()
        
        # 设置随机种子
        self._set_seed(self.config['training']['seed'])
        
        # 创建环境
        self.env = SteelPowerEnv(self.config)
        
        # 创建智能体
        self.setup_agent()
        
        # 设置输出目录
        self.setup_directories()
        
        # 初始化训练状态
        self.initialize_training_state()
        
        # 设置性能监控
        self.setup_performance_monitoring()
        
        self.logger.info("训练器初始化完成")
        
    def setup_device(self):
        """设置计算设备"""
        if torch.cuda.is_available() and self.config['training']['device'] == 'cuda':
            cuda_device = self.config['training']['cuda_device']
            self.device = torch.device(f'cuda:{cuda_device}')
            torch.cuda.set_device(self.device)
            
            if self.config['training']['use_multi_gpu']:
                if torch.cuda.device_count() > 1:
                    self.logger.info(f"使用 {torch.cuda.device_count()} 个GPU训练")
                else:
                    self.logger.warning("未检测到多个GPU，将使用单GPU训练")
        else:
            self.device = torch.device('cpu')
            self.logger.warning("未使用GPU加速")
            
        self.logger.info(f"使用设备: {self.device}")
        
    def setup_agent(self):
        """设置DQN智能体"""
        model_config = self.config['model']
        dqn_config = model_config['dqn']
        
        self.agent = DQNAgent(
            state_dim=self.config['environment']['state_dim'],
            action_dim=self.config['environment']['action_dim'],
            hidden_sizes=model_config['network']['hidden_size'],
            learning_rate=dqn_config['learning_rate'],
            gamma=dqn_config['gamma'],
            epsilon_start=dqn_config['epsilon_start'],
            epsilon_end=dqn_config['epsilon_end'],
            epsilon_decay=dqn_config['epsilon_decay'],
            memory_size=dqn_config['memory_size'],
            batch_size=dqn_config['batch_size'],
            target_update=dqn_config['target_update'],
            use_double=dqn_config['use_double'],
            use_dueling=dqn_config['use_dueling'],
            device=self.device
        )
        
        # 设置优化器调度器
        if dqn_config['scheduler']['type'] == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.agent.optimizer,
                mode='max',
                factor=dqn_config['scheduler']['factor'],
                patience=dqn_config['scheduler']['patience'],
                min_lr=dqn_config['scheduler']['min_lr']
            )
            
    def setup_directories(self):
        """设置输出目录"""
        # 检查点目录
        self.checkpoint_dir = Path(self.config['training']['checkpoint']['dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 分析输出目录
        self.analysis_dir = Path(self.config['evaluation']['output_dir'])
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize_training_state(self):
        """初始化训练状态"""
        self.episodes = 0
        self.total_steps = 0
        self.best_reward = float('-inf')
        self.training_start_time = None
        self.training_end_time = None
        self.training_time = None
        self.early_stopping_counter = 0
        self.best_model_score = float('-inf')
        
        # 初始化指标记录
        self.metrics_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon_values': [],
            'predictions': [],
            'actual_values': [],
            'errors': [],
            'gpu_memory_allocated': [],
            'gpu_memory_cached': [],
            'training_speed': []
        }
        
        # 创建可视化器和分析器
        self.visualizer = TrainingVisualizer(
            output_dir=self.analysis_dir,
            config=self.config['evaluation']['visualization']
        )
        self.analyzer = ModelAnalyzer(self.analysis_dir)
        
    def setup_performance_monitoring(self):
        """设置性能监控"""
        perf_config = self.config['training']['performance_monitoring']
        self.monitor_performance = perf_config['enabled']
        
        if self.monitor_performance:
            self.perf_metrics = {
                'gpu_memory_tracking': perf_config['gpu_memory_tracking'],
                'timing_analysis': perf_config['timing_analysis'],
                'resource_usage': perf_config['resource_usage']
            }
            
    def _set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def _update_metrics(self, 
                       episode_reward: float,
                       episode_length: int,
                       loss: float,
                       predictions: Optional[np.ndarray] = None,
                       actual_values: Optional[np.ndarray] = None,
                       step_time: Optional[float] = None):
        """更新训练指标"""
        # 基本指标
        self.metrics_history['episode_rewards'].append(episode_reward)
        self.metrics_history['episode_lengths'].append(episode_length)
        self.metrics_history['losses'].append(loss)
        self.metrics_history['epsilon_values'].append(self.agent.epsilon)
        
        # 预测相关指标
        if predictions is not None and actual_values is not None:
            self.metrics_history['predictions'].extend(predictions.tolist())
            self.metrics_history['actual_values'].extend(actual_values.tolist())
            self.metrics_history['errors'].extend((predictions - actual_values).tolist())
            
        # 性能指标
        if self.monitor_performance:
            if self.perf_metrics['gpu_memory_tracking'] and torch.cuda.is_available():
                self.metrics_history['gpu_memory_allocated'].append(
                    torch.cuda.memory_allocated(self.device)
                )
                self.metrics_history['gpu_memory_cached'].append(
                    torch.cuda.memory_reserved(self.device)
                )
                
            if self.perf_metrics['timing_analysis'] and step_time is not None:
                self.metrics_history['training_speed'].append(step_time)
                
        # 更新TensorBoard
        if hasattr(self, 'writer'):
            self.writer.add_scalar('Train/Reward', episode_reward, self.episodes)
            self.writer.add_scalar('Train/Loss', loss, self.episodes)
            self.writer.add_scalar('Train/Epsilon', self.agent.epsilon, self.episodes)
            
    def save_checkpoint(self, episode: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'episode': episode,
            'total_steps': self.total_steps,
            'best_reward': self.best_reward,
            'agent_state': self.agent.state_dict(),
            'optimizer_state': self.agent.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        # 保存检查点
        checkpoint_config = self.config['training']['checkpoint']
        if checkpoint_config['save_last'] or (
            episode % checkpoint_config['save_frequency'] == 0):
            checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pth"
            torch.save(checkpoint, checkpoint_path)
            
        # 保存最佳模型
        if is_best and checkpoint_config['save_best']:
            best_model_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_model_path)
            
        # 管理检查点数量
        if checkpoint_config['max_keep'] > 0:
            self._cleanup_checkpoints(checkpoint_config['max_keep'])
            
    def _cleanup_checkpoints(self, max_keep: int):
        """清理旧的检查点文件"""
        checkpoint_files = sorted(
            [f for f in self.checkpoint_dir.glob("checkpoint_ep*.pth")],
            key=lambda x: int(x.stem.split('ep')[1])
        )
        
        while len(checkpoint_files) > max_keep:
            oldest_checkpoint = checkpoint_files.pop(0)
            oldest_checkpoint.unlink()
            
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.episodes = checkpoint['episode']
        self.total_steps = checkpoint['total_steps']
        self.best_reward = checkpoint['best_reward']
        self.agent.load_state_dict(checkpoint['agent_state'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if checkpoint['scheduler_state'] is not None and hasattr(self, 'scheduler'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            
        self.metrics_history = checkpoint['metrics_history']
        
        self.logger.info(f"从 {checkpoint_path} 加载检查点")
        
    def train_episode(self) -> Tuple[float, int, float]:
        """训练单个回合"""
        state = self.env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []
        step_times = []
        
        while True:
            step_start_time = time.time()
            
            # 选择动作
            action = self.agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 存储经验并学习
            self.agent.store_transition(state, action, reward, next_state, done)
            loss = self.agent.learn()
            
            if loss is not None:
                losses.append(loss)
                
            episode_reward += reward
            episode_length += 1
            self.total_steps += 1
            
            # 记录步骤时间
            step_time = time.time() - step_start_time
            step_times.append(step_time)
            
            state = next_state
            
            if done:
                break
                
        # 计算平均值
        avg_loss = np.mean(losses) if losses else 0.0
        avg_step_time = np.mean(step_times)
        
        # 更新指标
        self._update_metrics(
            episode_reward=episode_reward,
            episode_length=episode_length,
            loss=avg_loss,
            predictions=info.get('predictions'),
            actual_values=info.get('actual_values'),
            step_time=avg_step_time
        )
        
        return episode_reward, episode_length, avg_loss
        
    def _check_early_stopping(self, current_reward: float) -> bool:
        """检查是否需要提前停止"""
        early_stopping_config = self.config['training']['early_stopping']
        if not early_stopping_config['enabled']:
            return False
            
        if current_reward > (self.best_model_score + early_stopping_config['min_delta']):
            self.best_model_score = current_reward
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1
            
        if self.early_stopping_counter >= early_stopping_config['patience']:
            self.logger.info("触发提前停止条件")
            return True
            
        return False
        
    def train(self, num_episodes: int):
        """
        训练指定回合数
        
        Args:
            num_episodes: 训练回合数
        """
        try:
            self.training_start_time = datetime.now()
            
            # 创建进度条
            progress_bar = tqdm(range(num_episodes), desc="训练进度")
            
            for episode in progress_bar:
                self.episodes = episode
                
                # 训练一个回合
                episode_reward, episode_length, avg_loss = self.train_episode()
                
                # 更新学习率调度器
                if hasattr(self, 'scheduler'):
                    self.scheduler.step(episode_reward)
                    
                # 检查是否是最佳模型
                is_best = episode_reward > self.best_reward
                if is_best:
                    self.best_reward = episode_reward
                    
                # 更新进度条信息
                progress_bar.set_postfix({
                    'reward': f"{episode_reward:.2f}",
                    'length': episode_length,
                    'loss': f"{avg_loss:.4f}",
                    'epsilon': f"{self.agent.epsilon:.4f}",
                    'lr': f"{self.agent.optimizer.param_groups[0]['lr']:.6f}"
                })
                
                # 定期保存检查点
                if (episode + 1) % self.config['training']['checkpoint']['save_frequency'] == 0:
                    self.save_checkpoint(episode, is_best)
                    
                # 定期进行评估
                if (episode + 1) % self.config['training']['eval_frequency'] == 0:
                    eval_metrics = self.evaluate(
                        num_episodes=self.config['evaluation']['num_episodes']
                    )
                    self.logger.info(f"评估结果: {eval_metrics}")
                    
                    if hasattr(self, 'writer'):
                        for metric, value in eval_metrics.items():
                            self.writer.add_scalar(f'Eval/{metric}', value, episode)
                            
                # 定期进行分析
                if (episode + 1) % self.config['training']['analysis_frequency'] == 0:
                    self.analyze_training()
                    
                # 检查是否需要提前停止
                if self._check_early_stopping(episode_reward):
                    self.logger.info("提前停止训练")
                    break
                    
                # 检查是否达到最大步数
                if self.total_steps >= self.config['training']['max_steps']:
                    self.logger.info("达到最大步数，停止训练")
                    break
                    
            self.training_end_time = datetime.now()
            self.training_time = self.training_end_time - self.training_start_time
            
            # 保存最终模型
            self.save_checkpoint(episode, is_best=False)
            
            # 生成最终分析报告
            final_report = self.analyze_training()
            self.logger.info("训练完成！")
            
            return final_report
            
        except KeyboardInterrupt:
            self.logger.info("用户中断训练")
            self.save_checkpoint(episode, is_best=False)
            raise
            
        except Exception as e:
            self.logger.error(f"训练过程中出错: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
            
        finally:
            # 清理资源
            if hasattr(self, 'writer'):
                self.writer.close()
            self.env.close()
            
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            num_episodes: 评估回合数
            
        Returns:
            评估指标字典
        """
        self.agent.eval()  # 设置为评估模式
        evaluation_metrics = []
        predictions = []
        actual_values = []
        
        try:
            for _ in range(num_episodes):
                state = self.env.reset()
                episode_reward = 0
                episode_predictions = []
                episode_actuals = []
                done = False
                
                while not done:
                    action = self.agent.select_action(state, evaluate=True)
                    next_state, reward, done, info = self.env.step(action)
                    
                    if 'predictions' in info and 'actual_values' in info:
                        episode_predictions.append(info['predictions'])
                        episode_actuals.append(info['actual_values'])
                        
                    episode_reward += reward
                    state = next_state
                    
                evaluation_metrics.append({
                    'episode_reward': episode_reward,
                    'episode_length': len(episode_predictions)
                })
                
                predictions.extend(episode_predictions)
                actual_values.extend(episode_actuals)
                
            # 计算评估指标
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            
            mse = np.mean((predictions - actual_values) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(predictions - actual_values))
            mape = np.mean(np.abs((predictions - actual_values) / actual_values)) * 100
            
            avg_metrics = {
                'mean_reward': np.mean([m['episode_reward'] for m in evaluation_metrics]),
                'std_reward': np.std([m['episode_reward'] for m in evaluation_metrics]),
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
            
            # 保存预测结果
            if self.config['evaluation']['save_predictions']:
                self._save_predictions(predictions, actual_values)
                
            return avg_metrics
            
        finally:
            self.agent.train()  # 恢复训练模式
            
    def _save_predictions(self, predictions: np.ndarray, actual_values: np.ndarray):
        """保存预测结果"""
        predictions_dir = self.analysis_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_df = pd.DataFrame({
            'predictions': predictions.flatten(),
            'actual_values': actual_values.flatten(),
            'error': predictions.flatten() - actual_values.flatten()
        })
        
        results_df.to_csv(predictions_dir / f'predictions_{timestamp}.csv', index=False)
        
    def analyze_training(self) -> Dict[str, Any]:
        """分析训练过程并生成报告"""
        try:
            report = self.analyzer.generate_analysis_report(
                metrics_history=self.metrics_history,
                config=self.config,
                hyperparameters={
                    'learning_rate': self.agent.optimizer.param_groups[0]['lr'],
                    'batch_size': self.agent.batch_size,
                    'gamma': self.agent.gamma,
                    'epsilon': self.agent.epsilon,
                    'target_update': self.agent.target_update,
                    'use_double': self.agent.use_double,
                    'use_dueling': self.agent.use_dueling
                },
                training_info={
                    'total_steps': self.total_steps,
                    'total_episodes': self.episodes,
                    'training_time': str(self.training_time),
                    'device': str(self.device),
                    'timestamp_start': str(self.training_start_time),
                    'timestamp_end': str(self.training_end_time)
                }
            )
            
            # 生成可视化
            if self.config['evaluation']['visualization']['enabled']:
                self.visualizer.plot_training_metrics(self.metrics_history)
                
            self.logger.info(f"训练分析报告已生成")
            return report
            
        except Exception as e:
            self.logger.error(f"生成训练分析报告时出错: {str(e)}")
            raise

def main():
    """主函数"""
    # 加载配置
    config_path = project_root / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    try:
        # 创建训练器
        trainer = Trainer(config)
        
        # 如果配置了加载检查点
        if config['training']['checkpoint']['load']:
            checkpoint_path = Path(config['evaluation']['model_path'])
            if checkpoint_path.exists():
                trainer.load_checkpoint(str(checkpoint_path))
                
        # 开始训练
        num_episodes = config['training']['epochs']
        trainer.train(num_episodes)
        
        # 评估模型
        evaluation_results = trainer.evaluate(
            num_episodes=config['evaluation']['num_episodes']
        )
        
        print("\n最终评估结果:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value:.4f}")
            
    except Exception as e:
        logging.error(f"训练过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
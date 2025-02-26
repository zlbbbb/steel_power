"""
Model Trainer Implementation
Current Date and Time (UTC): 2025-02-26 08:27:15
Current User: zlbbbb
"""

import torch
import numpy as np
from typing import Dict, Optional
import logging
from pathlib import Path
from datetime import datetime
import json

# 基础日志记录器
class BasicLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training_log.csv"
        self.metrics = []
        
    def add_scalar(self, tag: str, value: float, step: int):
        self.metrics.append({
            'step': step,
            'tag': tag,
            'value': value,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    def close(self):
        import pandas as pd
        pd.DataFrame(self.metrics).to_csv(self.log_file, index=False)
        
    def __del__(self):
        self.close()

from .agent import DQNAgent
from .environment import SteelPowerEnv


class ModelTrainer:
    """模型训练器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 设置保存目录
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用基础日志记录器替代 tensorboard
        self.log_dir = Path(config['training']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = BasicLogger(self.log_dir)
        
        # 初始化环境和智能体
        self.env = SteelPowerEnv(config)
        self.agent = DQNAgent(config)
        
        self.best_reward = float('-inf')
        self.no_improvement_count = 0
        
    def train(self) -> Dict:
        """训练模型"""
        try:
            self.logger.info("开始训练...")
            
            for epoch in range(self.config['training']['epochs']):
                epoch_rewards = []
                epoch_losses = []
                
                state, _ = self.env.reset()
                episode_reward = 0
                
                for step in range(self.config['training']['max_steps']):
                    # 选择动作
                    action = self.agent.select_action(state)
                    
                    # 执行动作
                    next_state, reward, done, _, info = self.env.step(action)
                    
                    # 更新智能体
                    update_info = self.agent.update(state, action, reward, next_state, done)
                    
                    episode_reward += reward
                    epoch_losses.append(update_info['loss'])
                    
                    if done:
                        break
                        
                    state = next_state
                
                epoch_rewards.append(episode_reward)
                avg_reward = np.mean(epoch_rewards)
                avg_loss = np.mean(epoch_losses)
                
                # 记录训练信息
                self.writer.add_scalar('Training/Average_Reward', avg_reward, epoch)
                self.writer.add_scalar('Training/Average_Loss', avg_loss, epoch)
                self.writer.add_scalar('Training/Epsilon', self.agent.epsilon, epoch)
                
                # 打印训练进度
                if (epoch + 1) % self.config['training']['eval_frequency'] == 0:
                    self.logger.info(
                        f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                        f"Avg. Reward: {avg_reward:.2f}, "
                        f"Avg. Loss: {avg_loss:.4f}, "
                        f"Epsilon: {self.agent.epsilon:.4f}"
                    )
                
                # 保存检查点
                if (epoch + 1) % self.config['training']['save_frequency'] == 0:
                    self._save_checkpoint(epoch + 1, avg_reward)
                
                # 早停
                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    self.no_improvement_count = 0
                    self._save_checkpoint(epoch + 1, avg_reward, is_best=True)
                else:
                    self.no_improvement_count += 1
                    
                if self.no_improvement_count >= self.config['training']['early_stopping_patience']:
                    self.logger.info("早停触发，停止训练")
                    break
            
            return {
                "best_reward": self.best_reward,
                "final_epsilon": self.agent.epsilon,
                "total_steps": self.agent.total_steps
            }
            
        except Exception as e:
            self.logger.error(f"训练过程出错: {str(e)}")
            raise
        finally:
            self.writer.close()
    
    def _save_checkpoint(self, epoch: int, reward: float, is_best: bool = False):
        """保存检查点"""
        checkpoint_name = f"checkpoint_epoch_{epoch}.pth"
        if is_best:
            checkpoint_name = "best_model.pth"
            
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        self.agent.save(checkpoint_path)
        
        # 保存训练元数据
        metadata = {
            'epoch': epoch,
            'reward': reward,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_best': is_best
        }
        
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)


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
    
    # 创建并运行训练器
    trainer = ModelTrainer(config)
    results = trainer.train()
    
    logging.info(f"训练完成，最佳奖励: {results['best_reward']:.2f}")
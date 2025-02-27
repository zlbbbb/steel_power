"""
Training Script for Steel Power Prediction
Current Date and Time (UTC): 2025-02-26 15:48:32
Current User: zlbbbb
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any
import json

# 添加项目根目录到 PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 本地导入
from src.models.environment import SteelPowerEnv
from src.models.agent import DQNAgent
from src.utils.logger import ExperimentLogger
from src.utils.visualizer import TrainingVisualizer

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config (Dict[str, Any]): 训练配置
        """
        self.config = config
        self.device = torch.device(self.config['training']['device'])
        print(f"Using device: {self.device}")
        
        # 设置日志
        self.setup_logging()
        
        # 设置随机种子
        self.set_seed(self.config['training']['seed'])
        
        # 设置训练组件
        self.setup_components()

        # 初始化可视化器
        self.visualizer = TrainingVisualizer(self.log_dir, self.config)
        
        # 保存初始配置
        self.save_config()
        
    def set_seed(self, seed: int):
        """设置随机种子"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        
    def setup_logging(self):
        """设置日志系统"""
        # 获取日志配置
        log_config = self.config['training']['logging']
        
        # 创建时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置日志目录
        self.log_dir = Path(log_config['dir']) / f"train_{self.timestamp}"
        self.model_dir = Path(self.config['training']['checkpoint']['dir'])
        
        # 创建必要的目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志记录器
        self.logger = logging.getLogger(f"train_{self.timestamp}")
        self.logger.setLevel(getattr(logging, log_config['level']))
        
        # 文件处理器
        fh = logging.FileHandler(self.log_dir / "training.log")
        fh.setLevel(logging.INFO)
        
        # 控制台处理器
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # 添加处理器
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info(f"Training started at {self.timestamp}")
        
    def setup_components(self):
        """设置训练组件"""
        # 创建环境
        env_config = self.config['environment']
        self.env = SteelPowerEnv({
            'state_dim': env_config['state_dim'],
            'action_dim': env_config['action_dim'],
            'power': env_config['power'],
            'rewards': env_config['rewards']
        })
        self.logger.info("环境创建成功")
        
        # 创建智能体
        self.agent = DQNAgent(
            state_dim=env_config['state_dim'],
            action_dim=env_config['action_dim'],
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
            device=self.device
        )
        self.logger.info("智能体创建成功")
        
    def save_config(self):
        """保存配置到文件"""
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"配置已保存至: {config_path}")
        
    def train(self):
        """执行训练过程"""
        self.logger.info("开始训练...")
        total_episodes = self.config['training']['epochs']
        max_steps = self.config['training']['max_steps']
        best_reward = float('-inf')
        
        for episode in range(total_episodes):
            state, _ = self.env.reset()
            episode_reward = 0
            episode_loss = 0
            n_steps = 0
            
            for step in range(max_steps):
                # 选择动作
                action = self.agent.select_action(state)
                
                # 执行动作
                next_state, reward, done, truncated, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_transition(state, action, reward, next_state, done)
                
                # 更新智能体
                loss = self.agent.learn()

                if loss is not None:
                    episode_loss += loss
                
                episode_reward += reward
                state = next_state
                n_steps += 1

                # 更新可视化指标
                if 'power_value' in info and 'target_power' in info:
                    metrics = {
                        'predicted_power': info['power_value'],
                        'actual_power': info['target_power'],
                        'prediction_error': abs(info['power_value'] - info['target_power'])
                    }
                    self.visualizer.update_metrics(episode * max_steps + step, metrics)

            # 计算平均损失
            avg_loss = episode_loss / n_steps if n_steps > 0 else 0

            # 更新训练指标
            metrics = {
                'episode_reward': episode_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon
            }
            self.visualizer.update_metrics(episode, metrics)
                
            # 日志记录和可视化
            if episode % self.config['training']['logging']['log_frequency'] == 0:
                self.logger.info(
                    f"Episode: {episode}/{total_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}"
                )
                # 绘制训练曲线
                self.visualizer.plot_training_curves()
                self.visualizer.plot_power_prediction()
                self.visualizer.plot_error_distribution()

            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model('best')
                
            # 定期保存模型
            if episode % self.config['training']['checkpoint']['save_frequency'] == 0:
                self.save_model(f"episode_{episode}")
                
                
        # 训练结束，保存最终可视化结果
        self.visualizer.plot_training_curves()
        self.visualizer.plot_power_prediction()
        self.visualizer.plot_error_distribution()
        self.visualizer.close()
        
        self.logger.info("训练完成！")
        
    def save_model(self, tag: str):
        """
        保存模型
        
        Args:
            tag (str): 模型标签
        """
        save_path = self.model_dir / f"model_{tag}.pth"
        self.agent.save(save_path)
        self.logger.info(f"模型已保存至: {save_path}")
        
    def load_model(self, model_path: str):
        """
        加载模型
        
        Args:
            model_path (str): 模型路径
        """
        self.agent.load(model_path)
        self.logger.info(f"模型已从 {model_path} 加载")

if __name__ == "__main__":
    try:
        # 加载配置文件
        config_path = project_root / "config" / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            print("已加载配置文件")
        
        # 创建训练器并开始训练
        trainer = Trainer(config)
        trainer.train()
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise
"""
DQN Agent Implementation
Current Date and Time (UTC): 2025-02-26 08:46:50
Current User: zlbbbb
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple
import logging

# 修改为绝对导入
from src.models.networks import DuelingDQN, DQN


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, config: Dict):
        """
        初始化DQN智能体
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        
        # 环境参数
        self.state_size = config['environment']['state_size']
        self.action_size = config['environment']['action_size']
        
        # 初始化网络
        NetworkClass = DuelingDQN if config['model']['use_dueling'] else DQN
        self.policy_net = NetworkClass(
            self.state_size,
            self.action_size,
            config['model']['hidden_size']
        ).to(self.device)
        
        self.target_net = NetworkClass(
            self.state_size,
            self.action_size,
            config['model']['hidden_size']
        ).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config['model']['learning_rate']
        )
        
        # 经验回放
        self.memory = deque(maxlen=config['model']['memory_size'])
        
        # 参数
        self.epsilon = config['model']['epsilon_start']
        self.epsilon_end = config['model']['epsilon_end']
        self.epsilon_decay = config['model']['epsilon_decay']
        self.gamma = config['model']['gamma']
        self.batch_size = config['model']['batch_size']
        
        self.total_steps = 0
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否为评估模式
            
        Returns:
            选择的动作索引
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
            
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, 
              reward: float, next_state: np.ndarray, 
              done: bool) -> Dict[str, float]:
        """
        更新智能体
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
            done: 是否结束
            
        Returns:
            更新信息字典
        """
        # 存储经验
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < self.batch_size:
            return {"loss": 0.0}
            
        # 采样经验
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # 准备批次数据
        state_batch = torch.FloatTensor(np.array(batch[0])).to(self.device)
        action_batch = torch.LongTensor(np.array(batch[1])).to(self.device)
        reward_batch = torch.FloatTensor(np.array(batch[2])).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch[3])).to(self.device)
        done_batch = torch.FloatTensor(np.array(batch[4])).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config['model']['use_double']:
                next_action = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_state_batch).gather(1, next_action)
            else:
                next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            
            target_q_values = reward_batch.unsqueeze(1) + \
                            (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.total_steps % 100 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 更新探索率
        self.epsilon = max(self.epsilon_end, 
                         self.epsilon * self.epsilon_decay)
        
        self.total_steps += 1
        
        return {
            "loss": loss.item(),
            "epsilon": self.epsilon,
            "q_mean": current_q_values.mean().item()
        }
        
    def save(self, path: Path):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'config': self.config
        }, path)
        self.logger.info(f"模型已保存至: {path}")
        
    def load(self, path: Path):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']
        self.logger.info(f"模型已从 {path} 加载")


# 确保导出 DQNAgent 类
__all__ = ['DQNAgent']
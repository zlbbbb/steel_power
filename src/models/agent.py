"""
DQN Agent Implementation
Current Date and Time (UTC): 2025-02-26 16:02:12
Current User: zlbbbb
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import random
from pathlib import Path
import os
import sys
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.models.networks import DQN, DuelingDQN

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: List[int],
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10,
        use_double: bool = True,
        use_dueling: bool = True,
        device: str = "cuda"
    ):
        """初始化 DQN 智能体"""
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double = use_double
        self.device = torch.device(device)
        
        # 初始化经验回放缓冲区
        self.memory = deque(maxlen=memory_size)
        self.update_count = 0
        
        # 创建网络
        NetworkClass = DuelingDQN if use_dueling else DQN
        self.policy_net = NetworkClass(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net = NetworkClass(state_dim, action_dim, hidden_sizes).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # 设置为评估模式
        self.policy_net.eval()
        self.target_net.eval()
        
        # 创建优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def select_action(self, state: np.ndarray) -> int:
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            self.policy_net.eval()  # 确保在评估模式
            q_values = self.policy_net(state)
            return q_values.argmax().item()
            
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
        
    def learn(self) -> Optional[float]:
        """从经验中学习"""
        if len(self.memory) < self.batch_size:
            return None
            
        # 采样经验批次
        transitions = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        # 转换为张量
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前 Q 值
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # 计算目标 Q 值
        with torch.no_grad():
            self.target_net.eval()  # 确保目标网络在评估模式
            if self.use_double:
                self.policy_net.eval()  # 确保策略网络在评估模式
                next_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
                next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
                self.policy_net.train()  # 恢复训练模式
            else:
                next_q_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
        # 计算目标 Q 值
        target_q_values = reward_batch.unsqueeze(1) + \
                         (1 - done_batch.unsqueeze(1)) * self.gamma * next_q_values
        
        # 计算 Huber 损失
        loss = nn.SmoothL1Loss()(current_q_values, target_q_values)
        
        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # 更新探索率
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # 返回训练模式
        self.policy_net.eval()
        
        return loss.item()
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

if __name__ == "__main__":
    # 测试代码
    agent = DQNAgent(
        state_dim=24,
        action_dim=100,
        hidden_sizes=[256, 128, 64]
    )
    
    # 测试动作选择
    state = np.random.rand(24)
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # 测试经验存储和学习
    next_state = np.random.rand(24)
    agent.store_transition(state, action, 1.0, next_state, False)
    
    # 添加更多经验
    for _ in range(64):
        state = np.random.rand(24)
        action = agent.select_action(state)
        next_state = np.random.rand(24)
        agent.store_transition(state, action, 1.0, next_state, False)
    
    # 测试学习
    loss = agent.learn()
    print(f"Training loss: {loss}")
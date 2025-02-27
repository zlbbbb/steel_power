"""
Neural Network Models for DQN
Current Date and Time (UTC): 2025-02-26 16:11:44
Current User: zlbbbb
"""

import torch
import torch.nn as nn
from typing import List

class DQN(nn.Module):
    """基础 DQN 网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        """
        初始化 DQN 网络
        
        Args:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_sizes (List[int]): 隐藏层大小列表
        """
        super(DQN, self).__init__()
        
        # 构建网络层
        layers = []
        current_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU()
            ])
            current_dim = hidden_size
            
        # 输出层
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)

class DuelingDQN(nn.Module):
    """Dueling DQN 网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: List[int]):
        """
        初始化 Dueling DQN 网络
        
        Args:
            state_dim (int): 状态空间维度
            action_dim (int): 动作空间维度
            hidden_sizes (List[int]): 隐藏层大小列表
        """
        super(DuelingDQN, self).__init__()
        
        # 特征提取层
        feature_layers = [
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU()
        ]
        self.feature_layer = nn.Sequential(*feature_layers)
        
        # 价值流
        value_layers = []
        current_dim = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            value_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU()
            ])
            current_dim = hidden_size
        value_layers.append(nn.Linear(current_dim, 1))
        self.value_stream = nn.Sequential(*value_layers)
        
        # 优势流
        advantage_layers = []
        current_dim = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            advantage_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU()
            ])
            current_dim = hidden_size
        advantage_layers.append(nn.Linear(current_dim, action_dim))
        self.advantage_stream = nn.Sequential(*advantage_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # 组合价值和优势
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
"""
Neural Network Models for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 10:56:51
Current User: zlbbbb
"""

import torch
import torch.nn as nn
from typing import List

class DQN(nn.Module):
    """标准DQN网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int],
                 dropout_rate: float = 0.1):
        """
        初始化标准DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_sizes: 隐藏层大小列表
            dropout_rate: Dropout比率
        """
        super(DQN, self).__init__()
        
        layers = []
        current_dim = state_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),  # 使用LayerNorm替代BatchNorm
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_size
            
        layers.append(nn.Linear(current_dim, action_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态张量
            
        Returns:
            Q值张量
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int],
                 dropout_rate: float = 0.1):
        """
        初始化Dueling DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_sizes: 隐藏层大小列表
            dropout_rate: Dropout比率
        """
        super(DuelingDQN, self).__init__()
        
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.LayerNorm(hidden_sizes[0]),
            nn.Dropout(dropout_rate)
        )
        
        # 价值流
        value_layers = []
        current_dim = hidden_sizes[0]
        for hidden_size in hidden_sizes[1:]:
            value_layers.extend([
                nn.Linear(current_dim, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate)
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
                nn.ReLU(),
                nn.LayerNorm(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_size
        advantage_layers.append(nn.Linear(current_dim, action_dim))
        self.advantage_stream = nn.Sequential(*advantage_layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态张量
            
        Returns:
            Q值张量
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class DQNetwork(nn.Module):
    """DQN网络封装类"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int],
                 use_dueling: bool = True,
                 dropout_rate: float = 0.1):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_sizes: 隐藏层大小列表
            use_dueling: 是否使用Dueling架构
            dropout_rate: Dropout比率
        """
        super(DQNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        # 选择网络类型
        if use_dueling:
            self.network = DuelingDQN(state_dim, action_dim, hidden_sizes, dropout_rate)
        else:
            self.network = DQN(state_dim, action_dim, hidden_sizes, dropout_rate)
            
        # 初始化权重
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入状态张量
            
        Returns:
            Q值张量
        """
        return self.network(x)

def test_networks():
    """测试网络实现"""
    state_dim = 10
    action_dim = 4
    hidden_sizes = [64, 64]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 测试标准DQN
    print("\n测试标准DQN...")
    dqn = DQN(state_dim, action_dim, hidden_sizes).to(device)
    x = torch.randn(1, state_dim).to(device)  # 测试单样本
    y = dqn(x)
    print(f"单样本输出形状: {y.shape}")
    
    x = torch.randn(32, state_dim).to(device)  # 测试批量样本
    y = dqn(x)
    print(f"批量样本输出形状: {y.shape}")
    
    # 测试Dueling DQN
    print("\n测试Dueling DQN...")
    dueling_dqn = DuelingDQN(state_dim, action_dim, hidden_sizes).to(device)
    x = torch.randn(1, state_dim).to(device)  # 测试单样本
    y = dueling_dqn(x)
    print(f"单样本输出形状: {y.shape}")
    
    x = torch.randn(32, state_dim).to(device)  # 测试批量样本
    y = dueling_dqn(x)
    print(f"批量样本输出形状: {y.shape}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_networks()
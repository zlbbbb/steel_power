"""
Neural Network Models
Current Date and Time (UTC): 2025-02-26 08:46:50
Current User: zlbbbb
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DQN(nn.Module):
    """基础DQN网络"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        """
        初始化DQN网络
        
        Args:
            input_size: 输入维度
            output_size: 输出维度（动作数量）
            hidden_sizes: 隐藏层大小列表
        """
        super().__init__()
        
        # 构建网络层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Q值张量
        """
        return self.network(x)


class DuelingDQN(nn.Module):
    """双重DQN网络"""
    
    def __init__(self, input_size: int, output_size: int, hidden_sizes: List[int]):
        """
        初始化双重DQN网络
        
        Args:
            input_size: 输入维度
            output_size: 输出维度（动作数量）
            hidden_sizes: 隐藏层大小列表
        """
        super().__init__()
        
        # 特征提取层
        self.feature_network = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU()
        )
        
        # 价值流
        value_sizes = hidden_sizes[1:]
        value_layers = []
        prev_size = hidden_sizes[0]
        
        for hidden_size in value_sizes:
            value_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        value_layers.append(nn.Linear(prev_size, 1))
        self.value_network = nn.Sequential(*value_layers)
        
        # 优势流
        advantage_layers = []
        prev_size = hidden_sizes[0]
        
        for hidden_size in value_sizes:
            advantage_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU()
            ])
            prev_size = hidden_size
            
        advantage_layers.append(nn.Linear(prev_size, output_size))
        self.advantage_network = nn.Sequential(*advantage_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            
        Returns:
            Q值张量
        """
        features = self.feature_network(x)
        
        value = self.value_network(features)
        advantage = self.advantage_network(features)
        
        # 合并价值和优势
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


if __name__ == "__main__":
    # 测试网络
    input_size = 24
    output_size = 100
    hidden_sizes = [256, 128, 64]
    batch_size = 32
    
    # 测试基础DQN
    dqn = DQN(input_size, output_size, hidden_sizes)
    test_input = torch.randn(batch_size, input_size)
    output = dqn(test_input)
    assert output.shape == (batch_size, output_size)
    print("基础DQN测试通过")
    
    # 测试双重DQN
    dueling_dqn = DuelingDQN(input_size, output_size, hidden_sizes)
    output = dueling_dqn(test_input)
    assert output.shape == (batch_size, output_size)
    print("双重DQN测试通过")
"""
DQN Agent for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 10:13:43
Current User: zlbbbb
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Tuple, List, Union, Optional
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# 使用绝对导入
from src.models.networks import DQNetwork

class DQNAgent:
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_sizes: List[int],
                 learning_rate: float,
                 gamma: float,
                 epsilon_start: float,
                 epsilon_end: float,
                 epsilon_decay: float,
                 memory_size: int,
                 batch_size: int,
                 target_update: int,
                 use_double: bool = True,
                 use_dueling: bool = True,
                 device: torch.device = None):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_sizes: 隐藏层大小列表
            learning_rate: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最小探索率
            epsilon_decay: 探索率衰减
            memory_size: 经验回放池大小
            batch_size: 批次大小
            target_update: 目标网络更新频率
            use_double: 是否使用Double DQN
            use_dueling: 是否使用Dueling DQN
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.use_double = use_double
        self.use_dueling = use_dueling
        self.memory = deque(maxlen=memory_size)
        self.learn_step_counter = 0
        
        # 设置设备
        self.device = device if device is not None else torch.device('cpu')
        
        # 创建策略网络和目标网络
        self.policy_net = DQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            use_dueling=use_dueling
        ).to(self.device)
        
        self.target_net = DQNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            use_dueling=use_dueling
        ).to(self.device)
        
        # 初始化目标网络
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        
    def store_transition(self, 
                        state: Union[np.ndarray, torch.Tensor],
                        action: int,
                        reward: float,
                        next_state: Union[np.ndarray, torch.Tensor],
                        done: bool):
        """存储经验到回放池"""
        # 转换为NumPy数组
        if isinstance(state, torch.Tensor):
            state = state.cpu().detach().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().detach().numpy()
            
        self.memory.append((state, action, reward, next_state, done))
        
    def select_action(self, 
                    state: Union[np.ndarray, torch.Tensor],
                    evaluate: bool = False) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            evaluate: 是否处于评估模式
            
        Returns:
            选择的动作
        """
        if not evaluate and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
            
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
                
            # 在评估模式下切换网络状态
            if evaluate:
                self.policy_net.eval_mode()
            
            q_values = self.policy_net(state)
            
            # 如果是在评估模式下，恢复训练模式
            if evaluate:
                self.policy_net.train_mode()
                
            return q_values.max(1)[1].item()
                
    def learn(self) -> Optional[float]:
        """从经验中学习"""
        if len(self.memory) < self.batch_size:
            return None
            
        # 采样批次
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # 转换为张量
        states = torch.FloatTensor(np.array(batch[0])).to(self.device)
        actions = torch.LongTensor(batch[1]).to(self.device)
        rewards = torch.FloatTensor(batch[2]).to(self.device)
        next_states = torch.FloatTensor(np.array(batch[3])).to(self.device)
        dones = torch.FloatTensor(batch[4]).to(self.device)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值
        with torch.no_grad():
            if self.use_double:
                # Double DQN
                next_actions = self.policy_net(next_states).max(1)[1]
                next_q_values = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                # 普通DQN
                next_q_values = self.target_net(next_states).max(1)[0]
                
        # 计算期望的Q值
        expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # 计算损失并优化
        loss = self.criterion(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # 更新目标网络和探索率
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - self.epsilon_decay
        )
        
        return loss.item()
        
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'learn_step_counter': self.learn_step_counter,
            'use_double': self.use_double,
            'use_dueling': self.use_dueling,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }, path)
        self.logger.info(f"模型已保存至: {path}")
        
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.learn_step_counter = checkpoint['learn_step_counter']
        self.use_double = checkpoint['use_double']
        self.use_dueling = checkpoint['use_dueling']
        self.state_dim = checkpoint['state_dim']
        self.action_dim = checkpoint['action_dim']
        self.logger.info(f"模型已从 {path} 加载")

def test_agent():
    """测试DQNAgent的功能"""
    print("开始测试DQNAgent...")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 测试参数
    state_dim = 10
    action_dim = 4
    hidden_sizes = [64, 64]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    try:
        # 创建智能体
        agent = DQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=hidden_sizes,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=1000,
            batch_size=32,
            target_update=10,
            device=device
        )
        print("智能体创建成功")
        
        # 测试动作选择
        state = torch.randn(state_dim)
        action = agent.select_action(state)
        print(f"选择的动作: {action}")
        
        # 测试经验存储和学习
        print("\n测试经验存储...")
        for i in range(100):
            state = torch.randn(state_dim)
            action = agent.select_action(state)
            next_state = torch.randn(state_dim)
            reward = random.random()
            done = random.choice([True, False])
            
            agent.store_transition(state, action, reward, next_state, done)
        print(f"已存储 {len(agent.memory)} 条经验")
        
        # 测试学习过程
        print("\n测试学习过程...")
        loss = agent.learn()
        if loss is not None:
            print(f"学习损失: {loss:.6f}")
        else:
            print("未进行学习（经验池样本不足）")
        
        # 测试模型保存和加载
        print("\n测试模型保存和加载...")
        save_path = "test_agent.pth"
        
        # 保存模型
        agent.save(save_path)
        print(f"模型已保存到: {save_path}")
        
        # 加载模型
        agent.load(save_path)
        print("模型加载成功")
        
        # 清理测试文件
        os.remove(save_path)
        print("测试文件已清理")
        
        print("\n性能测试...")
        # 测试推理速度
        start_time = time.time()
        num_inferences = 1000
        
        with torch.no_grad():
            for _ in range(num_inferences):
                state = torch.randn(state_dim).to(device)
                _ = agent.select_action(state, evaluate=True)
                
        end_time = time.time()
        avg_time = (end_time - start_time) / num_inferences * 1000  # 转换为毫秒
        print(f"平均推理时间: {avg_time:.3f} ms/样本")
        
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    print("\n所有测试完成！")
    return True

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    
    # 导入time模块（用于性能测试）
    import time
    
    # 运行测试
    test_agent()
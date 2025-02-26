"""
Steel Power Prediction Environment
Current Date and Time (UTC): 2025-02-26 08:11:26
Current User: zlbbbb
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional
from pathlib import Path
import logging


class SteelPowerEnv(gym.Env):
    """
    钢铁厂电力预测环境
    
    实现了标准的gym环境接口，用于DQN训练
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        Args:
            config: 环境配置字典，包含环境参数和奖励设置
        """
        super().__init__()
        
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 环境参数
        self.state_size = config['environment']['state_size']
        self.action_size = config['environment']['action_size']
        self.max_power = config['environment']['max_power']
        self.min_power = config['environment']['min_power']
        self.reward_scale = config['environment']['reward_scale']
        self.reward_shaping = config['environment']['reward_shaping']
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_size,),
            dtype=np.float32
        )
        
        # 定义动作空间（离散）
        self.action_space = spaces.Discrete(self.action_size)
        
        # 动作映射到实际调整值范围 [-10%, +10%]
        self.action_mapping = np.linspace(
            -0.1,
            0.1,
            self.action_size
        )
        
        # 环境状态
        self.current_step = 0
        self.episode_data = None
        self.episode_history = []
        self.current_power = None
        
    def _load_episode_data(self) -> None:
        """
        加载episode数据
        
        从特征工程模块获取处理后的数据
        """
        try:
            # 这里应该从数据加载器获取数据
            # 临时使用随机数据进行测试
            self.episode_data = {
                'power': np.random.normal(5000, 1000, size=1000),
                'features': np.random.randn(1000, self.state_size - 1)
            }
            self.logger.debug("成功加载episode数据")
        except Exception as e:
            self.logger.error(f"加载episode数据失败: {str(e)}")
            raise
    
    def _get_state(self) -> np.ndarray:
        """
        获取当前状态
        
        Returns:
            状态向量，包含功率和其他特征
        """
        if self.episode_data is None:
            self._load_episode_data()
            
        power = self.episode_data['power'][self.current_step]
        features = self.episode_data['features'][self.current_step]
        
        # 组合电力值和其他特征
        state = np.concatenate([
            [power / self.max_power],  # 归一化电力值
            features
        ]).astype(np.float32)
        
        return state
    
    def _calculate_reward(self, 
                         action: int,
                         predicted_power: float,
                         actual_power: float) -> float:
        """
        计算奖励值
        
        Args:
            action: 选择的动作索引
            predicted_power: 预测的电力值
            actual_power: 实际电力值
            
        Returns:
            奖励值
        """
        # 计算预测误差（MAPE）
        error = abs(actual_power - predicted_power) / actual_power
        
        # 基础奖励（负的误差）
        base_reward = -error * self.reward_scale
        
        # 奖励整形
        if self.reward_shaping:
            # 对过大的调整进行惩罚
            adjustment = abs(self.action_mapping[action])
            if adjustment > 0.05:  # 调整超过5%
                base_reward *= 0.8
            
            # 奖励准确预测
            if error < 0.05:  # 误差小于5%
                base_reward += self.config['environment']['success_reward']
            elif error > 0.2:  # 误差大于20%
                base_reward += self.config['environment']['done_penalty']
            
            # 步数惩罚
            base_reward += self.config['environment']['step_penalty']
        
        return float(base_reward)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        环境步进
        
        Args:
            action: 动作索引
            
        Returns:
            (下一状态，奖励，是否结束，是否截断，信息字典)
        """
        # 获取当前电力值
        current_power = self.episode_data['power'][self.current_step]
        
        # 应用动作获取预测值
        adjustment = self.action_mapping[action]
        predicted_power = current_power * (1 + adjustment)
        
        # 确保预测值在合理范围内
        predicted_power = np.clip(
            predicted_power,
            self.min_power,
            self.max_power
        )
        
        # 获取实际下一时刻电力值
        next_power = self.episode_data['power'][self.current_step + 1]
        
        # 计算奖励
        reward = self._calculate_reward(action, predicted_power, next_power)
        
        # 更新步数
        self.current_step += 1
        
        # 检查是否结束
        done = (self.current_step >= len(self.episode_data['power']) - 1) or \
               (abs(predicted_power - next_power) / next_power > 0.5)  # 预测误差过大
        
        # 获取下一个状态
        next_state = self._get_state()
        
        # 记录历史
        self.episode_history.append({
            'step': self.current_step,
            'action': action,
            'adjustment': adjustment,
            'predicted_power': predicted_power,
            'actual_power': next_power,
            'reward': reward
        })
        
        info = {
            'predicted_value': predicted_power,
            'actual_value': next_power,
            'error': abs(next_power - predicted_power),
            'relative_error': abs(next_power - predicted_power) / next_power,
            'step': self.current_step
        }
        
        return next_state, reward, done, False, info
    
    def reset(self, 
              seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 重置选项
            
        Returns:
            (初始状态，信息字典)
        """
        super().reset(seed=seed)
        
        # 重置环境状态
        self.current_step = 0
        self.episode_history = []
        
        # 加载新的episode数据
        self._load_episode_data()
        
        # 获取初始状态
        initial_state = self._get_state()
        
        info = {
            'episode_start': True,
            'step': self.current_step
        }
        
        return initial_state, info
    
    def render(self) -> None:
        """
        渲染环境（可选实现）
        """
        pass
    
    def close(self) -> None:
        """
        关闭环境，清理资源
        """
        pass
    
    def get_episode_history(self) -> pd.DataFrame:
        """
        获取episode历史记录
        
        Returns:
            包含episode历史数据的DataFrame
        """
        return pd.DataFrame(self.episode_history)
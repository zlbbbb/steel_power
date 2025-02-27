"""
Steel Power Environment
Current Date and Time (UTC): 2025-02-26 15:26:52
Current User: zlbbbb
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到 PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional

class SteelPowerEnv(gym.Env):
    """钢铁厂电力预测环境"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境
        
        Args:
            config: 环境配置字典，包含以下字段：
                - state_dim: 状态空间维度
                - action_dim: 动作空间维度
                - power: 电力配置
                - rewards: 奖励配置
        """
        super().__init__()
        
        # 处理配置字典的层级结构
        if isinstance(config.get('state_dim'), int):
            # 直接使用顶层键
            self.state_dim = config['state_dim']
            self.action_dim = config['action_dim']
            power_config = config.get('power', {})
            reward_config = config.get('rewards', {})
        else:
            # 配置可能在 'environment' 键下
            env_config = config.get('environment', config)
            self.state_dim = env_config['state_dim']
            self.action_dim = env_config['action_dim']
            power_config = env_config.get('power', {})
            reward_config = env_config.get('rewards', {})
        
        # 电力范围配置
        self.max_power = power_config.get('max', 1000.0)
        self.min_power = power_config.get('min', 0.0)
        self.power_scale = power_config.get('scaling_factor', 0.001)
        
        # 奖励配置
        self.reward_scale = reward_config.get('scale', 1.0)
        self.reward_shaping = reward_config.get('shaping', True)
        self.done_penalty = reward_config.get('done_penalty', -100.0)
        self.success_reward = reward_config.get('success_reward', 100.0)
        self.step_penalty = reward_config.get('step_penalty', -1.0)
        self.error_threshold = reward_config.get('error_threshold', 0.2)
        
        # 定义动作空间（离散选择不同的电力值）
        self.action_space = spaces.Discrete(self.action_dim)
        
        # 定义状态空间（连续值，包含历史数据和其他特征）
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # 初始化状态
        self.state = None
        self.current_step = 0
        self.target_power = None

        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一个环境步骤
        
        Args:
            action: 选择的动作（电力值索引）
            
        Returns:
            tuple: (下一个状态, 奖励值, 是否结束, 是否截断, 信息字典)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # 将动作转换为实际电力值
        power_value = self._action_to_power(action)
        
        # 计算与目标的误差
        error = abs(power_value - self.target_power)
        
        # 计算奖励
        reward = self._compute_reward(error)
        
        # 更新状态
        self.state = self._update_state(power_value)
        self.current_step += 1
        
        # 检查是否结束
        done = self._check_done(error)
        
        # 准备信息字典
        info = {
            'power_value': power_value,
            'target_power': self.target_power,
            'error': error,
            'step': self.current_step
        }
        
        return self.state, reward, done, False, info
        
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            
        Returns:
            tuple: (初始状态, 信息字典)
        """
        super().reset(seed=seed)
        
        # 生成新的目标电力值
        self.target_power = self._generate_target_power()
        
        # 初始化状态（包含历史数据和当前目标）
        self.state = self._initialize_state()
        self.current_step = 0
        
        return self.state, {'target_power': self.target_power}
        
    def _action_to_power(self, action: int) -> float:
        """将离散动作转换为实际电力值"""
        power_range = self.max_power - self.min_power
        power_step = power_range / (self.action_dim - 1)
        return self.min_power + action * power_step
        
    def _compute_reward(self, error: float) -> float:
        """计算奖励值"""
        # 基础奖励（根据误差）
        if error < self.error_threshold * self.max_power:
            reward = self.success_reward * (1 - error / (self.error_threshold * self.max_power))
        else:
            reward = self.done_penalty * (error / self.max_power)
            
        # 每步惩罚
        reward += self.step_penalty
        
        # 应用奖励缩放
        reward *= self.reward_scale
        
        return reward
        
    def _check_done(self, error: float) -> bool:
        """检查是否结束"""
        # 达到最大步数或误差过大时结束
        if self.current_step >= self.state_dim:
            return True
        if error > self.max_power * 0.5:  # 误差超过50%时结束
            return True
        return False
        
    def _initialize_state(self) -> np.ndarray:
        """初始化状态"""
        # 创建初始状态（可以包含历史数据和其他特征）
        state = np.zeros(self.state_dim, dtype=np.float32)
        # 在这里可以添加一些初始的历史数据或特征
        return state
        
    def _update_state(self, power_value: float) -> np.ndarray:
        """更新状态"""
        # 更新状态（例如，移动历史窗口并添加新值）
        new_state = np.roll(self.state, -1)
        new_state[-1] = power_value
        return new_state
        
    def _generate_target_power(self) -> float:
        """生成目标电力值"""
        # 生成一个随机的目标电力值
        return np.random.uniform(self.min_power, self.max_power)
        
    def render(self):
        """渲染环境（可选）"""
        pass
        
    def close(self):
        """关闭环境（可选）"""
        pass

if __name__ == "__main__":
    # 测试配置
    test_config = {
        'state_dim': 24,
        'action_dim': 100,
        'power': {
            'max': 1000.0,
            'min': 0.0,
            'scaling_factor': 0.001
        },
        'rewards': {
            'scale': 1.0,
            'shaping': True,
            'done_penalty': -100.0,
            'success_reward': 100.0,
            'step_penalty': -1.0,
            'error_threshold': 0.2
        }
    }
    
    # 创建并测试环境
    env = SteelPowerEnv(test_config)
    print("Environment created successfully!")
    
    # 测试重置
    state, info = env.reset()
    print(f"\nInitial state shape: {state.shape}")
    print(f"Target power: {info['target_power']:.2f}")
    
    # 测试步骤
    action = env.action_space.sample()
    next_state, reward, done, truncated, info = env.step(action)
    print(f"\nStep test:")
    print(f"Action taken: {action}")
    print(f"Reward received: {reward:.2f}")
    print(f"Power value: {info['power_value']:.2f}")
    print(f"Error: {info['error']:.2f}")
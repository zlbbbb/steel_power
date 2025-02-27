"""
Training Script for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 09:58:37
Current User: zlbbbb
"""

import os
import sys
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from typing import Dict, Any, Tuple, List
import json

# 添加项目根目录到 PYTHONPATH
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 本地导入
from src.models.environment import SteelPowerEnv
from src.models.agent import DQNAgent
from src.utils.logger import setup_logger
from src.utils.visualizer import TrainingVisualizer
from src.utils.analysis import ModelAnalyzer

class Trainer:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config (Dict[str, Any]): 训练配置
        """
        self.config = config
        
        # 设置日志系统（需要最先初始化）
        self.setup_logging()
        
        # GPU 设备选择和配置
        self._setup_device()
        
        # 设置随机种子
        self.set_seed(self.config['training']['seed'])
        
        # 初始化训练历史记录
        self._init_training_history()
        
        # 设置训练组件
        self.setup_components()

        # 初始化可视化器和分析器
        self.visualizer = TrainingVisualizer(self.log_dir, self.config)
        self.analyzer = ModelAnalyzer(self.log_dir)
        
        # 保存初始配置
        self.save_config()
        
    def _setup_device(self):
        """设置训练设备"""
        if torch.cuda.is_available() and self.config['training']['device'] == 'cuda':
            # 获取CUDA设备索引
            cuda_device_index = self.config['training'].get('cuda_device', 0)
            
            # 确保设备索引是整数
            if not isinstance(cuda_device_index, int):
                self.logger.warning(f"CUDA设备索引 '{cuda_device_index}' 不是整数，使用默认值 0")
                cuda_device_index = 0
            
            # 检查设备索引是否有效
            if cuda_device_index >= torch.cuda.device_count():
                self.logger.warning(
                    f"指定的CUDA设备索引 {cuda_device_index} 超出可用范围 "
                    f"(0 to {torch.cuda.device_count()-1})，使用设备 0"
                )
                cuda_device_index = 0
            
            # 设置CUDA设备
            torch.cuda.set_device(cuda_device_index)
            self.device = torch.device(f'cuda:{cuda_device_index}')
            
            # 启用 cudnn benchmark 模式以优化性能
            torch.backends.cudnn.benchmark = True
            
            # 设置多GPU支持
            self.multi_gpu = (torch.cuda.device_count() > 1 and 
                            self.config['training'].get('use_multi_gpu', False))
            
            if self.multi_gpu:
                self.logger.info(f"使用 {torch.cuda.device_count()} 个GPU")
                self.logger.info(f"主GPU设备: {torch.cuda.get_device_name(cuda_device_index)}")
            else:
                self.logger.info(f"使用单个GPU: {torch.cuda.get_device_name(cuda_device_index)}")
        else:
            self.device = torch.device('cpu')
            self.multi_gpu = False
            self.logger.info("使用CPU进行训练")
        
        # 记录设备信息
        if torch.cuda.is_available():
            self.logger.info(f"当前CUDA版本: {torch.version.cuda}")
            self.logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
            self.logger.info(f"当前GPU设备: {self.device}")
            self.logger.info(f"GPU名称: {torch.cuda.get_device_name(self.device)}")
            self.logger.info(f"CUDNN是否可用: {torch.backends.cudnn.is_available()}")
            self.logger.info(f"CUDNN版本: {torch.backends.cudnn.version()}")
    
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
        self.logger = setup_logger(
            f"train_{self.timestamp}",
            level=log_config['level'],
            log_file=self.log_dir / "training.log"
        )
        
    def set_seed(self, seed: int):
        """
        设置随机种子
        
        Args:
            seed: 随机种子值
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _init_training_history(self):
        """初始化训练历史记录"""
        self.training_history = {
            'losses': [],               # 训练损失
            'rewards': [],              # 每轮回报
            'predictions': [],          # 模型预测值
            'actual_values': [],        # 实际值
            'episodes_time': [],        # 每轮训练时间
            'gpu_memory': [],           # GPU内存使用
            'learning_rates': [],       # 学习率变化
            'epsilon_values': [],       # epsilon值变化
            'prediction_errors': [],    # 预测误差
            'avg_losses': [],           # 平均损失
            'steps_per_episode': [],    # 每轮步数
            'validation_metrics': [],   # 验证指标
            'training_speed': [],       # 训练速度（步/秒）
            'gpu_utilization': []       # GPU利用率
        }
        
    def setup_components(self):
        """设置训练组件"""
        # 创建智能体
        self.agent = DQNAgent(
            state_dim=self.config['environment']['state_dim'],
            action_dim=self.config['environment']['action_dim'],
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
        
        # 如果使用多 GPU，将模型转换为 DataParallel
        if self.multi_gpu:
            self.agent.policy_net = torch.nn.DataParallel(self.agent.policy_net)
            self.agent.target_net = torch.nn.DataParallel(self.agent.target_net)
        self.logger.info("智能体创建成功")
        
    def save_config(self):
        """保存配置到文件"""
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"配置已保存至: {config_path}")
        
    def _update_training_history(self, metrics: Dict[str, float]):
        """
        更新训练历史记录
        
        Args:
            metrics: 包含当前训练指标的字典
        """
        # 更新基本指标
        if 'loss' in metrics:
            self.training_history['losses'].append(metrics['loss'])
        if 'episode_reward' in metrics:
            self.training_history['rewards'].append(metrics['episode_reward'])
        if 'epsilon' in metrics:
            self.training_history['epsilon_values'].append(metrics['epsilon'])
        
        # 更新预测相关指标
        if 'predicted_power' in metrics and 'actual_power' in metrics:
            self.training_history['predictions'].append(metrics['predicted_power'])
            self.training_history['actual_values'].append(metrics['actual_power'])
            if 'prediction_error' in metrics:
                self.training_history['prediction_errors'].append(metrics['prediction_error'])
        
        # 更新性能指标
        if 'gpu_memory_allocated' in metrics:
            self.training_history['gpu_memory'].append({
                'allocated': metrics['gpu_memory_allocated'],
                'reserved': metrics['gpu_memory_reserved']
            })
        
        if 'training_speed' in metrics:
            self.training_history['training_speed'].append(metrics['training_speed'])
        
        if 'steps' in metrics:
            self.training_history['steps_per_episode'].append(metrics['steps'])
            
    def train(self, env: SteelPowerEnv):
        """
        执行训练过程
        
        Args:
            env: 训练环境实例
        """
        self.logger.info("开始训练...")
        self.training_start_time = datetime.now()
        
        total_episodes = self.config['training']['epochs']
        max_steps = self.config['training']['max_steps']
        best_reward = float('-inf')
        
        for episode in range(total_episodes):
            episode_start_time = datetime.now()
            
            # CUDA 事件用于性能分析
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
            
            # 执行训练回合
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = 0
            n_steps = 0
            
            # 确保状态tensor在正确的设备上
            if isinstance(state, torch.Tensor):
                state = state.to(self.device)
            else:
                state = torch.FloatTensor(state).to(self.device)
            
            for step in range(max_steps):
                # 训练步骤
                action = self.agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                
                # 确保所有张量都在GPU上
                if isinstance(next_state, torch.Tensor):
                    next_state = next_state.to(self.device)
                else:
                    next_state = torch.FloatTensor(next_state).to(self.device)
                
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
                
                if done:
                    break
                    
            # 计算性能指标
            episode_time = (datetime.now() - episode_start_time).total_seconds()
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                step_time = start_event.elapsed_time(end_event) / 1000  # 转换为秒
            else:
                step_time = episode_time
                
            # 计算平均损失和其他指标
            avg_loss = episode_loss / n_steps if n_steps > 0 else 0
            training_speed = n_steps / step_time if step_time > 0 else 0

            # 更新训练指标
            metrics = {
                'episode_reward': episode_reward,
                'loss': avg_loss,
                'epsilon': self.agent.epsilon,
                'steps': n_steps,
                'training_speed': training_speed,
                'episode_time': episode_time
            }
            
            # 添加GPU指标
            if torch.cuda.is_available():
                metrics.update({
                    'gpu_memory_allocated': torch.cuda.memory_allocated() / 1024**2,
                    'gpu_memory_reserved': torch.cuda.memory_reserved() / 1024**2
                })
            
            # 更新训练历史
            self._update_training_history(metrics)
                
            # 日志记录和可视化
            if episode % self.config['training']['logging']['log_frequency'] == 0:
                self.logger.info(
                    f"Episode: {episode}/{total_episodes}, "
                    f"Reward: {episode_reward:.2f}, "
                    f"Loss: {avg_loss:.6f}, "
                    f"Epsilon: {self.agent.epsilon:.4f}, "
                    f"Speed: {training_speed:.2f} steps/s"
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
                
        # 训练结束，保存最终可视化结果和训练历史
        self._save_training_history()
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
        
    def _save_training_history(self):
        """保存训练历史记录"""
        history_path = self.log_dir / 'training_history.json'
        
        # 转换numpy数组为列表以便JSON序列化
        serializable_history = {}
        for key, value in self.training_history.items():
            if isinstance(value, (np.ndarray, np.generic)):
                serializable_history[key] = value.tolist()
            elif isinstance(value, list):
                # 处理列表中的numpy数组或其他特殊类型
                processed_list = []
                for item in value:
                    if isinstance(item, (np.ndarray, np.generic)):
                        processed_list.append(item.tolist())
                    elif isinstance(item, dict):
                        # 处理字典中的numpy数组
                        processed_dict = {}
                        for k, v in item.items():
                            if isinstance(v, (np.ndarray, np.generic)):
                                processed_dict[k] = v.tolist()
                            else:
                                processed_dict[k] = v
                        processed_list.append(processed_dict)
                    else:
                        processed_list.append(item)
                serializable_history[key] = processed_list
            elif isinstance(value, dict):
                # 处理字典类型的数据
                processed_dict = {}
                for k, v in value.items():
                    if isinstance(v, (np.ndarray, np.generic)):
                        processed_dict[k] = v.tolist()
                    else:
                        processed_dict[k] = v
                serializable_history[key] = processed_dict
            else:
                serializable_history[key] = value
        
        # 添加元数据
        serializable_history['metadata'] = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': 'zlbbbb',
            'total_episodes': self.config['training']['epochs'],
            'max_steps_per_episode': self.config['training']['max_steps'],
            'device': str(self.device),
            'multi_gpu': self.multi_gpu
        }
        
        # 保存到文件
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_history, f, indent=4)
            
        self.logger.info(f"训练历史已保存至: {history_path}")
        
    def analyze_training(self):
        """分析训练结果"""
        if not hasattr(self, 'analyzer'):
            self.analyzer = ModelAnalyzer(self.log_dir)
        
        # 收集训练历史数据
        training_history = self.training_history
        
        # 计算统计指标
        stats = {
            'total_episodes': len(training_history['rewards']),
            'best_reward': max(training_history['rewards']) if training_history['rewards'] else None,
            'average_reward': np.mean(training_history['rewards']) if training_history['rewards'] else None,
            'average_loss': np.mean(training_history['losses']) if training_history['losses'] else None,
            'training_time': (datetime.now() - self.training_start_time).total_seconds(),
            'average_steps_per_episode': np.mean(training_history['steps_per_episode']) if training_history['steps_per_episode'] else None,
            'final_epsilon': self.agent.epsilon
        }
        
        # 生成分析报告
        report = self.analyzer.generate_analysis_report(
            training_history=training_history,
            stats=stats,
            config=self.config
        )
        
        # 保存分析报告
        report_path = self.log_dir / 'training_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
            
        self.logger.info(f"训练分析报告已保存至: {report_path}")
        return report
    
    def evaluate(self, env: SteelPowerEnv, num_episodes: int = None) -> Dict[str, float]:
        """
        评估模型性能
        
        Args:
            env: 评估环境
            num_episodes: 评估回合数，如果为None则使用配置文件中的值
            
        Returns:
            包含评估指标的字典
        """
        if num_episodes is None:
            num_episodes = self.config['evaluation']['num_episodes']
            
        self.logger.info(f"开始评估模型，评估回合数: {num_episodes}")
        self.agent.eval()  # 将模型设置为评估模式
        
        metrics = {
            'total_reward': 0,
            'episode_rewards': [],
            'predictions': [],
            'actual_values': [],
            'prediction_errors': [],
            'steps': []
        }
        
        with torch.no_grad():
            for episode in range(num_episodes):
                state, _ = env.reset()
                episode_reward = 0
                episode_steps = 0
                
                while True:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action = self.agent.select_action(state_tensor, evaluate=True)
                    next_state, reward, done, _, info = env.step(action)
                    
                    episode_reward += reward
                    episode_steps += 1
                    
                    if 'power_value' in info and 'target_power' in info:
                        metrics['predictions'].append(info['power_value'])
                        metrics['actual_values'].append(info['target_power'])
                        metrics['prediction_errors'].append(
                            abs(info['power_value'] - info['target_power'])
                        )
                    
                    state = next_state
                    if done:
                        break
                
                metrics['total_reward'] += episode_reward
                metrics['episode_rewards'].append(episode_reward)
                metrics['steps'].append(episode_steps)
                
                if (episode + 1) % 10 == 0:
                    self.logger.info(f"评估进度: {episode + 1}/{num_episodes}, "
                                   f"平均奖励: {metrics['total_reward']/(episode + 1):.2f}")
        
        # 计算统计指标
        metrics['mean_reward'] = metrics['total_reward'] / num_episodes
        metrics['std_reward'] = np.std(metrics['episode_rewards'])
        metrics['mean_steps'] = np.mean(metrics['steps'])
        metrics['mean_error'] = np.mean(metrics['prediction_errors'])
        metrics['rmse'] = np.sqrt(np.mean(np.square(metrics['prediction_errors'])))
        
        self.logger.info(f"评估完成 - 平均奖励: {metrics['mean_reward']:.2f}, "
                        f"RMSE: {metrics['rmse']:.4f}")
        
        # 将评估结果添加到训练历史
        self.training_history['evaluation_metrics'] = metrics
        
        return metrics

if __name__ == "__main__":
    try:
        # 加载配置文件
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 创建训练器
        trainer = Trainer(config)
        
        # 创建环境
        env_config = config['environment']
        env = SteelPowerEnv(env_config)
        
        # 训练模型
        trainer.train(env)
        
        # 分析训练结果
        analysis_report = trainer.analyze_training()
        print("\n训练分析报告:")
        print(analysis_report)
        
        # 评估模型
        eval_metrics = trainer.evaluate(env)
        print("\n评估结果:")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key}: {value:.4f}")
            
    except Exception as e:
        print(f"错误: {str(e)}")
        raise
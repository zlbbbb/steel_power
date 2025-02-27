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
        
        # 设置训练组件
        self.setup_components()

        # 初始化可视化器
        self.visualizer = TrainingVisualizer(self.log_dir, self.config)
        self.analyzer = ModelAnalyzer(self.log_dir)
        
        # 保存初始配置
        self.save_config()

    def _setup_device(self):
        """设置训练设备"""
        if torch.cuda.is_available() and self.config['training']['device'] == 'cuda':
            self.device = torch.device('cuda')
            # 设置 CUDA 设备
            if 'cuda_device' in self.config['training']:
                cuda_device = self.config['training']['cuda_device']
                torch.cuda.set_device(cuda_device)
                self.logger.info(f"Using CUDA Device: {cuda_device}")
            # 启用 cudnn benchmark 模式以优化性能
            torch.backends.cudnn.benchmark = True
            
            # 设置多 GPU 支持
            self.multi_gpu = (torch.cuda.device_count() > 1 and 
                            self.config['training'].get('use_multi_gpu', False))
            if self.multi_gpu:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUs")
        else:
            self.device = torch.device('cpu')
            self.multi_gpu = False
            
        self.logger.info(f"Using device: {self.device}")

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
        # 如果使用多 GPU，将模型转换为 DataParallel
        if self.multi_gpu:
            self.agent.policy_net = torch.nn.DataParallel(self.agent.policy_net)
            self.agent.target_net = torch.nn.DataParallel(self.agent.target_net)
        self.logger.info("智能体创建成功")

    def _init_training_history(self):
        """初始化训练历史记录"""
        self.training_history = {
            'losses': [],
            'rewards': [],
            'predictions': [],
            'actual_values': [],
            'episodes_time': [],
            'gpu_memory': [],
            'learning_rate': [],
            'epsilon_values': []
        }
        
    def save_config(self):
        """保存配置到文件"""
        config_path = self.log_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"配置已保存至: {config_path}")
    def analyze_training(self):
        """分析训练结果"""
        analyzer = ModelAnalyzer(self.log_dir)
        
        # 收集训练历史数据
        training_history = {
            'losses': self.training_losses,
            'rewards': self.episode_rewards,
            'predictions': self.predictions,
            'actual_values': self.actual_values
        }
        
        # 绘制训练历史
        analyzer.plot_training_history(
            losses=training_history['losses'],
            rewards=training_history['rewards']
        )
        
        # 分析预测误差
        error_stats = analyzer.analyze_prediction_errors(
            predictions=np.array(training_history['predictions']),
            actual_values=np.array(training_history['actual_values'])
        )
        
        # 生成分析报告
        training_time = (datetime.now() - self.training_start_time).total_seconds()
        model_params = sum(p.numel() for p in self.agent.policy_net.parameters())
        
        report = analyzer.generate_analysis_report(
            error_stats=error_stats,
            training_time=training_time,
            model_params=model_params
        )
        
        self.logger.info("模型分析完成，报告已生成")
        self.logger.info("\n" + report)
            
    def train(self):
        """执行训练过程"""
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
            metrics = self._train_episode(episode, max_steps)
            
            # 记录 GPU 性能指标
            if torch.cuda.is_available():
                end_event.record()
                torch.cuda.synchronize()
                step_time = start_event.elapsed_time(end_event)
                metrics['gpu_time'] = step_time
                metrics['gpu_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
                metrics['gpu_memory_cached'] = torch.cuda.memory_cached() / 1024**2
            
            # 更新训练历史
            self._update_training_history(metrics)
            
            # 记录和可视化
            self._log_training_progress(episode, total_episodes, metrics)
            
            # 保存模型
            if metrics['episode_reward'] > best_reward:
                best_reward = metrics['episode_reward']
                self.save_model('best')
            
            if episode % self.config['training']['checkpoint']['save_frequency'] == 0:
                self.save_model(f"episode_{episode}")
                
            # 分析训练进度
            if episode % self.config['training']['analysis_frequency'] == 0:
                self._analyze_training_progress()
        
        # 训练结束，生成最终分析报告
        self._generate_final_report()
        self.logger.info("训练完成！")

    def _train_episode(self, episode: int, max_steps: int) -> Dict[str, float]:
        """执行单个训练回合"""
        state, _ = self.env.reset()
        episode_reward = 0
        episode_loss = 0
        n_steps = 0
        predictions = []
        actual_values = []
        
        # 确保状态tensor在正确的设备上
        state = self._to_device(state)
        
        for step in range(max_steps):
            # 选择动作
            action = self.agent.select_action(state)
            next_state, reward, done, truncated, info = self.env.step(action)
            next_state = self._to_device(next_state)
            
            # 存储经验和学习
            self.agent.store_transition(state, action, reward, next_state, done)
            loss = self.agent.learn()
            
            # 更新统计信息
            if loss is not None:
                episode_loss += loss
            episode_reward += reward
            n_steps += 1
            
            # 记录预测结果
            if 'power_value' in info and 'target_power' in info:
                predictions.append(info['power_value'])
                actual_values.append(info['target_power'])
            
            state = next_state
            if done:
                break
                
        # 计算回合统计指标
        metrics = {
            'episode_reward': episode_reward,
            'average_loss': episode_loss / n_steps if n_steps > 0 else 0,
            'epsilon': self.agent.epsilon,
            'steps': n_steps,
            'predictions': predictions,
            'actual_values': actual_values
        }
        
        return metrics

    def _update_training_history(self, metrics: Dict[str, Any]):
        """更新训练历史记录"""
        self.training_history['losses'].append(metrics['average_loss'])
        self.training_history['rewards'].append(metrics['episode_reward'])
        self.training_history['epsilon_values'].append(metrics['epsilon'])
        
        if 'predictions' in metrics and 'actual_values' in metrics:
            self.training_history['predictions'].extend(metrics['predictions'])
            self.training_history['actual_values'].extend(metrics['actual_values'])
        
        if 'gpu_time' in metrics:
            self.training_history['episodes_time'].append(metrics['gpu_time'])
        
        if 'gpu_memory_allocated' in metrics:
            self.training_history['gpu_memory'].append({
                'allocated': metrics['gpu_memory_allocated'],
                'cached': metrics['gpu_memory_cached']
            })
            
    def _analyze_training_progress(self):
        """分析训练进度"""
        analysis_results = self.analyzer.analyze_training_progress(
            self.training_history,
            self.config['training']['analysis_metrics']
        )
        
        # 更新学习率或其他超参数
        if analysis_results.get('needs_lr_update', False):
            self.agent.adjust_learning_rate(analysis_results['suggested_lr'])
            
        # 记录分析结果
        self.logger.info("\n" + analysis_results['summary'])
        
    def _generate_final_report(self):
        """生成最终训练报告"""
        # 计算总训练时间
        training_time = (datetime.now() - self.training_start_time).total_seconds()
        
        # 获取模型参数统计
        model_stats = {
            'total_params': sum(p.numel() for p in self.agent.policy_net.parameters()),
            'trainable_params': sum(p.numel() for p in self.agent.policy_net.parameters() if p.requires_grad)
        }
        
        # 生成完整分析报告
        final_report = self.analyzer.generate_final_report(
            self.training_history,
            training_time,
            model_stats,
            self.config
        )
        
        # 保存报告
        report_path = self.log_dir / "final_training_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
            
        self.logger.info("最终训练报告已生成：" + str(report_path))

    def _to_device(self, x):
        """将数据转移到正确的设备上"""
        if isinstance(x, torch.Tensor):
            return x.to(self.device)
        elif isinstance(x, np.ndarray):
            return torch.FloatTensor(x).to(self.device)
        return x

            
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
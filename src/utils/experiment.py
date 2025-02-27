"""
Experiment Management Module
Current Date and Time (UTC): 2025-02-27 03:15:24
Current User: zlbbbb
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger
from src.utils.metrics import MetricsCalculator
from src.utils.visualizer import TrainingVisualizer

import yaml
import json
from typing import Dict, Any, Optional
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    """
    实验管理器，整合了日志记录、指标计算、可视化等功能
    """
    
    def __init__(
        self,
        exp_name: str,
        config: Dict[str, Any],
        exp_dir: Optional[Path] = None,
        use_tensorboard: bool = True
    ):
        """
        初始化实验管理器
        
        Args:
            exp_name: 实验名称
            config: 配置字典
            exp_dir: 实验目录，默认为 'logs'
            use_tensorboard: 是否使用 TensorBoard
        """
        self.exp_name = exp_name
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 设置实验目录
        if exp_dir is None:
            exp_dir = Path("logs")
        self.exp_dir = exp_dir / f"{exp_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoints_dir = self.exp_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.plots_dir = self.exp_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # 保存配置
        self.save_config()
        
        # 设置日志记录器
        self.logger = setup_logger(
            name=exp_name,
            level=config.get('training', {}).get('logging', {}).get('level', 'INFO'),
            log_file=self.exp_dir / "experiment.log"
        )
        
        # 设置可视化器
        self.visualizer = TrainingVisualizer(
            log_dir=self.exp_dir,
            config=config
        )
        
        # 设置指标计算器
        self.metrics = MetricsCalculator()
        
        # 设置 TensorBoard
        if use_tensorboard:
            self.writer = SummaryWriter(self.exp_dir / "tensorboard")
        else:
            self.writer = None
            
        self.logger.info(f"实验 '{exp_name}' 初始化完成")
        
    def save_config(self):
        """保存配置文件"""
        config_path = self.exp_dir / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
    def save_checkpoint(
        self,
        model_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """
        保存检查点
        
        Args:
            model_state: 模型状态字典
            optimizer_state: 优化器状态字典
            epoch: 当前轮次
            metrics: 指标字典
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': optimizer_state,
            'metrics': metrics
        }
        
        # 保存最新检查点
        latest_path = self.checkpoints_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoints_dir / "best.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"保存最佳模型检查点: {best_path}")
            
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            检查点字典
        """
        checkpoint_path = self.checkpoints_dir / path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path)
        self.logger.info(f"加载检查点: {checkpoint_path}")
        return checkpoint
        
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """
        记录指标
        
        Args:
            metrics: 指标字典
            step: 当前步骤
        """
        # 更新可视化器
        self.visualizer.update_metrics(step, metrics)
        
        # 写入 TensorBoard
        if self.writer:
            for name, value in metrics.items():
                self.writer.add_scalar(f"metrics/{name}", value, step)
                
        # 记录到日志
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step} - {metrics_str}")
        
    def finish(self):
        """完成实验，清理资源"""
        if self.writer:
            self.writer.close()
        self.visualizer.close()
        self.logger.info(f"实验 '{self.exp_name}' 完成")

if __name__ == "__main__":
    # 测试代码
    try:
        # 从 config 目录加载配置文件
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        print(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        print("已成功加载配置文件")
        
        # 创建实验管理器
        exp = ExperimentManager(
            exp_name="test_experiment",
            config=config
        )
        
        print("实验管理器创建成功")
        
        # 测试记录指标（使用配置文件中定义的指标）
        metrics = {}
        for metric in config['training']['logging']['metrics']:
            if metric == 'episode_reward':
                metrics[metric] = 100
            elif metric == 'average_loss':
                metrics[metric] = 0.5
            elif metric == 'epsilon':
                metrics[metric] = 0.9
            elif metric == 'episode_length':
                metrics[metric] = 500
            elif metric == 'predicted_power':
                metrics[metric] = 1000
            elif metric == 'actual_power':
                metrics[metric] = 950
            elif metric == 'prediction_error':
                metrics[metric] = 50
        
        print("开始记录测试指标...")
        print(f"记录的指标: {list(metrics.keys())}")
        
        # 模拟多个训练步骤
        for i in range(10):
            # 模拟指标变化
            if 'average_loss' in metrics:
                metrics['average_loss'] *= 0.95  # 损失下降
            if 'epsilon' in metrics:
                metrics['epsilon'] *= 0.9  # 探索率衰减
            if 'episode_reward' in metrics:
                metrics['episode_reward'] *= 1.05  # 奖励增加
            if 'prediction_error' in metrics:
                metrics['prediction_error'] *= 0.95  # 预测误差减小
                
            exp.log_metrics(metrics, step=i)
            print(f"Step {i}: 已记录指标")
        
        print("测试指标记录完成")
        
        # 完成实验
        exp.finish()
        print("实验完成")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        print(f"当前工作目录: {Path.cwd()}")
        print(f"项目根目录: {project_root}")
        print(f"配置文件路径: {config_path}")
        print(f"配置文件是否存在: {config_path.exists() if config_path else False}")
        print(f"Python路径: {sys.path}")
        raise
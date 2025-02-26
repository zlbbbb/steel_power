"""
Logging Utility Functions
Current Date and Time (UTC): 2025-02-26 13:32:20
Current User: zlbbbb
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import yaml
import json
from .time_utils import get_current_time

class CustomFormatter(logging.Formatter):
    """自定义日志格式化器，支持彩色输出"""
    
    COLORS = {
        logging.DEBUG: '\033[0;36m',      # Cyan
        logging.INFO: '\033[0;32m',       # Green
        logging.WARNING: '\033[0;33m',    # Yellow
        logging.ERROR: '\033[0;31m',      # Red
        logging.CRITICAL: '\033[0;35m'    # Magenta
    }
    RESET = '\033[0m'
    
    def __init__(self, use_colors: bool = True):
        """
        初始化格式化器
        
        Args:
            use_colors: 是否使用彩色输出
        """
        super().__init__()
        self.use_colors = use_colors
        self.datefmt = '%Y-%m-%d %H:%M:%S'
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if self.use_colors and record.levelno in self.COLORS:
            record.levelname = f'{self.COLORS[record.levelno]}{record.levelname}{self.RESET}'
            record.msg = f'{self.COLORS[record.levelno]}{record.msg}{self.RESET}'
        
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            self.datefmt
        ).format(record)

def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别
        log_file: 日志文件路径
        use_colors: 是否使用彩色输出
        
    Returns:
        配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    # 添加控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter(use_colors))
    logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)
    
    return logger

class ExperimentLogger:
    """实验日志记录器，用于管理实验过程中的日志和配置"""
    
    def __init__(
        self,
        exp_dir: Union[str, Path],
        exp_name: str,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化实验日志记录器
        
        Args:
            exp_dir: 实验目录
            exp_name: 实验名称
            config: 配置字典
        """
        self.exp_dir = Path(exp_dir)
        self.exp_name = exp_name
        self.exp_path = self.exp_dir / exp_name
        self.exp_path.mkdir(parents=True, exist_ok=True)
        
        # 设置日志文件
        self.log_file = self.exp_path / 'experiment.log'
        self.logger = setup_logger(
            f"experiment_{exp_name}",
            log_file=self.log_file
        )
        
        # 保存实验配置
        if config:
            self.save_config(config)
        
        # 初始化指标记录
        self.metrics = {}
    
    def save_config(self, config: Dict[str, Any]):
        """保存实验配置"""
        config_path = self.exp_path / 'config.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        self.logger.info(f"配置已保存至: {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """记录训练指标"""
        # 更新指标记录
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
            self.metrics[name].append((step, value))
        
        # 保存指标到文件
        metrics_path = self.exp_path / 'metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=4)
        
        # 记录到日志
        metrics_str = ', '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"Step {step}: {metrics_str}")
    
    def log_model(self, model_path: Union[str, Path], metrics: Dict[str, float]):
        """记录模型保存信息"""
        model_info = {
            'path': str(model_path),
            'metrics': metrics,
            'timestamp': get_current_time().isoformat()
        }
        
        # 保存模型信息
        model_info_path = self.exp_path / 'model_info.json'
        with open(model_info_path, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4)
        
        self.logger.info(f"模型已保存至: {model_path}")
    
    def log_artifact(self, artifact_path: Union[str, Path], artifact_type: str):
        """记录训练过程中生成的文件"""
        artifact_path = Path(artifact_path)
        if artifact_path.exists():
            dest_path = self.exp_path / f"{artifact_type}_{artifact_path.name}"
            import shutil
            shutil.copy2(artifact_path, dest_path)
            self.logger.info(f"{artifact_type}已保存至: {dest_path}")
    
    def get_exp_dir(self) -> Path:
        """获取实验目录"""
        return self.exp_path
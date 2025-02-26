"""
Time Utility Functions
Current Date and Time (UTC): 2025-02-26 13:38:10
Current User: zlbbbb
"""

import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)

class Timer:
    """计时器类，用于记录代码执行时间"""
    
    def __init__(self, name: Optional[str] = None):
        """
        初始化计时器
        
        Args:
            name: 计时器名称，用于日志记录
        """
        self.name = name or 'Timer'
        self.start_time = None
        self.end_time = None
        self.splits = []
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.splits = []
        return self
    
    def split(self, name: str = None) -> float:
        """
        记录分段时间
        
        Args:
            name: 分段名称
            
        Returns:
            分段用时（秒）
        """
        if self.start_time is None:
            raise RuntimeError("Timer hasn't been started")
        
        current_time = time.time()
        split_time = current_time - (self.splits[-1][1] if self.splits else self.start_time)
        self.splits.append((name or f"Split_{len(self.splits)+1}", current_time))
        
        return split_time
    
    def stop(self) -> float:
        """
        停止计时
        
        Returns:
            总用时（秒）
        """
        if self.start_time is None:
            raise RuntimeError("Timer hasn't been started")
        
        self.end_time = time.time()
        return self.get_elapsed_time()
    
    def get_elapsed_time(self) -> float:
        """获取运行时间"""
        if self.start_time is None:
            return 0.0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time
    
    def get_split_times(self) -> dict:
        """获取所有分段时间"""
        split_times = {}
        for i, (name, time_stamp) in enumerate(self.splits):
            prev_time = self.start_time if i == 0 else self.splits[i-1][1]
            split_times[name] = time_stamp - prev_time
        return split_times
    
    def __enter__(self):
        """上下文管理器入口"""
        return self.start()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        total_time = self.stop()
        if self.name:
            logger.info(f"{self.name} 总用时: {format_time(total_time)}")
        if self.splits:
            for name, time_value in self.get_split_times().items():
                logger.debug(f"{name}: {format_time(time_value)}")

def get_current_time(tz: Optional[timezone] = None) -> datetime:
    """
    获取当前时间
    
    Args:
        tz: 时区，默认为UTC
        
    Returns:
        当前时间
    """
    return datetime.now(tz) if tz else datetime.now(timezone.utc)

def format_time(seconds: Union[int, float]) -> str:
    """
    格式化时间
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.2f}s"
    
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

def parse_time_str(time_str: str) -> int:
    """
    解析时间字符串为秒数
    
    Args:
        time_str: 时间字符串 (如 "1h 30m", "45m", "90s")
        
    Returns:
        总秒数
    
    Examples:
        >>> parse_time_str("1h 30m")
        5400
        >>> parse_time_str("90s")
        90
    """
    total_seconds = 0
    parts = time_str.lower().split()
    
    for part in parts:
        if part.endswith('h'):
            total_seconds += int(part[:-1]) * 3600
        elif part.endswith('m'):
            total_seconds += int(part[:-1]) * 60
        elif part.endswith('s'):
            total_seconds += float(part[:-1])
        else:
            raise ValueError(f"无效的时间格式: {part}")
    
    return int(total_seconds)

def get_time_delta(start_time: datetime, end_time: datetime = None) -> timedelta:
    """
    计算时间差
    
    Args:
        start_time: 开始时间
        end_time: 结束时间，默认为当前时间
        
    Returns:
        时间差
    """
    if end_time is None:
        end_time = get_current_time()
    return end_time - start_time
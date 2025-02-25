import os
import yaml
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def get_current_time() -> str:
        """获取当前UTC时间的格式化字符串"""
        return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def get_current_user() -> str:
        """获取当前用户名"""
        return os.getenv('USERNAME', 'zlbbbb')
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._setup_logging()
        self.load_config()
        
    def _setup_logging(self) -> None:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if not self.config_path.exists():
                self.logger.error(f"配置文件不存在: {self.config_path}")
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.logger.info("配置加载成功")
            return self.config
            
        except Exception as e:
            self.logger.error(f"加载配置文件时出错: {str(e)}")
            raise
            
    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            # 获取当前时间和用户信息
            current_time = self.get_current_time()
            current_user = self.get_current_user()
            
            # 更新配置中的时间戳和用户信息
            self.config['project']['updated_at'] = current_time
            self.config['project']['updated_by'] = current_user
            
            # 创建配置文件头部注释
            header_comment = (
                f"# Steel Power Prediction Project Configuration\n"
                f"# Current Date and Time (UTC): {current_time}\n"
                f"# Current User: {current_user}\n\n"
            )
            
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存配置
            config_str = yaml.safe_dump(
                self.config,
                allow_unicode=True,
                default_flow_style=False,
                sort_keys=False
            )
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                f.write(header_comment + config_str)
                
            self.logger.info("配置保存成功")
            
        except Exception as e:
            self.logger.error(f"保存配置文件时出错: {str(e)}")
            raise
            
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            updates: 要更新的配置项
        """
        try:
            self._recursive_update(self.config, updates)
            self.save_config()
            self.logger.info("配置更新成功")
            
        except Exception as e:
            self.logger.error(f"更新配置时出错: {str(e)}")
            raise
            
    def _recursive_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
        """递归更新字典"""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._recursive_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
        
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            required_sections = ['project', 'model', 'training', 'data', 'environment']
            for section in required_sections:
                if section not in self.config:
                    raise ValueError(f"缺少必要的配置部分: {section}")
                    
            # 验证数据比例
            data_config = self.config['data']
            total_ratio = sum([
                data_config['train_ratio'],
                data_config['val_ratio'],
                data_config['test_ratio']
            ])
            if not 0.99 <= total_ratio <= 1.01:
                raise ValueError(f"数据集比例之和必须为1，当前为: {total_ratio}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {str(e)}")
            return False
            
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取特定配置部分"""
        if section not in self.config:
            raise KeyError(f"配置中不存在部分: {section}")
        return self.config[section]

def test_config_manager():
    """测试配置管理器功能"""
    print("\n开始测试配置管理器...")
    
    # 创建配置管理器实例
    config_manager = ConfigManager()
    
    # 测试1: 加载配置
    print("\n测试1: 加载配置")
    config = config_manager.get_config()
    print(f"项目名称: {config['project']['name']}")
    print(f"模型类型: {config['model']['type']}")
    
    # 测试2: 更新配置
    print("\n测试2: 更新配置")
    updates = {
        "model": {
            "learning_rate": 0.0005,
            "batch_size": 64
        }
    }
    config_manager.update_config(updates)
    updated_config = config_manager.get_config()
    print(f"更新后的学习率: {updated_config['model']['learning_rate']}")
    print(f"更新后的批次大小: {updated_config['model']['batch_size']}")
    
    # 测试3: 验证配置
    print("\n测试3: 验证配置")
    is_valid = config_manager.validate_config()
    print(f"配置是否有效: {is_valid}")
    
    # 测试4: 获取特定部分
    print("\n测试4: 获取特定配置部分")
    data_config = config_manager.get_section('data')
    print(f"数据配置:\n{json.dumps(data_config, indent=2, ensure_ascii=False)}")
    
    # 测试5: 时间和用户信息
    print("\n测试5: 时间和用户信息")
    print(f"当前UTC时间: {config_manager.get_current_time()}")
    print(f"当前用户: {config_manager.get_current_user()}")
    
    print("\n配置管理器测试完成!")

if __name__ == "__main__":
    test_config_manager()
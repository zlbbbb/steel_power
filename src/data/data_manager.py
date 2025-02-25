"""
数据管理器：负责数据加载、存储和版本控制
Current Date and Time (UTC): 2025-02-25 12:13:38
Current User: zlbbbb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import Dict, Any, Optional, List, Union, Tuple
import json
import shutil
import hashlib
import sys
import yaml

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入配置管理器
from config.config_manager import ConfigManager

class DataManager:
    """数据管理器类,负责数据文件的管理和版本控制"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置管理器实例,如果为None则创建新实例
        """
        try:
            # 初始化日志记录器
            self.logger = logging.getLogger(__name__)
            
            # 记录初始化时间和用户信息
            self.init_time = datetime.now(timezone.utc)
            self.current_user = "zlbbbb"
            
            # 如果未提供配置,则创建新实例
            if config is None:
                config_path = project_root / 'config' / 'config.yaml'
                config = ConfigManager(config_path)
            
            self.config = config.get_config()
            
            # 初始化路径
            self._initialize_paths()
            
            # 加载数据版本历史记录
            self.version_history = self._load_version_history()
            
            self.logger.info(
                f"数据管理器初始化完成:\n"
                f"- 初始化时间: {self.init_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- 当前用户: {self.current_user}\n"
                f"- 数据目录: {self.data_root}"
            )
            
        except Exception as e:
            self.logger.error(f"数据管理器初始化失败: {str(e)}")
            raise

    def _initialize_paths(self) -> None:
        """初始化所有数据相关路径"""
        try:
            # 设置主数据目录
            self.data_root = project_root / 'data'
            
            # 设置各子目录,符合项目标准结构
            self.paths = {
                'raw': self.data_root / 'raw',  # 原始数据
                'processed': {
                    'interim': self.data_root / 'processed' / 'interim',  # 中间数据
                    'features': self.data_root / 'processed' / 'features',  # 特征数据
                    'final': self.data_root / 'processed' / 'final'  # 最终数据
                }
            }
            
            # 创建必要的目录
            for path in [self.paths['raw']] + list(self.paths['processed'].values()):
                path.mkdir(parents=True, exist_ok=True)
                
            # 版本历史记录文件路径
            self.version_file = self.data_root / 'processed' / 'version_history.json'
            
            self.logger.info(
                f"数据目录结构初始化完成:\n"
                f"- 原始数据: {self.paths['raw']}\n"
                f"- 中间数据: {self.paths['processed']['interim']}\n"
                f"- 特征数据: {self.paths['processed']['features']}\n"
                f"- 最终数据: {self.paths['processed']['final']}"
            )
            
        except Exception as e:
            self.logger.error(f"路径初始化失败: {str(e)}")
            raise

    def save_data(self, 
                  df: pd.DataFrame, 
                  stage: str, 
                  description: str = "") -> str:
        """
        保存数据文件并创建新版本
        
        Args:
            df: 待保存的数据框
            stage: 处理阶段('interim', 'features', 'final')
            description: 版本描述
            
        Returns:
            str: 版本哈希值
        """
        try:
            if stage not in self.paths['processed']:
                raise ValueError(f"无效的处理阶段: {stage}")
            
            # 生成文件名(不含时间戳)
            file_name = f"{stage}_data.csv"
            file_path = self.paths['processed'][stage] / file_name
            
            # 备份旧文件(如果存在)
            if file_path.exists():
                backup_dir = file_path.parent / 'backup'
                backup_dir.mkdir(exist_ok=True)
                backup_time = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                backup_path = backup_dir / f"{stage}_data_{backup_time}.csv"
                shutil.move(str(file_path), str(backup_path))
                self.logger.info(f"已备份旧文件: {backup_path}")
            
            # 保存新数据文件
            df.to_csv(file_path, index=False)
            
            # 计算文件哈希值
            file_hash = self._calculate_hash(file_path)
            
            # 创建版本信息
            version_info = {
                'hash': file_hash,
                'stage': stage,
                'file_path': str(file_path),
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'user': self.current_user,
                'description': description,
                'data_info': {
                    'rows': len(df),
                    'columns': list(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
                }
            }
            
            # 更新版本历史
            self.version_history['versions'].append(version_info)
            self._save_version_history()
            
            self.logger.info(
                f"数据保存完成:\n"
                f"- 阶段: {stage}\n"
                f"- 文件: {file_path}\n"
                f"- 版本哈希: {file_hash}\n"
                f"- 大小: {version_info['data_info']['memory_usage_mb']:.2f}MB"
            )
            
            return file_hash
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {str(e)}")
            raise

    def load_data(self, 
                  stage: str, 
                  version_hash: Optional[str] = None) -> pd.DataFrame:
        """
        加载指定版本的数据
        
        Args:
            stage: 处理阶段('interim', 'features', 'final')
            version_hash: 版本哈希值,为None则加载最新版本
            
        Returns:
            pd.DataFrame: 加载的数据框
        """
        try:
            if stage not in self.paths['processed']:
                raise ValueError(f"无效的处理阶段: {stage}")
            
            # 如果未指定版本,加载最新文件
            if version_hash is None:
                file_path = self.paths['processed'][stage] / f"{stage}_data.csv"
                if not file_path.exists():
                    raise FileNotFoundError(f"未找到数据文件: {file_path}")
            else:
                # 在版本历史中查找指定版本
                version = next(
                    (v for v in self.version_history['versions'] 
                     if v['hash'] == version_hash and v['stage'] == stage),
                    None
                )
                if version is None:
                    raise ValueError(f"未找到指定版本: {version_hash}")
                file_path = Path(version['file_path'])
            
            # 加载数据
            df = pd.read_csv(file_path)
            
            self.logger.info(
                f"数据加载完成:\n"
                f"- 阶段: {stage}\n"
                f"- 文件: {file_path}\n"
                f"- 形状: {df.shape}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {str(e)}")
            raise
    def _load_version_history(self) -> Dict:
        """
        加载版本历史记录
        
        Returns:
            Dict: 版本历史记录字典
        """
        try:
            if self.version_file.exists():
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                self.logger.info(f"加载版本历史记录: {len(history.get('versions', []))} 个版本")
                return history
            
            # 如果文件不存在，创建新的版本历史记录
            history = {'versions': []}
            self._save_version_history(history)
            self.logger.info("创建新的版本历史记录")
            return history
            
        except Exception as e:
            self.logger.error(f"加载版本历史记录失败: {str(e)}")
            # 如果加载失败，返回空的版本历史
            return {'versions': []}

    def _save_version_history(self, history: Optional[Dict] = None) -> None:
        """
        保存版本历史记录
        
        Args:
            history: 要保存的版本历史记录,如果为None则保存当前的version_history
        """
        try:
            # 确保版本文件的目录存在
            self.version_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果没有提供history，使用当前的version_history
            if history is None:
                history = self.version_history
            
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
                
            self.logger.info(f"版本历史记录已保存: {self.version_file}")
            
        except Exception as e:
            self.logger.error(f"保存版本历史记录失败: {str(e)}")
            raise

    def _calculate_hash(self, file_path: Path) -> str:
        """
        计算文件的MD5哈希值
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: MD5哈希值
        """
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                # 分块读取文件以处理大文件
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
            
        except Exception as e:
            self.logger.error(f"计算文件哈希值失败: {str(e)}")
            raise

    def get_version_info(self, version_hash: str) -> Dict:
        """
        获取指定版本的详细信息
        
        Args:
            version_hash: 版本哈希值
            
        Returns:
            Dict: 版本信息字典
        """
        try:
            version = next(
                (v for v in self.version_history['versions'] 
                 if v['hash'] == version_hash),
                None
            )
            
            if version is None:
                raise ValueError(f"未找到指定版本: {version_hash}")
                
            return version
            
        except Exception as e:
            self.logger.error(f"获取版本信息失败: {str(e)}")
            raise

    def list_versions(self, stage: Optional[str] = None) -> List[Dict]:
        """
        列出指定阶段或所有的数据版本
        
        Args:
            stage: 处理阶段,为None则列出所有版本
            
        Returns:
            List[Dict]: 版本信息列表
        """
        try:
            versions = self.version_history['versions']
            
            if stage:
                versions = [v for v in versions if v['stage'] == stage]
                
            # 按时间戳降序排序
            sorted_versions = sorted(
                versions,
                key=lambda x: x['timestamp'],
                reverse=True
            )
            
            self.logger.info(
                f"获取版本列表:\n"
                f"- 阶段: {stage if stage else '所有'}\n"
                f"- 版本数量: {len(sorted_versions)}"
            )
            
            return sorted_versions
            
        except Exception as e:
            self.logger.error(f"获取版本列表失败: {str(e)}")
            raise

    def __str__(self) -> str:
        """返回数据管理器的字符串表示"""
        try:
            version_count = len(self.version_history.get('versions', []))
            return (
                f"DataManager(\n"
                f"    用户: {self.current_user}\n"
                f"    初始化时间: {self.init_time}\n"
                f"    版本数量: {version_count}\n"
                f")"
            )
        except Exception:
            return "DataManager(初始化失败)"

    def __repr__(self) -> str:
        """返回数据管理器的详细表示"""
        try:
            return (
                f"DataManager(\n"
                f"    用户: {self.current_user}\n"
                f"    初始化时间: {self.init_time}\n"
                f"    数据目录: {self.data_root}\n"
                f"    版本文件: {self.version_file}\n"
                f"    版本数量: {len(self.version_history.get('versions', []))}\n"
                f")"
            )
        except Exception:
            return "DataManager(初始化失败)"    

    

if __name__ == '__main__':
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 测试数据管理器
    try:
        # 初始化数据管理器
        data_manager = DataManager()
        
        # 测试目录结构
        print("\n数据目录结构:")
        for stage, path in {
            'raw': data_manager.paths['raw'],
            'interim': data_manager.paths['processed']['interim'],
            'features': data_manager.paths['processed']['features'],
            'final': data_manager.paths['processed']['final']
        }.items():
            print(f"- {stage}: {path}")
            
        # 加载示例数据
        df = pd.DataFrame({
            'A': range(5),
            'B': range(5, 10)
        })
        
        # 测试各阶段数据保存
        for stage in ['interim', 'features', 'final']:
            version_hash = data_manager.save_data(
                df=df,
                stage=stage,
                description=f'测试{stage}阶段数据'
            )
            
            # 测试加载数据
            loaded_df = data_manager.load_data(stage)
            print(f"\n{stage}阶段数据加载成功,形状: {loaded_df.shape}")
        
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
        raise
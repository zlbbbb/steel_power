"""
Steel Power Prediction Feature Engineering Module
Current Date and Time (UTC): 2025-02-25 14:10:48
Current User: zlbbbb
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime, timezone
import logging
import json
import warnings
from sklearn.preprocessing import StandardScaler, RobustScaler
import yaml


class FeatureEngineer:
    """
    特征工程类：用于钢铁行业数据的特征生成和转换
    
    属性:
        config: 特征工程配置
        logger: 日志记录器
        scalers: 特征缩放器字典
        feature_info: 特征信息字典
        process_start_time: 处理开始时间
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化特征工程类
        
        Args:
            config_path: 配置文件路径，默认为None
        """
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_info = {}
        self.process_start_time = datetime.now(timezone.utc)

        # 设置数据存储路径
        self.data_dir = Path("data/processed/features")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 文件名配置
        self.file_names = {
            'time': 'time_features.csv',
            'lag': 'lag_features.csv',
            'rolling': 'rolling_features.csv',
            'final': 'final_features.csv',
            'metadata': 'feature_engineering_metadata.json'
        }

        
        # 加载配置
        self.config = self._load_config(config_path) if config_path else {
            'feature_engineering': {
                'time_features': {
                    'enabled': True,
                    'cyclical_encoding': True
                },
                'lag_features': {
                    'enabled': True,
                    'periods': [1, 2, 3, 7, 14, 30]
                },
                'rolling_features': {
                    'enabled': True,
                    'windows': [24, 72, 168],  # 1天，3天，7天
                    'functions': ['mean', 'std', 'min', 'max']
                },
                'interaction_features': {
                    'enabled': True
                },
                'scaling': {
                    'method': 'standard',
                    'target_cols': []
                }
            }
        }
        
        self.logger.info(
            f"特征工程器初始化完成\n"
            f"数据存储路径: {self.data_dir}\n"
            f"配置信息: {json.dumps(self.config, indent=2, ensure_ascii=False)}"
        )
    def save_features(self, 
                     df: pd.DataFrame, 
                     feature_type: str,
                     description: str = "") -> Path:
        """
        保存特征数据到指定目录
        
        Args:
            df: 特征数据框
            feature_type: 特征类型（如'time', 'lag', 'rolling'等）
            description: 特征描述
            
        Returns:
            保存文件的路径
        """
        try:
            # 生成时间戳 规避重复多个
            #timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            
            # 创建特征类型子目录
            feature_dir = self.data_dir / feature_type
            feature_dir.mkdir(parents=True, exist_ok=True)
            
            # 构建文件名
            file_path = feature_dir / self.file_names[feature_type]
            # 删除旧文件（如果存在）
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"已删除旧文件: {file_path}")

            # 保存数据
            df.to_csv(file_path, index=False)
            
            # 保存元数据
            metadata = {
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'feature_type': feature_type,
                'description': description,
                'columns': list(df.columns),
                'shape': df.shape,
                'created_by': 'zlbbbb',
                'feature_info': self.feature_info.get(feature_type, {})
            }
            
            metadata_path = feature_dir / f"{feature_type}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.info(
                f"特征数据已保存:\n"
                f"- 类型: {feature_type}\n"
                f"- 文件: {file_path}\n"
                f"- 元数据: {metadata_path}\n"
                f"- 特征数量: {len(df.columns)}"
            )
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"保存特征数据失败: {str(e)}")
            raise
    def save_feature_metadata(self, metadata_path: Union[str, Path]) -> None:
        """
        保存特征工程的完整元数据
        
        Args:
            metadata_path: 元数据保存路径
        """
        try:
            if metadata_path is None:
                metadata_path = self.data_dir / self.file_names['metadata']
            else:
                metadata_path = Path(metadata_path)
                
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 如果存在旧的元数据文件，先删除
            if metadata_path.exists():
                metadata_path.unlink()
                self.logger.info(f"已删除旧的元数据文件: {metadata_path}")
                
            metadata = {
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'created_by': 'zlbbbb',
                'process_time': (datetime.now(timezone.utc) - 
                               self.process_start_time).total_seconds(),
                'config': self.config,
                'feature_info': self.feature_info,
                'scaling_info': {
                    name: {
                        'type': type(scaler).__name__,
                        'parameters': scaler.get_params(),
                        'mean': float(scaler.mean_[0]) if hasattr(scaler, 'mean_') else None,
                        'scale': float(scaler.scale_[0]) if hasattr(scaler, 'scale_') else None
                    } for name, scaler in self.scalers.items()
                },
                'feature_counts': {
                    'time_features': len(self.feature_info.get('time_features', [])),
                    'lag_features': len(self.feature_info.get('lag_features', [])),
                    'rolling_features': len(self.feature_info.get('rolling_features', [])),
                    'scaled_features': len(self.feature_info.get('scaled_features', []))
                }
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(
                f"特征工程元数据已保存:\n"
                f"- 路径: {metadata_path}\n"
                f"- 特征统计: {json.dumps(metadata['feature_counts'], indent=2)}"
            )
            
        except Exception as e:
            self.logger.error(f"保存特征工程元数据失败: {str(e)}")
            raise

    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"成功加载配置文件: {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {str(e)}")
            raise
    
    def create_time_features(self, 
                           df: pd.DataFrame, 
                           datetime_col: str = 'date',
                           cyclical_encoding: bool = True) -> pd.DataFrame:
        """
        创建时间特征
        
        Args:
            df: 输入数据框
            datetime_col: 时间列名
            cyclical_encoding: 是否使用周期编码
            
        Returns:
            添加时间特征后的数据框
        """
        try:
            df = df.copy()
            if datetime_col not in df.columns:
                raise ValueError(f"未找到时间列: {datetime_col}")
                
            # 确保时间列为datetime类型
            df[datetime_col] = pd.to_datetime(df[datetime_col])
            
            # 记录特征生成信息
            time_features = []
            
            # 基础时间特征
            base_features = {
                'hour': df[datetime_col].dt.hour,
                'day': df[datetime_col].dt.day,
                'month': df[datetime_col].dt.month,
                'year': df[datetime_col].dt.year,
                'dayofweek': df[datetime_col].dt.dayofweek,
                'quarter': df[datetime_col].dt.quarter
            }
            
            df = df.assign(**base_features)
            time_features.extend(base_features.keys())
            
            # 周期性编码
            if cyclical_encoding:
                cyclical_features = {
                    'hour_sin': np.sin(2 * np.pi * df['hour']/24),
                    'hour_cos': np.cos(2 * np.pi * df['hour']/24),
                    'month_sin': np.sin(2 * np.pi * df['month']/12),
                    'month_cos': np.cos(2 * np.pi * df['month']/12),
                    'dayofweek_sin': np.sin(2 * np.pi * df['dayofweek']/7),
                    'dayofweek_cos': np.cos(2 * np.pi * df['dayofweek']/7)
                }
                
                df = df.assign(**cyclical_features)
                time_features.extend(cyclical_features.keys())
            
            # 特殊时间特征
            special_features = {
                'is_weekend': df['dayofweek'].isin([5, 6]).astype(int),
                'is_month_start': df[datetime_col].dt.is_month_start.astype(int),
                'is_month_end': df[datetime_col].dt.is_month_end.astype(int)
            }
            
            df = df.assign(**special_features)
            time_features.extend(special_features.keys())
            
            # 更新特征信息
            self.feature_info['time_features'] = time_features
            
            self.logger.info(
                f"时间特征创建完成:\n"
                f"- 基础特征: {list(base_features.keys())}\n"
                f"- 周期特征: {list(cyclical_features.keys()) if cyclical_encoding else []}\n"
                f"- 特殊特征: {list(special_features.keys())}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建时间特征失败: {str(e)}")
            raise
    
    def create_lag_features(self, 
                          df: pd.DataFrame,
                          columns: List[str],
                          periods: List[int],
                          group_col: Optional[str] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入数据框
            columns: 需要创建滞后特征的列
            periods: 滞后周期列表
            group_col: 分组列名，用于分组计算滞后特征
            
        Returns:
            添加滞后特征后的数据框
        """
        try:
            df = df.copy()
            lag_features = []
            
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"未找到列: {col}")
                
                for period in periods:
                    lag_name = f"{col}_lag_{period}"
                    
                    if group_col:
                        df[lag_name] = df.groupby(group_col)[col].shift(period)
                    else:
                        df[lag_name] = df[col].shift(period)
                    
                    lag_features.append(lag_name)
            
            # 更新特征信息
            self.feature_info['lag_features'] = lag_features
            
            self.logger.info(
                f"滞后特征创建完成:\n"
                f"- 特征数量: {len(columns)}\n"
                f"- 滞后周期: {periods}\n"
                f"- 分组列: {group_col if group_col else '无'}\n"
                f"- 总计生成: {len(lag_features)} 个特征"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建滞后特征失败: {str(e)}")
            raise
    def create_rolling_features(self,
                              df: pd.DataFrame,
                              columns: List[str],
                              windows: List[int],
                              functions: List[str]) -> pd.DataFrame:
        """
        创建滚动统计特征
        
        Args:
            df: 输入数据框
            columns: 需要创建滚动特征的列
            windows: 窗口大小列表
            functions: 统计函数列表 ['mean', 'std', 'min', 'max']
            
        Returns:
            添加滚动统计特征后的数据框
        """
        try:
            df = df.copy()
            rolling_features = []
            valid_functions = ['mean', 'std', 'min', 'max', 'median']
            
            # 验证函数名
            invalid_functions = [f for f in functions if f not in valid_functions]
            if invalid_functions:
                raise ValueError(f"不支持的统计函数: {invalid_functions}")
            
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"未找到列: {col}")
                    
                for window in windows:
                    for func in functions:
                        feature_name = f"{col}_{func}_{window}"
                        df[feature_name] = (df[col]
                                          .rolling(window=window, min_periods=1)
                                          .agg(func))
                        rolling_features.append(feature_name)
            
            # 更新特征信息
            self.feature_info['rolling_features'] = rolling_features
            
            self.logger.info(
                f"滚动统计特征创建完成:\n"
                f"- 特征数量: {len(columns)}\n"
                f"- 窗口大小: {windows}\n"
                f"- 统计函数: {functions}\n"
                f"- 总计生成: {len(rolling_features)} 个特征"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"创建滚动统计特征失败: {str(e)}")
            raise
            
    def create_all_features(self, 
                          df: pd.DataFrame,
                          datetime_col: str = 'date',
                          save_intermediate: bool = True) -> pd.DataFrame:
        """
        创建并可选保存所有特征
        
        Args:
            df: 输入数据框
            datetime_col: 时间列名
            save_intermediate: 是否保存中间特征
            
        Returns:
            处理后的数据框
        """
        try:
            self.logger.info("开始创建所有特征...")
            df_result = df.copy()
            
            config = self.config['feature_engineering']
            
            # 1. 创建时间特征
            if config['time_features']['enabled']:
                df_result = self.create_time_features(
                    df_result,
                    datetime_col=datetime_col,
                    cyclical_encoding=config['time_features']['cyclical_encoding']
                )
                if save_intermediate:
                    self.save_features(
                        df_result,
                        'time',
                        "时间特征，包含基础和周期性特征"
                    )
            
            # 2. 创建滞后特征
            if config['lag_features']['enabled']:
                lag_cols = config['scaling']['target_cols']
                if lag_cols:
                    df_result = self.create_lag_features(
                        df_result,
                        columns=lag_cols,
                        periods=config['lag_features']['periods']
                    )
                    if save_intermediate:
                        self.save_features(
                            df_result,
                            'lag',
                            f"滞后特征，周期：{config['lag_features']['periods']}"
                        )
            
            # 3. 创建滚动特征
            if config['rolling_features']['enabled']:
                rolling_cols = config['scaling']['target_cols']
                if rolling_cols:
                    df_result = self.create_rolling_features(
                        df_result,
                        columns=rolling_cols,
                        windows=config['rolling_features']['windows'],
                        functions=config['rolling_features']['functions']
                    )
                    if save_intermediate:
                        self.save_features(
                            df_result,
                            'rolling',
                            f"滚动特征，窗口：{config['rolling_features']['windows']}"
                        )
            
            # 4. 特征缩放
            if config['scaling']['target_cols']:
                df_result = self.scale_features(
                    df_result,
                    columns=config['scaling']['target_cols'],
                    method=config['scaling']['method']
                )
            
# 修改保存元数据的部分：

            # 保存最终特征
            if config['output']['save_features']:
                final_path = self.save_features(
                    df_result,
                    'final',
                    "所有特征处理完成后的最终数据"
                )
                
                # 保存特征工程元数据
                if config['output']['save_metadata']:
                    metadata_dir = self.data_dir / 'metadata'
                    metadata_dir.mkdir(parents=True, exist_ok=True)
                    
                    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
                    metadata_path = metadata_dir / f'feature_engineering_metadata_{timestamp}.json'
                    
                    self.save_feature_metadata(metadata_path)
            
            return df_result
            
        except Exception as e:
            self.logger.error(f"创建所有特征失败: {str(e)}")
            raise


    def get_feature_summary(self) -> Dict[str, Any]:
        """
        获取特征工程处理摘要
        
        Returns:
            包含特征信息的字典
        """
        return {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'processing_time': (datetime.now(timezone.utc) - 
                              self.process_start_time).total_seconds(),
            'feature_counts': {
                category: len(features) 
                for category, features in self.feature_info.items()
            },
            'feature_details': self.feature_info,
            'scaling_info': {
                name: type(scaler).__name__ 
                for name, scaler in self.scalers.items()
            }
        }

    def scale_features(self, 
                      df: pd.DataFrame,
                      columns: List[str],
                      method: str = 'standard') -> pd.DataFrame:
        """
        对特征进行缩放
        
        Args:
            df: 输入数据框
            columns: 需要缩放的列
            method: 缩放方法 ('standard' 或 'robust')
            
        Returns:
            缩放后的数据框
        """
        try:
            df = df.copy()
            scaled_features = []
            
            for col in columns:
                if col not in df.columns:
                    raise ValueError(f"未找到列: {col}")
                
                # 检查是否已经存在缩放器
                if col not in self.scalers:
                    if method == 'standard':
                        self.scalers[col] = StandardScaler()
                    elif method == 'robust':
                        self.scalers[col] = RobustScaler()
                    else:
                        raise ValueError(f"不支持的缩放方法: {method}")
                
                # 重塑数据为2D形式
                data_2d = df[col].values.reshape(-1, 1)
                
                # 执行缩放
                df[col] = self.scalers[col].fit_transform(data_2d).ravel()
                scaled_features.append(col)
            
            # 更新特征信息
            self.feature_info['scaled_features'] = scaled_features
            
            # 记录缩放器信息
            scalers_info = {}
            for col, scaler in self.scalers.items():
                scalers_info[col] = {
                    'type': type(scaler).__name__
                }
                if hasattr(scaler, 'mean_'):
                    scalers_info[col]['mean'] = float(scaler.mean_[0])
                if hasattr(scaler, 'scale_'):
                    scalers_info[col]['scale'] = float(scaler.scale_[0])
            
            self.logger.info(
                f"特征缩放完成:\n"
                f"- 缩放方法: {method}\n"
                f"- 缩放特征数: {len(scaled_features)}\n"
                f"- 缩放器信息: {json.dumps(scalers_info, indent=2)}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征缩放失败: {str(e)}")
            raise
    
    def inverse_scale_features(self,
                             df: pd.DataFrame,
                             columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        将缩放后的特征还原为原始尺度
        
        Args:
            df: 输入数据框
            columns: 需要还原的列，如果为None则还原所有已缩放的特征
            
        Returns:
            还原后的数据框
        """
        try:
            df = df.copy()
            columns = columns or list(self.scalers.keys())
            
            for col in columns:
                if col not in self.scalers:
                    raise ValueError(f"列 {col} 没有对应的缩放器")
                    
                if col not in df.columns:
                    raise ValueError(f"未找到列: {col}")
                
                # 重塑数据为2D形式
                data_2d = df[col].values.reshape(-1, 1)
                
                # 执行反向变换
                df[col] = self.scalers[col].inverse_transform(data_2d).ravel()
            
            self.logger.info(f"已将 {len(columns)} 个特征还原至原始尺度")
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征还原失败: {str(e)}")
            raise

    def get_scaling_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取特征缩放信息
        
        Returns:
            包含缩放信息的字典
        """
        scaling_info = {}
        
        for col, scaler in self.scalers.items():
            info = {
                'type': type(scaler).__name__,
                'parameters': scaler.get_params()
            }
            
            if hasattr(scaler, 'mean_'):
                info['mean'] = float(scaler.mean_[0])
            if hasattr(scaler, 'scale_'):
                info['scale'] = float(scaler.scale_[0])
            
            scaling_info[col] = info
        
        return scaling_info

    def process_features(self, 
                        df: pd.DataFrame, 
                        datetime_col: str = 'date') -> pd.DataFrame:
        """
        特征处理主函数
        
        Args:
            df: 输入数据框
            datetime_col: 时间列名
            
        Returns:
            处理后的数据框
        """
        try:
            self.logger.info("开始特征处理...")
            df = df.copy()
            
            # 1. 创建时间特征
            if self.config['feature_engineering']['time_features']['enabled']:
                df = self.create_time_features(
                    df, 
                    datetime_col=datetime_col,
                    cyclical_encoding=self.config['feature_engineering']
                                            ['time_features']['cyclical_encoding']
                )
            
            # 2. 创建滞后特征
            if self.config['feature_engineering']['lag_features']['enabled']:
                target_cols = self.config['feature_engineering']['scaling']['target_cols']
                periods = self.config['feature_engineering']['lag_features']['periods']
                
                df = self.create_lag_features(df, target_cols, periods)
            
            # 3. 创建滚动特征
            if self.config['feature_engineering']['rolling_features']['enabled']:
                windows = self.config['feature_engineering']['rolling_features']['windows']
                functions = self.config['feature_engineering']['rolling_features']['functions']
                
                df = self.create_rolling_features(df, target_cols, windows, functions)
            
            # 4. 特征缩放
            if self.config['feature_engineering']['scaling']['target_cols']:
                scale_cols = self.config['feature_engineering']['scaling']['target_cols']
                scale_method = self.config['feature_engineering']['scaling']['method']
                
                df = self.scale_features(df, scale_cols, method=scale_method)
            
            self.logger.info("特征处理完成")
            return df
            
        except Exception as e:
            self.logger.error(f"特征处理失败: {str(e)}")
            raise
    
    def __str__(self) -> str:
        """返回特征工程器的字符串表示"""
        return (
            f"FeatureEngineer(\n"
            f"    特征数量: {len(self.feature_info)}\n"
            f"    处理开始时间: {self.process_start_time}\n"
            f")"
        )
    
    def __repr__(self) -> str:
        """返回特征工程器的详细表示"""
        return (
            f"FeatureEngineer(\n"
            f"    配置: {json.dumps(self.config, indent=2, ensure_ascii=False)}\n"
            f"    特征信息: {json.dumps(self.feature_info, indent=2, ensure_ascii=False)}\n"
            f"    处理开始时间: {self.process_start_time}\n"
            f")"
        )


def _run_internal_tests():
    """
    运行内部测试
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始运行内部测试...")
        
        # 创建测试数据
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
        test_data = pd.DataFrame({
            'date': dates,
            'power': np.random.normal(100, 10, len(dates)),
            'temperature': np.random.normal(25, 5, len(dates))
        })
        
        # 测试配置
        test_config = {
            'feature_engineering': {
                'time_features': {
                    'enabled': True,
                    'cyclical_encoding': True
                },
                'lag_features': {
                    'enabled': True,
                    'periods': [1, 2, 3]
                },
                'rolling_features': {
                    'enabled': True,
                    'windows': [24, 48],
                    'functions': ['mean', 'std']
                },
                'scaling': {
                    'method': 'standard',
                    'target_cols': ['power', 'temperature']
                },
                'output': {
                    'save_features': True,
                    'save_metadata': True
                }
            }
        }
        
        # 初始化特征工程器
        fe = FeatureEngineer()
        fe.config = test_config
        
        # ... [其他测试步骤保持不变] ...
        
        # 测试7：特征保存
        logger.info("测试7：特征保存")
        df_final = fe.create_all_features(test_data, save_intermediate=True)
        
        # 验证保存的文件
        assert (Path("data/processed/features/final").exists()), "未找到特征保存目录"
        
        logger.info("所有测试通过!")
        
    except Exception as e:
        logger.error(f"测试失败: {str(e)}")
        raise
        
    finally:
        logger.info("内部测试完成")

if __name__ == '__main__':
    _run_internal_tests()

"""
数据预处理器
Current Date and Time (UTC): 2025-02-25 06:26:36
Current User: zlbbbb
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime,timezone
import logging
import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 现在可以导入 config 模块
from config.config_manager import ConfigManager
class DataPreprocessor:
    """数据预处理器类，负责数据的预处理和特征转换"""
    
    def __init__(self, config):
        """
        初始化数据预处理器
        
        Args:
            config: 配置管理器实例
        """
        try:
            if config is None:
                config_path = project_root / 'config' / 'config.yaml'
                config = ConfigManager(config_path)
                
            self.config = config.get_config()
            
            # 验证配置中是否包含数据配置
            if 'data' not in self.config:
                raise KeyError("配置文件中缺少 'data' 配置")
                
            self.data_config = self.config['data']
            self.logger = logging.getLogger(__name__)
            
            # 获取项目根目录
            self.project_root = project_root
            
            # 初始化数据路径
            self._initialize_paths()
            
            # 初始化标准化器字典
            self.scalers: Dict[str, Any] = {}
            
            # 记录处理开始时间
            self.process_start_time = datetime.now(timezone.utc)
            
            self.logger.info(
                f"数据预处理器初始化完成:\n"
                f"- 配置项: {list(self.data_config.keys())}\n"
                f"- 时间: {self.process_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
        except Exception as e:
            logging.error(f"数据预处理器初始化失败: {str(e)}")
            raise


    def _initialize_paths(self):
        """初始化所有相关路径"""
        # 构建数据相关路径
        self.raw_data_path = self.project_root / self.data_config['paths']['raw']
        self.processed_dir = self.project_root / self.data_config['paths']['processed']
        self.interim_dir = self.project_root / self.data_config['paths']['interim']
        self.features_dir = self.project_root / self.data_config['paths']['features']
        self.final_dir = self.project_root / self.data_config['paths']['final']
        
        # 创建必要的目录
        for path in [self.processed_dir, self.interim_dir, self.features_dir, self.final_dir]:
            path.mkdir(parents=True, exist_ok=True)
            
        self.logger.info(
            f"数据目录初始化完成:\n"
            f"- 原始数据: {self.raw_data_path}\n"
            f"- 处理后数据: {self.processed_dir}\n"
            f"- 中间结果: {self.interim_dir}\n"
            f"- 特征工程: {self.features_dir}\n"
            f"- 最终结果: {self.final_dir}"
        )
    def _validate_date_format(self, date_series: pd.Series) -> None:
        """
        验证日期格式是否符合 DD/MM/YYYY HH:MM
        数据源数据类型
        
        Args:
            date_series: 待验证的日期序列
            
        Raises:
            ValueError: 日期格式错误时抛出
        """
        import re

        # 日期格式正则表达式
        date_pattern = r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$'
        
        # 检查格式
        invalid_dates = date_series[~date_series.str.match(date_pattern)]
        if not invalid_dates.empty:
            error_msg = (
                f"发现{len(invalid_dates)}条无效日期格式。示例:\n"
                f"{invalid_dates.iloc[:5].tolist()}\n"
                f"期望格式: DD/MM/YYYY HH:MM"
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)
            
        # 检查日期值的有效性
        try:
            pd.to_datetime(date_series, format='%d/%m/%Y %H:%M')
        except ValueError as e:
            error_msg = f"日期值无效: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)       
        
    def _process_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理时间列，保持原始格式 (DD/MM/YYYY HH:MM)
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
            
        Raises:
            ValueError: 日期格式错误时抛出
        """
        try:
            # 验证日期格式
            self._validate_date_format(df['date'])
            
            # 转换为datetime进行排序
            temp_date = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
            
            # 按时间排序
            df = df.assign(temp_date=temp_date)\
                   .sort_values('temp_date')\
                   .reset_index(drop=True)
            
            # 移除临时列
            df = df.drop('temp_date', axis=1)
            
            self.logger.info(
                f"时间处理完成:\n"
                f"- 数据时间范围: {df['date'].iloc[0]} 至 {df['date'].iloc[-1]}\n"
                f"- 总记录数: {len(df)}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"时间处理出错: {str(e)}")
            raise

    def _process_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理数值特征，包括类型转换、缺失值处理、异常值处理和标准化
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 处理后的数据框
        """
        numeric_columns = [
            'Usage_kWh',
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor'
        ]
        
        try:
            for col in numeric_columns:
                # 1. 转换为float类型
                df[col] = df[col].astype(float)
                
                # 2. 处理缺失值
                if df[col].isnull().any():
                    df[col] = self._handle_missing_values(df[col])
                
                # 3. 处理异常值
                df[col] = self._handle_outliers(df[col])
                
                # 4. 标准化
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                df[col] = self.scalers[col].fit_transform(
                    df[col].values.reshape(-1, 1)
                ).flatten()
                
            return df
            
        except Exception as e:
            self.logger.error(f"数值特征处理出错: {str(e)}")
            raise       

    def _handle_missing_values(self, series: pd.Series) -> pd.Series:
        """处理缺失值
        Args:
            series: 输入数据序列
            
        Returns:
            pd.Series: 处理后的数据序列        
        """
        method = self.data_config['preprocessing']['handle_missing']
        if method == 'interpolate':
            return series.interpolate(method='linear')
        elif method == 'mean':
            return series.fillna(series.mean())
        elif method == 'median':
            return series.fillna(series.median())
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")
        
    def _handle_outliers(self, series: pd.Series) -> pd.Series:
        """处理异常值
        Args:
            series: 输入数据序列
            
        Returns:
            pd.Series: 处理后的数据序列        
        """
        if not self.data_config['preprocessing']['outlier_detection']:
            return series
            
        method = self.data_config['preprocessing']['outlier_method']
        if method == 'IQR':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return series.clip(lower=lower_bound, upper=upper_bound)
        else:
            raise ValueError(f"不支持的异常值处理方法: {method}")

    def _process_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理类别特征"""
        try:
            # WeekStatus处理
            df['WeekStatus'] = pd.Categorical(df['WeekStatus'], 
                                            categories=['Weekday', 'Weekend'])
            
            # Day_of_week处理
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
            df['Day_of_week'] = pd.Categorical(df['Day_of_week'], 
                                             categories=days)
            
            # Load_Type处理
            load_types = ['Light_Load', 'Medium_Load', 'Maximum_Load']
            df['Load_Type'] = pd.Categorical(df['Load_Type'], 
                                           categories=load_types)
            
            # One-hot编码 eg 0 1 2
            df = pd.get_dummies(df, columns=['WeekStatus', 'Day_of_week', 'Load_Type'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"类别特征处理出错: {str(e)}")
            raise

    def _process_nsm(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理NSM特征 最大散耗功率
        
        """
        try:
            # 确保NSM是整数类型
            df['NSM'] = df['NSM'].astype(int)
            
            # 验证NSM范围
            if (df['NSM'] < 0).any() or (df['NSM'] > 100000).any():
                self.logger.warning("NSM值超出有效范围[0, 100000]")
                df['NSM'] = df['NSM'].clip(0, 100000)
            
            # 标准化NSM
            df['NSM'] = df['NSM'] / 100000  # 归一化到[0, 1]范围
            
            return df
            
        except Exception as e:
            self.logger.error(f"NSM处理出错: {str(e)}")
            raise

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加时间特征
        
        Args:
            df: 输入数据框
            
        Returns:
            pd.DataFrame: 添加时间特征后的数据框
        """
        try:
            # 临时转换为datetime以提取特征
            temp_date = pd.to_datetime(df['date'], format='%d/%m/%Y %H:%M')
            
            # 基础时间特征
            time_features = {
                'hour': temp_date.dt.hour / 23.0,  # [0, 1]
                'minute': temp_date.dt.minute / 59.0,  # [0, 1]
                'day_of_week': temp_date.dt.dayofweek / 6.0,  # [0, 1]
                'day_of_month': (temp_date.dt.day - 1) / 30.0,  # [0, 1]
                'month': (temp_date.dt.month - 1) / 11.0,  # [0, 1]
                'is_weekend': temp_date.dt.dayofweek.isin([5, 6]).astype(int),
                'is_business_hour': ((temp_date.dt.hour >= 9) & 
                                   (temp_date.dt.hour < 17)).astype(int)
            }
            
            # 添加周期性特征
            time_features.update({
                'hour_sin': np.sin(2 * np.pi * temp_date.dt.hour / 24),
                'hour_cos': np.cos(2 * np.pi * temp_date.dt.hour / 24),
                'month_sin': np.sin(2 * np.pi * temp_date.dt.month / 12),
                'month_cos': np.cos(2 * np.pi * temp_date.dt.month / 12),
                'day_of_week_sin': np.sin(2 * np.pi * temp_date.dt.dayofweek / 7),
                'day_of_week_cos': np.cos(2 * np.pi * temp_date.dt.dayofweek / 7)
            })
            
            # 添加特征到数据框
            df = df.assign(**time_features)
            
            self.logger.info(
                f"时间特征添加完成，新增特征:\n"
                f"- 基础特征: {list(time_features.keys())[:7]}\n"
                f"- 周期特征: {list(time_features.keys())[7:]}"
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"添加时间特征出错: {str(e)}")
            raise

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整的预处理流程
        
        Args:
            df: 原始数据框
            
        Returns:
            pd.DataFrame: 预处理后的数据框
        """
        try:
            # 1. 时间处理
            df = self._process_datetime(df)
            self.save_interim_data(df, 'datetime_processed')
            
            # 2. 数值特征处理
            df = self._process_numeric_features(df)
            self.save_interim_data(df, 'numeric_processed')
            
            # 3. 类别特征处理
            df = self._process_categorical_features(df)
            self.save_interim_data(df, 'categorical_processed')
            
            # 4. NSM处理
            df = self._process_nsm(df)
            self.save_interim_data(df, 'nsm_processed')
            
            # 5. 添加时间特征
            df = self._add_time_features(df)
            
            # 6. 保存最终结果
            self.save_final_data(df)
            
            # 记录处理完成时间
            process_end_time = datetime.now(timezone.utc)
            processing_time = (process_end_time - self.process_start_time).total_seconds()
            
            self.logger.info(
                f"预处理完成:\n"
                f"- 最终数据形状: {df.shape}\n"
                f"- 处理时间: {processing_time:.2f}秒\n"
                f"- 特征数量: {len(df.columns)}"
            )
            
            # 保存处理元数据
            #self._save_preprocessing_metadata(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"预处理过程出错: {str(e)}")
            raise


      
    def save_interim_data(self, df: pd.DataFrame, stage: str):
        """
        保存中间处理结果,去掉时间戳
        
        Args:
            df: 待保存的数据框
            stage: 处理阶段标识
        """
        try:
            # 使用下划线替换空格和冒号
            #timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') 避免重复生成
            file_path = self.interim_dir / f"{stage}.csv"
            
            #删除同名旧文件
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"已删除旧文件: {file_path}")
            # 确保目录存在
            #file_path.parent.mkdir(parents=True, exist_ok=True)
            #保存新文件
            df.to_csv(file_path, index=False)
            self.logger.info(f"中间结果已保存: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存中间结果失败: {str(e)}")
            raise
        
        
    def save_final_data(self, df: pd.DataFrame):
        """
        保存最终处理结果,只保留最新的文件
        
        Args:
            df: 待保存的数据框
        """
        try:
            # 使用下划线替换空格和冒 去掉时间戳
            #timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            file_path = self.final_dir / f"final_processed.csv"
            
            # 确保目录存在
            file_path.parent.mkdir(parents=True, exist_ok=True)
            #删除旧文件（如果存在）
            if file_path.exists():
                file_path.unlink()
                self.logger.info(f"已删除旧文件: {file_path}")
            #保存新文件
            
            df.to_csv(file_path, index=False)
            self.logger.info(f"最终处理结果已保存: {file_path}")
            
        except Exception as e:
            self.logger.error(f"保存最终结果失败: {str(e)}")
            raise
        
    def _save_preprocessing_metadata(self, df: pd.DataFrame) -> None:
        """
        保存预处理元数据,只保留最新的
        
        Args:
            df: 处理后的数据框
        """
        try:
            # 使用下划线替换空格和冒号
            #timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            
            metadata = {
                'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                'data_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'memory_usage': df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
                    'date_range': {
                        'start': df['date'].iloc[0],
                        'end': df['date'].iloc[-1]
                    }
                },
                'preprocessing_info': {
                    'config': self.data_config['preprocessing'],
                    'processing_time': (datetime.now(timezone.utc) - 
                                    self.process_start_time).total_seconds(),
                    'scalers': {
                        name: {
                            'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                            'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                        } for name, scaler in self.scalers.items()
                    }
                },
                'feature_info': {
                    'numeric_features': [col for col in df.select_dtypes(include=[np.number]).columns],
                    'categorical_features': [col for col in df.select_dtypes(include=['category']).columns],
                    'generated_features': [col for col in df.columns if col.startswith(('hour_', 'day_', 'month_', 'is_'))]
                }
            }
            
            # 使用有效的文件名格式(不含时间戳)
            metadata_path = self.final_dir / f"preprocessing_metadata.json"
            
            # 确保目录存在
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            #删除旧文件（如果存在）
            if metadata_path.exists():
                metadata_path.unlink()
                self.logger.info(f"已删除旧文件: {metadata_path}")
            
            # 保存新元数据
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"预处理元数据已保存: {metadata_path}")
            
        except Exception as e:
            self.logger.error(f"保存预处理元数据失败: {str(e)}")
            raise

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        获取预处理摘要信息
        
        Returns:
            Dict[str, Any]: 预处理摘要
        """
        return {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
            'scalers': {
                name: {
                    'mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
                    'scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None
                } for name, scaler in self.scalers.items()
            },
            'config': self.data_config['preprocessing'],
            'processing_duration': (datetime.now(timezone.utc) - 
                                 self.process_start_time).total_seconds()
        }

    def __str__(self) -> str:
        """返回预处理器的字符串表示"""
        return (
            f"DataPreprocessor(config={self.data_config['preprocessing']}, "
            f"scalers={list(self.scalers.keys())})"
        )

    def __repr__(self) -> str:
        """返回预处理器的详细表示"""
        return (
            f"DataPreprocessor(\n"
            f"    config={self.data_config},\n"
            f"    scalers={self.scalers},\n"
            f"    start_time={self.process_start_time}\n"
            f")"
        )

if __name__ == '__main__':
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("开始初始化数据预处理器...")
        
        # 初始化配置和预处理器
        config_path = project_root / 'config' / 'config.yaml'
        logger.info(f"加载配置文件: {config_path}")
        
        config = ConfigManager(config_path)
        preprocessor = DataPreprocessor(config)
        
        # 加载数据
        data_path = project_root / 'data' / 'raw' / 'steel_industry_data.csv'
        logger.info(f"加载数据文件: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        df = pd.read_csv(data_path)
        logger.info(f"数据加载完成，形状: {df.shape}")
        
        # 执行预处理
        logger.info("开始执行数据预处理...")
        processed_df = preprocessor.preprocess(df)
        
        # 获取处理摘要
        summary = preprocessor.get_preprocessing_summary()
        logger.info("预处理摘要: %s", summary)
        
        logger.info("数据预处理完成")
        
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        raise
"""
数据验证器：负责数据质量检查和验证
Current Date and Time (UTC): 2025-02-25 12:22:39
Current User: zlbbbb
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
import json
import sys
from dataclasses import dataclass
from scipy import stats

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 导入配置管理器
from config.config_manager import ConfigManager

@dataclass
class ValidationResult:
    """数据验证结果类"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    timestamp: str
    validated_by: str

class NumpyEncoder(json.JSONEncoder):
    """用于处理NumPy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif pd.isna(obj):
            return None
        return super().default(obj)

class DataValidator:
    """数据验证器类"""
    
    def __init__(self, config: Optional[ConfigManager] = None):
        """
        初始化数据验证器
        
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
            
            # 初始化验证规则
            self._initialize_validation_rules()
            
            self.logger.info(
                f"数据验证器初始化完成:\n"
                f"- 初始化时间: {self.init_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"- 当前用户: {self.current_user}"
            )
            
        except Exception as e:
            self.logger.error(f"数据验证器初始化失败: {str(e)}")
            raise
            
    def _initialize_validation_rules(self) -> None:
        """初始化验证规则"""
        # 必需列
        self.required_columns = {
            'date',
            'Usage_kWh', 
            'Lagging_Current_Reactive.Power_kVarh',
            'Leading_Current_Reactive_Power_kVarh',
            'CO2(tCO2)',
            'Lagging_Current_Power_Factor',
            'Leading_Current_Power_Factor',
            'Load_Type',
            'WeekStatus'
        }
        
        # 数值列及其取值范围
        self.numeric_ranges = {
            'Usage_kWh': (0,200),
            'Lagging_Current_Reactive.Power_kVarh': (0, 100),
            'Leading_Current_Reactive_Power_kVarh': (0, 30),
            'CO2(tCO2)': (0, 0.1),
            'Lagging_Current_Power_Factor': (0, 100),
            'Leading_Current_Power_Factor': (0, 100)
        }
        
        # 类别列及其有效值
        self.categorical_values = {
            'Load_Type': {'Light_Load', 'Medium_Load', 'Maximum_Load'},
            'WeekStatus': {'Weekday', 'Weekend'}
        }
        
        # 时间格式
        self.date_format = '%d/%m/%Y %H:%M'

    def validate_schema(self, df: pd.DataFrame) -> List[str]:
        """
        验证数据框架结构
        
        Args:
            df: 待验证的数据框
            
        Returns:
            List[str]: 错误消息列表
        """
        errors = []
        
        # 检查必需列
        missing_columns = self.required_columns - set(df.columns)
        if missing_columns:
            errors.append(f"缺少必需列: {missing_columns}")
            
        return errors

    def validate_data_types(self, df: pd.DataFrame) -> List[str]:
        """
        验证数据类型
        
        Args:
            df: 待验证的数据框
            
        Returns:
            List[str]: 错误消息列表
        """
        errors = []
        
        # 验证数值列
        for col in self.numeric_ranges.keys():
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"列 {col} 应为数值类型")
                
        # 验证类别列
        for col in self.categorical_values.keys():
            if col in df.columns and not pd.api.types.is_categorical_dtype(df[col]):
                if not set(df[col].unique()).issubset(self.categorical_values[col]):
                    errors.append(f"列 {col} 包含无效值")
                    
        # 验证日期列
        if 'date' in df.columns:
            try:
                pd.to_datetime(df['date'], format=self.date_format)
            except ValueError:
                errors.append("日期格式无效")
                
        return errors

    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证数值范围
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 错误消息列表和统计信息
        """
        errors = []
        stats_info = {}
        
        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col in df.columns:
                # 计算基本统计量
                stats = df[col].describe()
                stats_info[col] = {
                    'min': stats['min'],
                    'max': stats['max'],
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'null_count': df[col].isnull().sum(),
                    'out_of_range': ((df[col] < min_val) | (df[col] > max_val)).sum()
                }
                
                # 检查范围
                if stats_info[col]['out_of_range'] > 0:
                    errors.append(
                        f"列 {col} 有 {stats_info[col]['out_of_range']} 个值超出范围 "
                        f"[{min_val}, {max_val}]"
                    )
                
        return errors, stats_info

    def validate_categorical_values(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证类别值
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 错误消息列表和统计信息
        """
        errors = []
        stats_info = {}
        
        for col, valid_values in self.categorical_values.items():
            if col in df.columns:
                # 计算值分布
                value_counts = df[col].value_counts()
                invalid_values = set(df[col].unique()) - valid_values
                
                stats_info[col] = {
                    'value_counts': value_counts.to_dict(),
                    'null_count': df[col].isnull().sum(),
                    'invalid_values': list(invalid_values)
                }
                
                # 检查无效值
                if invalid_values:
                    errors.append(
                        f"列 {col} 包含无效值: {invalid_values}"
                    )
                
        return errors, stats_info

    def validate_temporal_consistency(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证时间一致性
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 错误消息列表和统计信息
        """
        errors = []
        stats_info = {}
        
        if 'date' in df.columns:
            try:
                #转换日期列
                dates = pd.to_datetime(df['date'], format=self.date_format)
                
                # 检查时间戳排序 时间戳本身即有序
                #is_sorted = dates.is_monotonic_increasing
                
                # 计算时间跨度和基本统计
                stats_info['temporal'] = {
                    'start_date': dates.min().strftime(self.date_format),
                    'end_date': dates.max().strftime(self.date_format),
                    'total_periods': len(dates),
                    'time_span_days': (dates.max() - dates.min()).days,
                    'unique_dates': len(dates.unique()),
                    'duplicated_dates': (dates.duplicated()).sum()
                }
                
                # 检查重复的时间戳
                if stats_info['temporal']['duplicated_dates'] > 0:
                    errors.append(
                        f"发现 {stats_info['temporal']['duplicated_dates']} 个重复的时间戳"
                    )
                    
            except ValueError as e:
                errors.append(f"时间格式无效: {str(e)}")
            except Exception as e:
                errors.append(f"时间一致性检查失败: {str(e)}")
                
        return errors, stats_info


    def detect_outliers(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        检测异常值
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 警告消息列表和统计信息
        """
        warnings = []
        stats_info = {}
        
        for col in self.numeric_ranges.keys():
            if col in df.columns:
                # 计算Z分数
                z_scores = np.abs(stats.zscore(df[col].dropna()))
                outliers = (z_scores > 3).sum()  # 3个标准差
                
                # IQR方法
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                iqr_outliers = ((df[col] < (Q1 - 1.5 * IQR)) | 
                               (df[col] > (Q3 + 1.5 * IQR))).sum()
                
                stats_info[col] = {
                    'z_score_outliers': int(outliers),
                    'iqr_outliers': int(iqr_outliers),
                    'Q1': Q1,
                    'Q3': Q3,
                    'IQR': IQR
                }
                
                if outliers > 0:
                    warnings.append(
                        f"列 {col} 使用Z分数方法检测到 {outliers} 个异常值"
                    )
                if iqr_outliers > 0:
                    warnings.append(
                        f"列 {col} 使用IQR方法检测到 {iqr_outliers} 个异常值"
                    )
                
        return warnings, stats_info

    def validate_correlations(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证特征相关性
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 警告消息列表和相关性矩阵
        """
        warnings = []
        numeric_cols = [col for col in self.numeric_ranges.keys() if col in df.columns]
        
        if len(numeric_cols) > 1:
            # 计算相关性矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 检查高相关性
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) > 0.9:  # 高相关性阈值
                        high_corr_pairs.append({
                            'feature1': numeric_cols[i],
                            'feature2': numeric_cols[j],
                            'correlation': corr
                        })
                        warnings.append(
                            f"特征 {numeric_cols[i]} 和 {numeric_cols[j]} "
                            f"存在高度相关性 ({corr:.2f})"
                        )
            
            return warnings, {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr_pairs
            }
            
        return warnings, {}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        执行完整的数据验证
        
        Args:
            df: 待验证的数据框
            
        Returns:
            ValidationResult: 验证结果
        """
        try:
            all_errors = []
            all_warnings = []
            all_stats = {}
            
            # 1. 架构验证
            schema_errors = self.validate_schema(df)
            all_errors.extend(schema_errors)
            
            if not schema_errors:  # 只有在架构验证通过后才继续
                # 2. 数据类型验证
                type_errors = self.validate_data_types(df)
                all_errors.extend(type_errors)
                
                # 3. 数值范围验证
                range_errors, range_stats = self.validate_value_ranges(df)
                all_errors.extend(range_errors)
                all_stats['value_ranges'] = range_stats
                
                # 4. 类别值验证
                cat_errors, cat_stats = self.validate_categorical_values(df)
                all_errors.extend(cat_errors)
                all_stats['categorical_values'] = cat_stats
                
                # 5. 时间一致性验证
                temp_errors, temp_stats = self.validate_temporal_consistency(df)
                all_errors.extend(temp_errors)
                all_stats['temporal'] = temp_stats
                
                # 6. 异常值检测
                outlier_warnings, outlier_stats = self.detect_outliers(df)
                all_warnings.extend(outlier_warnings)
                all_stats['outliers'] = outlier_stats
                
                # 7. 相关性分析
                corr_warnings, corr_stats = self.validate_correlations(df)
                all_warnings.extend(corr_warnings)
                all_stats['correlations'] = corr_stats
                
            # 汇总验证结果
            result = ValidationResult(
                is_valid=len(all_errors) == 0,
                errors=all_errors,
                warnings=all_warnings,
                statistics=all_stats,
                timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),
                validated_by=self.current_user
            )
            
            # 记录验证结果
            self._save_validation_result(result, df)
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            raise

    def _save_validation_result(self, result: ValidationResult, df: pd.DataFrame) -> None:
        """
        保存验证结果
        
        Args:
            result: 验证结果
            df: 验证的数据框
        """
        try:
            # 创建验证报告目录
            report_dir = project_root / 'reports' / 'validation'
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成报告文件名
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
            report_path = report_dir / f"validation_report_{timestamp}.json"
            
            # 准备报告内容
            report = {
                'validation_info': {
                    'timestamp': result.timestamp,
                    'validated_by': result.validated_by,
                    'is_valid': result.is_valid,
                    'data_shape': {
                        'rows': int(df.shape[0]),  # 确保转换为Python原生类型
                        'columns': int(df.shape[1])
                    }
                },
                'errors': result.errors,
                'warnings': result.warnings,
                'statistics': self._convert_stats_to_serializable(result.statistics)
            }
            
            # 使用自定义编码器保存报告
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
                
            self.logger.info(f"验证报告已保存: {report_path}")
            
        except Exception as e:
            self.logger.error(f"保存验证报告失败: {str(e)}")
            raise
#复写方法
    def _convert_stats_to_serializable(self, stats: Dict) -> Dict:
        """
        将统计信息转换为可序列化的格式
        
        Args:
            stats: 原始统计信息字典
            
        Returns:
            Dict: 转换后的统计信息字典
        """
        converted = {}
        
        for key, value in stats.items():
            if isinstance(value, dict):
                converted[key] = self._convert_stats_to_serializable(value)
            elif isinstance(value, (np.integer, np.floating)):
                converted[key] = value.item()
            elif isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, pd.Series):
                converted[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                converted[key] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                # 处理列表或元组中的每个元素
                converted[key] = [
                    item.item() if isinstance(item, (np.integer, np.floating))
                    else item for item in value
                ]
            elif hasattr(value, '__array__'):
                # 处理其他可能的numpy或pandas对象
                converted[key] = np.array(value).tolist()
            else:
                converted[key] = value
                
        return converted
    def validate_value_ranges(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证数值范围（确保返回可序列化的值）
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 错误消息列表和统计信息
        """
        errors = []
        stats_info = {}
        
        for col, (min_val, max_val) in self.numeric_ranges.items():
            if col in df.columns:
                # 计算基本统计量并确保转换为Python原生类型
                stats = df[col].describe()
                stats_info[col] = {
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'null_count': int(df[col].isnull().sum()),
                    'out_of_range': int(((df[col] < min_val) | (df[col] > max_val)).sum())
                }
                
                if stats_info[col]['out_of_range'] > 0:
                    errors.append(
                        f"列 {col} 有 {stats_info[col]['out_of_range']} 个值超出范围 "
                        f"[{min_val}, {max_val}]"
                    )
                
        return errors, stats_info

    def validate_correlations(self, df: pd.DataFrame) -> Tuple[List[str], Dict[str, Any]]:
        """
        验证特征相关性（确保返回可序列化的值）
        
        Args:
            df: 待验证的数据框
            
        Returns:
            Tuple[List[str], Dict[str, Any]]: 警告消息列表和相关性矩阵
        """
        warnings = []
        numeric_cols = [col for col in self.numeric_ranges.keys() if col in df.columns]
        
        if len(numeric_cols) > 1:
            # 计算相关性矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 检查高相关性
            high_corr_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = float(corr_matrix.iloc[i, j])  # 转换为Python float
                    if abs(corr) > 0.9:
                        high_corr_pairs.append({
                            'feature1': numeric_cols[i],
                            'feature2': numeric_cols[j],
                            'correlation': corr
                        })
                        warnings.append(
                            f"特征 {numeric_cols[i]} 和 {numeric_cols[j]} "
                            f"存在高度相关性 ({corr:.2f})"
                        )
            
            return warnings, {
                'correlation_matrix': corr_matrix.to_dict(),
                'high_correlations': high_corr_pairs
            }
            
        return warnings, {}

    def check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查缺失值
        
        Args:
            df: 待检查的数据框
            
        Returns:
            Dict[str, Any]: 缺失值统计信息
        """
        missing_stats = {}
        
        # 计算每列的缺失值
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100
        
        for column in df.columns:
            missing_stats[column] = {
                'count': int(missing_counts[column]),
                'percentage': float(missing_percentages[column])
            }
            
            if missing_counts[column] > 0:
                self.logger.warning(
                    f"列 {column} 包含 {missing_counts[column]} 个缺失值 "
                    f"({missing_percentages[column]:.2f}%)"
                )
                
        return missing_stats

    def check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        检查重复记录
        
        Args:
            df: 待检查的数据框
            
        Returns:
            Dict[str, Any]: 重复记录统计信息
        """
        # 完全重复
        full_duplicates = df.duplicated().sum()
        
        # 按时间列重复
        time_duplicates = 0
        if 'date' in df.columns:
            time_duplicates = df.duplicated(subset=['date']).sum()
        
        duplicate_stats = {
            'full_duplicates': int(full_duplicates),
            'time_duplicates': int(time_duplicates)
        }
        
        if full_duplicates > 0:
            self.logger.warning(f"发现 {full_duplicates} 条完全重复记录")
        if time_duplicates > 0:
            self.logger.warning(f"发现 {time_duplicates} 条时间重复记录")
            
        return duplicate_stats

    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        生成数据摘要报告
        
        Args:
            df: 数据框
            
        Returns:
            Dict[str, Any]: 摘要报告
        """
        try:
            report = {
                'basic_info': {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'data_types': df.dtypes.astype(str).to_dict()
                },
                'missing_values': self.check_missing_values(df),
                'duplicates': self.check_duplicates(df),
                'numeric_summary': {},
                'categorical_summary': {},
                'temporal_summary': {}
            }
            
            # 数值列摘要
            for col in self.numeric_ranges.keys():
                if col in df.columns:
                    stats = df[col].describe()
                    report['numeric_summary'][col] = {
                        'min': float(stats['min']),
                        'max': float(stats['max']),
                        'mean': float(stats['mean']),
                        'std': float(stats['std']),
                        'quartiles': {
                            '25%': float(stats['25%']),
                            '50%': float(stats['50%']),
                            '75%': float(stats['75%'])
                        }
                    }
            
            # 类别列摘要
            for col in self.categorical_values.keys():
                if col in df.columns:
                    value_counts = df[col].value_counts()
                    report['categorical_summary'][col] = {
                        'unique_values': len(value_counts),
                        'value_counts': value_counts.to_dict()
                    }
            
            # 时间列摘要
            if 'date' in df.columns:
                dates = pd.to_datetime(df['date'], format=self.date_format)
                report['temporal_summary'] = {
                    'start_date': dates.min().strftime(self.date_format),
                    'end_date': dates.max().strftime(self.date_format),
                    'time_span_days': (dates.max() - dates.min()).days,
                    'unique_dates': len(dates.unique()),
                    'duplicated_dates': int(dates.duplicated().sum())
                }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成摘要报告失败: {str(e)}")
            raise

if __name__ == '__main__':
    # 设置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 初始化验证器
        validator = DataValidator()
        
        # 加载示例数据
        data_path = project_root / 'data' / 'raw' / 'steel_industry_data.csv'
        if not data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
        df = pd.read_csv(data_path)
        
        # 执行验证
        validation_result = validator.validate(df)
        
        # 打印验证结果
        print("\n数据验证结果:")
        print(f"验证状态: {'通过' if validation_result.is_valid else '失败'}")
        
        if validation_result.errors:
            print("\n错误:")
            for error in validation_result.errors:
                print(f"- {error}")
                
        if validation_result.warnings:
            print("\n警告:")
            for warning in validation_result.warnings:
                print(f"- {warning}")
                
        # 生成并显示摘要报告
        summary_report = validator.generate_summary_report(df)
        print("\n数据摘要:")
        print(f"- 总行数: {summary_report['basic_info']['rows']}")
        print(f"- 总列数: {summary_report['basic_info']['columns']}")
        print(f"- 内存占用: {summary_report['basic_info']['memory_usage_mb']:.2f} MB")
        
    except Exception as e:
        logging.error(f"测试过程出错: {str(e)}")
        raise
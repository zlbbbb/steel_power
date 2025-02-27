"""
Main Script for Steel Power Prediction
Current Date and Time (UTC): 2025-02-27 08:55:16
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
from sklearn.model_selection import train_test_split
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
from src.train import Trainer

class ModelValidator:
    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型验证器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = setup_logger(
            "ModelValidator",
            level=config['training']['logging']['level']
        )
        
        # 设置设备
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and 
            config['training']['device'] == 'cuda' else 'cpu'
        )
        
        # 创建结果目录
        self.results_dir = Path(config['evaluation']['output_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化分析器
        self.analyzer = ModelAnalyzer(self.results_dir)
        
    def load_processed_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        加载预处理后的数据
        
        Returns:
            训练集、验证集和测试集
        """
        processed_data_path = Path(self.config['data']['paths']['final'])
        self.logger.info(f"加载处理后的数据: {processed_data_path}")
        
        try:
            # 尝试加载CSV文件
            data = pd.read_csv(processed_data_path / 'final_processed.csv')
            self.logger.info(f"成功加载数据，形状: {data.shape}")
            self.logger.info(f"可用列: {', '.join(data.columns)}")

            # 检查数据列并筛选可用特征
            available_features = []
            missing_features = []
            
            # 检查数值特征
            for feat in self.config['data']['features']['numerical']:
                if feat['name'] in data.columns:
                    available_features.append(feat['name'])
                else:
                    missing_features.append(feat['name'])
            
            # 检查分类特征
            if 'categorical' in self.config['data']['features']:
                for feat in self.config['data']['features']['categorical']:
                    if feat['name'] in data.columns:
                        available_features.append(feat['name'])
                    else:
                        missing_features.append(feat['name'])
            
            # 记录缺失特征
            if missing_features:
                self.logger.warning(f"以下特征在数据中不存在: {', '.join(missing_features)}")
                self.logger.info("将仅使用可用特征进行训练")
            
            # 确保至少有一些特征可用
            if not available_features:
                raise ValueError("没有找到任何可用的特征列")
            
            # 提取特征和目标变量
            X = data[available_features].values
            
            # 确保目标变量存在
            if 'Usage_kWh' not in data.columns:
                raise ValueError("找不到目标变量 'Usage_kWh'")
            
            y = data['Usage_kWh'].values
            
            # 数据划分
            train_ratio = self.config['data']['split']['train_ratio']
            val_ratio = self.config['data']['split']['val_ratio']
            
            # 首先划分训练集和临时测试集
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                train_size=train_ratio,
                random_state=self.config['training']['seed']
            )
            
            # 然后将临时测试集划分为验证集和测试集
            val_ratio_adjusted = val_ratio / (1 - train_ratio)
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                train_size=val_ratio_adjusted,
                random_state=self.config['training']['seed']
            )
            
            # 保存特征信息
            self.feature_info = {
                'available_features': available_features,
                'missing_features': missing_features,
                'feature_dim': len(available_features)
            }
            
            # 组合数据
            train_data = self._combine_features_target(X_train, y_train)
            val_data = self._combine_features_target(X_val, y_val)
            test_data = self._combine_features_target(X_test, y_test)
            
            self.logger.info(f"数据划分完成 - 训练集: {len(train_data)}, "
                           f"验证集: {len(val_data)}, 测试集: {len(test_data)}")
            
            # 保存数据集信息
            self._save_dataset_info(train_data, val_data, test_data)
            
            return train_data, val_data, test_data
            
        except FileNotFoundError:
            self.logger.error(f"找不到数据文件: {processed_data_path / 'processed_data.csv'}")
            self.logger.info("请确保数据预处理步骤已完成，并且数据文件存在于正确位置。")
            raise
        except Exception as e:
            self.logger.error(f"加载数据时出错: {str(e)}")
            raise
    def _save_dataset_info(self, 
                          train_data: np.ndarray, 
                          val_data: np.ndarray, 
                          test_data: np.ndarray):
        """保存数据集信息"""
        info = {
            'dataset_sizes': {
                'train': len(train_data),
                'validation': len(val_data),
                'test': len(test_data)
            },
            'feature_info': self.feature_info,
            'data_shape': {
                'train': train_data.shape,
                'validation': val_data.shape,
                'test': test_data.shape
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 保存信息到文件
        info_path = self.results_dir / 'dataset_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4)
            
        self.logger.info(f"数据集信息已保存至: {info_path}")

    def _combine_features_target(self, 
                               features: np.ndarray, 
                               target: np.ndarray = None) -> np.ndarray:
        """
        将特征和目标变量组合成所需的格式
        
        Args:
            features: 特征数组
            target: 目标变量数组
            
        Returns:
            组合后的数组
        """
        if target is not None:
            # 根据环境的需求组合数据
            # 这里需要根据具体的环境实现来调整
            combined = np.column_stack((features, target))
            return combined
        return features
        
    def create_environments(self, 
                          train_data: np.ndarray, 
                          val_data: np.ndarray, 
                          test_data: np.ndarray) -> Tuple[SteelPowerEnv, SteelPowerEnv, SteelPowerEnv]:
        """创建环境实例"""
        env_config = self.config['environment']
        
        # 更新环境配置以匹配实际特征维度
        env_config['state_dim'] = self.feature_info['feature_dim']
        
        # 创建环境
        train_env = SteelPowerEnv({
            'state_dim': env_config['state_dim'],
            'action_dim': env_config['action_dim'],
            'power': env_config['power'],
            'rewards': env_config['rewards'],
            'data': train_data
        })
        
        val_env = SteelPowerEnv({
            'state_dim': env_config['state_dim'],
            'action_dim': env_config['action_dim'],
            'power': env_config['power'],
            'rewards': env_config['rewards'],
            'data': val_data
        })
        
        test_env = SteelPowerEnv({
            'state_dim': env_config['state_dim'],
            'action_dim': env_config['action_dim'],
            'power': env_config['power'],
            'rewards': env_config['rewards'],
            'data': test_data
        })
        
        return train_env, val_env, test_env

        
    def validate_model(self, 
                      model: DQNAgent, 
                      env: SteelPowerEnv, 
                      phase: str) -> Dict[str, float]:
        """
        在指定环境上验证模型
        
        Args:
            model: 待验证的模型
            env: 验证环境
            phase: 验证阶段名称（'validation' 或 'test'）
        
        Returns:
            包含验证指标的字典
        """
        self.logger.info(f"开始{phase}阶段验证...")
        model.eval()
        
        metrics = {
            'total_reward': 0,
            'predictions': [],
            'actual_values': [],
            'steps': 0
        }
        
        episodes = self.config['evaluation']['num_episodes']
        
        with torch.no_grad():
            for episode in range(episodes):
                state, _ = env.reset()
                done = False
                episode_reward = 0
                
                while not done:
                    state_tensor = torch.FloatTensor(state).to(self.device)
                    action = model.select_action(state_tensor, evaluate=True)
                    next_state, reward, done, _, info = env.step(action)
                    
                    episode_reward += reward
                    metrics['steps'] += 1
                    
                    if 'power_value' in info and 'target_power' in info:
                        metrics['predictions'].append(info['power_value'])
                        metrics['actual_values'].append(info['target_power'])
                    
                    state = next_state
                    
                metrics['total_reward'] += episode_reward
                
        # 计算平均指标
        metrics['avg_reward'] = metrics['total_reward'] / episodes
        metrics['avg_steps'] = metrics['steps'] / episodes
        
        # 计算预测误差指标
        if metrics['predictions'] and metrics['actual_values']:
            predictions = np.array(metrics['predictions'])
            actual_values = np.array(metrics['actual_values'])
            metrics.update(self._calculate_error_metrics(predictions, actual_values))
        
        self.logger.info(f"{phase}阶段验证完成")
        return metrics
    
    def _calculate_error_metrics(self, 
                               predictions: np.ndarray, 
                               actual_values: np.ndarray) -> Dict[str, float]:
        """计算误差指标"""
        mse = np.mean((predictions - actual_values) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actual_values))
        mape = np.mean(np.abs((predictions - actual_values) / actual_values)) * 100
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape
        }
    
    def save_results(self, 
                    val_metrics: Dict[str, float], 
                    test_metrics: Dict[str, float]):
        """保存验证结果"""
        results = {
            'validation': val_metrics,
            'test': test_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config
        }
        
        # 保存结果
        results_path = self.results_dir / 'validation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4)
            
        self.logger.info(f"验证结果已保存至: {results_path}")
        
        # 生成验证报告
        self.analyzer.generate_validation_report(results)

def main():
    """主函数"""
    try:
        # 加载配置
        config_path = project_root / "config" / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 确保数据目录存在
        data_dir = project_root / "data" / "processed" / "final"
        if not data_dir.exists():
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"创建数据目录: {data_dir}")
        
        # 创建验证器
        validator = ModelValidator(config)
        
        # 加载和划分数据
        train_data, val_data, test_data = validator.load_processed_data()
        
        # 创建环境
        train_env, val_env, test_env = validator.create_environments(
            train_data, val_data, test_data
        )
        
        # 创建并训练模型
        trainer = Trainer(config)
        trainer.train(train_env)
        
        # 验证模型
        val_metrics = validator.validate_model(
            trainer.agent, val_env, phase='validation'
        )
        test_metrics = validator.validate_model(
            trainer.agent, test_env, phase='test'
        )
        
        # 保存结果
        validator.save_results(val_metrics, test_metrics)
        
        print("模型训练和验证完成！")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()
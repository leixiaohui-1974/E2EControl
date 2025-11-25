"""
配置管理模块
负责加载、验证和访问系统配置
"""

import yaml
import os
from typing import Any, Dict, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """配置错误异常"""
    pass


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._validate_config()
    
    def _load_config(self):
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise ConfigurationError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise ConfigurationError(f"加载配置文件失败: {e}")
    
    def _validate_config(self):
        """验证配置完整性和有效性"""
        required_sections = ['simulation', 'mpc', 'default_control', 'scenarios']
        
        for section in required_sections:
            if section not in self.config:
                raise ConfigurationError(f"缺少必需的配置节: {section}")
        
        # 验证数值范围
        sim = self.config['simulation']
        if sim['total_hours'] <= 0:
            raise ConfigurationError("total_hours 必须大于0")
        if sim['dt'] <= 0:
            raise ConfigurationError("dt 必须大于0")
        if sim['area'] <= 0:
            raise ConfigurationError("area 必须大于0")
        if sim['delay_steps'] < 0:
            raise ConfigurationError("delay_steps 不能为负")
        
        mpc = self.config['mpc']
        if mpc['horizon'] <= 0:
            raise ConfigurationError("horizon 必须大于0")
        if mpc['Z_min'] >= mpc['Z_max']:
            raise ConfigurationError("Z_min 必须小于 Z_max")
        
        # 验证场景配置
        if not self.config['scenarios']:
            raise ConfigurationError("至少需要定义一个场景")
        
        for scenario in self.config['scenarios']:
            if 'name' not in scenario or 'keywords' not in scenario or 'config' not in scenario:
                raise ConfigurationError(f"场景配置不完整: {scenario}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值（支持点分路径）
        
        Args:
            key_path: 配置键路径，如 "simulation.dt"
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        获取配置节
        
        Args:
            section: 节名称
            
        Returns:
            配置节字典
        """
        return self.config.get(section, {})
    
    def get_scenario_config(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """
        根据场景名称获取配置
        
        Args:
            scenario_name: 场景名称
            
        Returns:
            场景配置字典或None
        """
        for scenario in self.config['scenarios']:
            if scenario['name'] == scenario_name:
                return scenario['config']
        return None
    
    def get_all_scenarios(self) -> list:
        """获取所有场景定义"""
        return self.config.get('scenarios', [])
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._validate_config()
    
    def __repr__(self):
        return f"ConfigManager(config_path='{self.config_path}')"


# 全局配置实例（单例模式）
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: str = "config.yaml") -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        ConfigManager实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance


def reload_config():
    """重新加载全局配置"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()


if __name__ == "__main__":
    # 测试配置管理器
    try:
        config = get_config()
        print(f"✓ 配置加载成功")
        print(f"  仿真时长: {config.get('simulation.total_hours')} 小时")
        print(f"  MPC时域: {config.get('mpc.horizon')} 步")
        print(f"  场景数量: {len(config.get_all_scenarios())}")
        
        for scenario in config.get_all_scenarios():
            print(f"  - {scenario['name']}: {', '.join(scenario['keywords'])}")
            
    except ConfigurationError as e:
        print(f"✗ 配置错误: {e}")

"""
Python API 客户端
方便的API调用封装
"""

import requests
import time
import json
from typing import Dict, List, Optional, Any
from datetime import datetime


class SmartPoolAPIClient:
    """
    智能闸门API客户端
    
    使用示例:
        client = SmartPoolAPIClient("http://localhost:5000")
        
        # 解释指令
        result = client.interpret("收到暴雨预警")
        print(f"置信度: {result['confidence']}")
        
        # 运行仿真
        script = [[0, "保持水位平稳"]]
        sim_id = client.run_simulation(script, async_mode=True)
        
        # 等待完成
        result = client.wait_for_completion(sim_id)
        
        # 获取历史
        history = client.get_history(sim_id)
    """
    
    def __init__(self, base_url: str = "http://localhost:5000", timeout: int = 30):
        """
        初始化客户端
        
        Args:
            base_url: API服务器地址
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        发送HTTP请求
        
        Args:
            method: HTTP方法
            endpoint: API端点
            **kwargs: 其他requests参数
            
        Returns:
            响应JSON
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(
                method, url, timeout=self.timeout, **kwargs
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise APIError(f"请求失败: {e}")
    
    def health_check(self) -> Dict:
        """
        健康检查
        
        Returns:
            健康状态
        """
        return self._request('GET', '/health')
    
    def is_healthy(self) -> bool:
        """
        检查服务是否健康
        
        Returns:
            是否健康
        """
        try:
            result = self.health_check()
            return result.get('status') == 'healthy'
        except:
            return False
    
    def get_config(self) -> Dict:
        """
        获取系统配置
        
        Returns:
            配置信息
        """
        return self._request('GET', '/config')
    
    def interpret(self, instruction: str) -> Dict:
        """
        解释自然语言指令
        
        Args:
            instruction: 指令文本
            
        Returns:
            解释结果（包含confidence和config）
        """
        return self._request('POST', '/interpret', json={'instruction': instruction})
    
    def run_simulation(self, script: List[List], async_mode: bool = False) -> int:
        """
        运行仿真
        
        Args:
            script: 场景脚本 [[时间, 指令], ...]
            async_mode: 是否异步运行
            
        Returns:
            仿真ID
        """
        result = self._request(
            'POST', '/simulation/run',
            json={'script': script, 'async': async_mode}
        )
        
        if not result.get('success'):
            raise APIError(f"仿真启动失败: {result.get('error')}")
        
        return result['simulation_id']
    
    def get_status(self, sim_id: int) -> Dict:
        """
        获取仿真状态
        
        Args:
            sim_id: 仿真ID
            
        Returns:
            状态信息
        """
        return self._request('GET', f'/simulation/{sim_id}/status')
    
    def get_history(self, sim_id: int) -> List[Dict]:
        """
        获取仿真历史数据
        
        Args:
            sim_id: 仿真ID
            
        Returns:
            历史记录列表
        """
        result = self._request('GET', f'/simulation/{sim_id}/history')
        
        if not result.get('success'):
            raise APIError(f"获取历史失败: {result.get('error')}")
        
        return result['history']
    
    def get_alerts(self, sim_id: int) -> List[Dict]:
        """
        获取仿真告警记录
        
        Args:
            sim_id: 仿真ID
            
        Returns:
            告警列表
        """
        result = self._request('GET', f'/simulation/{sim_id}/alerts')
        
        if not result.get('success'):
            raise APIError(f"获取告警失败: {result.get('error')}")
        
        return result['alerts']
    
    def get_report(self, sim_id: int) -> str:
        """
        获取仿真报告
        
        Args:
            sim_id: 仿真ID
            
        Returns:
            报告文本
        """
        result = self._request('GET', f'/simulation/{sim_id}/report')
        
        if not result.get('success'):
            raise APIError(f"获取报告失败: {result.get('error')}")
        
        return result['report']
    
    def download_result_image(self, sim_id: int, save_path: str):
        """
        下载仿真结果图表
        
        Args:
            sim_id: 仿真ID
            save_path: 保存路径
        """
        url = f"{self.base_url}/simulation/{sim_id}/result.png"
        
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            with open(save_path, 'wb') as f:
                f.write(response.content)
        except requests.exceptions.RequestException as e:
            raise APIError(f"下载图表失败: {e}")
    
    def list_simulations(self) -> List[Dict]:
        """
        获取所有仿真列表
        
        Returns:
            仿真列表
        """
        result = self._request('GET', '/simulations')
        
        if not result.get('success'):
            raise APIError(f"获取仿真列表失败: {result.get('error')}")
        
        return result['simulations']
    
    def list_scenarios(self) -> List[Dict]:
        """
        获取所有场景定义
        
        Returns:
            场景列表
        """
        result = self._request('GET', '/scenarios')
        
        if not result.get('success'):
            raise APIError(f"获取场景列表失败: {result.get('error')}")
        
        return result['scenarios']
    
    def wait_for_completion(self, sim_id: int, 
                          interval: int = 2,
                          max_wait: int = 300,
                          callback: Optional[callable] = None) -> Dict:
        """
        等待仿真完成
        
        Args:
            sim_id: 仿真ID
            interval: 轮询间隔（秒）
            max_wait: 最大等待时间（秒）
            callback: 状态更新回调函数
            
        Returns:
            最终状态
        """
        start_time = time.time()
        
        while True:
            status = self.get_status(sim_id)
            current_status = status.get('status')
            
            if callback:
                callback(status)
            
            if current_status != 'running':
                return status
            
            if time.time() - start_time > max_wait:
                raise APIError(f"等待超时（超过{max_wait}秒）")
            
            time.sleep(interval)
    
    def close(self):
        """关闭会话"""
        self.session.close()


class APIError(Exception):
    """API错误"""
    pass


def demo():
    """演示客户端使用"""
    print("🌊 智能闸门API客户端演示\n")
    
    # 创建客户端
    client = SmartPoolAPIClient("http://localhost:5000")
    
    try:
        # 1. 健康检查
        print("1. 健康检查...")
        if client.is_healthy():
            print("   ✓ 服务正常\n")
        else:
            print("   ✗ 服务不可用")
            return
        
        # 2. 获取配置
        print("2. 获取配置...")
        config = client.get_config()
        sim_config = config['config']['simulation']
        print(f"   仿真时长: {sim_config['total_hours']}h")
        print(f"   时间步长: {sim_config['dt']}s\n")
        
        # 3. 解释指令
        print("3. 解释指令...")
        result = client.interpret("收到暴雨预警")
        print(f"   指令: {result['instruction']}")
        print(f"   置信度: {result['confidence']:.2f}")
        print(f"   目标水位: {result['config']['Z_ref']}m\n")
        
        # 4. 列出场景
        print("4. 获取场景列表...")
        scenarios = client.list_scenarios()
        print(f"   场景数量: {len(scenarios)}")
        for s in scenarios[:3]:
            print(f"   • {s['name']}")
        print()
        
        # 5. 异步运行仿真
        print("5. 运行异步仿真...")
        script = [[0, "保持水位平稳，正常供水。"]]
        sim_id = client.run_simulation(script, async_mode=True)
        print(f"   仿真ID: {sim_id}")
        print(f"   等待完成...", end='', flush=True)
        
        def progress_callback(status):
            print(".", end='', flush=True)
        
        final_status = client.wait_for_completion(sim_id, callback=progress_callback)
        print(f"\n   状态: {final_status['status']}\n")
        
        # 6. 获取历史数据
        print("6. 获取历史数据...")
        history = client.get_history(sim_id)
        print(f"   记录数: {len(history)}")
        if history:
            print(f"   最终水位: {history[-1]['level']:.2f}m\n")
        
        print("✓ 演示完成！")
        
    except APIError as e:
        print(f"\n✗ API错误: {e}")
    except Exception as e:
        print(f"\n✗ 错误: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    print("提示: 此演示需要API服务器运行")
    print("启动命令: python3 api.py\n")
    
    choice = input("API服务器是否已启动？(y/n): ").lower()
    if choice == 'y':
        demo()
    else:
        print("\n请先启动API服务器:")
        print("  python3 api.py")
        print("\n然后重新运行此脚本")

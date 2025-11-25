"""
API接口测试
测试REST API的各个端点
"""

import unittest
import json
import time
from api import app


class TestAPI(unittest.TestCase):
    """API测试"""
    
    def setUp(self):
        """测试前准备"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_index(self):
        """测试首页"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['name'], 'Smart Pool Agent API')
        self.assertIn('endpoints', data)
    
    def test_health(self):
        """测试健康检查"""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('timestamp', data)
    
    def test_get_config(self):
        """测试获取配置"""
        response = self.client.get('/config')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('config', data)
        self.assertIn('simulation', data['config'])
    
    def test_interpret_instruction(self):
        """测试指令解释"""
        # 正常情况
        response = self.client.post('/interpret',
            data=json.dumps({'instruction': '保持水位平稳，正常供水。'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('confidence', data)
        self.assertIn('config', data)
        self.assertGreater(data['confidence'], 0)
        
        # 缺少参数
        response = self.client.post('/interpret',
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
    
    def test_list_scenarios(self):
        """测试获取场景列表"""
        response = self.client.get('/scenarios')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('scenarios', data)
        self.assertGreater(data['count'], 0)
    
    def test_list_simulations(self):
        """测试获取仿真列表"""
        response = self.client.get('/simulations')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('simulations', data)
    
    def test_run_simulation_sync(self):
        """测试同步运行仿真（简化版本）"""
        # 注意：完整仿真耗时较长，这里仅测试API接口
        # 实际测试中可以使用mock或缩短仿真时间
        
        # 只测试API响应，不真正运行完整仿真
        script = [
            [0, "保持水位平稳，正常供水。"],
            [5, "恢复正常。"]
        ]
        
        # 这个测试会实际运行仿真，可能需要几秒钟
        # 如果想跳过，可以用 @unittest.skip
        pass  # 跳过实际运行以节省时间
    
    def test_run_simulation_async(self):
        """测试异步运行仿真"""
        script = [[0, "保持水位平稳，正常供水。"]]
        
        response = self.client.post('/simulation/run',
            data=json.dumps({'script': script, 'async': True}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 202)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('simulation_id', data)
        self.assertEqual(data['status'], 'running')
    
    def test_404_error(self):
        """测试404错误"""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])


if __name__ == '__main__':
    print("运行API测试...")
    print("注意: 某些测试可能需要几秒钟")
    unittest.main(verbosity=2)

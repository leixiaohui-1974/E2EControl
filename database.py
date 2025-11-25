"""
数据持久化模块
使用SQLite存储仿真数据
"""

import sqlite3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path
from logger import get_logger
from exceptions import DatabaseError


class SimulationDatabase:
    """仿真数据库"""
    
    def __init__(self, db_path: str = "simulation_data.db"):
        """
        初始化数据库
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.logger = get_logger()
        self.conn: Optional[sqlite3.Connection] = None
        
        self._init_database()
    
    def _init_database(self):
        """初始化数据库结构"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row  # 使用字典式访问
            
            cursor = self.conn.cursor()
            
            # 仿真会话表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    total_hours INTEGER,
                    dt REAL,
                    area REAL,
                    config TEXT,
                    notes TEXT
                )
            """)
            
            # 状态记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER,
                    time_step INTEGER,
                    timestamp REAL,
                    level REAL,
                    q_in REAL,
                    q_out REAL,
                    target_level REAL,
                    instruction TEXT,
                    config TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
                )
            """)
            
            # 告警记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER,
                    time_step INTEGER,
                    alert_time TIMESTAMP,
                    level TEXT,
                    alert_type TEXT,
                    message TEXT,
                    data TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
                )
            """)
            
            # 性能指标表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id INTEGER,
                    metric_name TEXT,
                    metric_value REAL,
                    unit TEXT,
                    FOREIGN KEY (simulation_id) REFERENCES simulations(id)
                )
            """)
            
            self.conn.commit()
            self.logger.info(f"数据库初始化完成: {self.db_path}")
            
        except sqlite3.Error as e:
            raise DatabaseError(f"数据库初始化失败: {e}")
    
    def create_simulation(self, total_hours: int, dt: float, area: float,
                         config: Dict, notes: str = "") -> int:
        """
        创建新的仿真会话
        
        Args:
            total_hours: 仿真总时长
            dt: 时间步长
            area: 渠池面积
            config: 配置信息
            notes: 备注
            
        Returns:
            仿真ID
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO simulations (start_time, total_hours, dt, area, config, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now(),
                total_hours,
                dt,
                area,
                json.dumps(config, ensure_ascii=False),
                notes
            ))
            self.conn.commit()
            
            sim_id = cursor.lastrowid
            self.logger.info(f"创建仿真会话: ID={sim_id}")
            return sim_id
            
        except sqlite3.Error as e:
            raise DatabaseError(f"创建仿真会话失败: {e}")
    
    def save_state(self, simulation_id: int, time_step: int, 
                   level: float, q_in: float, q_out: float,
                   target_level: float, instruction: str, config: Dict):
        """
        保存状态数据
        
        Args:
            simulation_id: 仿真ID
            time_step: 时间步
            level: 水位
            q_in: 入流
            q_out: 出流
            target_level: 目标水位
            instruction: 指令
            config: 配置
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO states 
                (simulation_id, time_step, timestamp, level, q_in, q_out, 
                 target_level, instruction, config)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_id,
                time_step,
                datetime.now().timestamp(),
                level,
                q_in,
                q_out,
                target_level,
                instruction,
                json.dumps(config, ensure_ascii=False)
            ))
            
            # 批量插入时不每次提交
            if time_step % 10 == 0:  # 每10步提交一次
                self.conn.commit()
                
        except sqlite3.Error as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def save_alert(self, simulation_id: int, time_step: int, 
                   level: str, alert_type: str, message: str, data: Dict):
        """
        保存告警记录
        
        Args:
            simulation_id: 仿真ID
            time_step: 时间步
            level: 告警级别
            alert_type: 告警类型
            message: 告警信息
            data: 附加数据
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO alerts 
                (simulation_id, time_step, alert_time, level, alert_type, message, data)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                simulation_id,
                time_step,
                datetime.now(),
                level,
                alert_type,
                message,
                json.dumps(data, ensure_ascii=False)
            ))
            self.conn.commit()
            
        except sqlite3.Error as e:
            self.logger.error(f"保存告警失败: {e}")
    
    def save_metric(self, simulation_id: int, metric_name: str, 
                   metric_value: float, unit: str = ""):
        """
        保存性能指标
        
        Args:
            simulation_id: 仿真ID
            metric_name: 指标名称
            metric_value: 指标值
            unit: 单位
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                INSERT INTO metrics (simulation_id, metric_name, metric_value, unit)
                VALUES (?, ?, ?, ?)
            """, (simulation_id, metric_name, metric_value, unit))
            self.conn.commit()
            
        except sqlite3.Error as e:
            self.logger.error(f"保存指标失败: {e}")
    
    def finish_simulation(self, simulation_id: int):
        """
        完成仿真会话
        
        Args:
            simulation_id: 仿真ID
        """
        try:
            self.conn.commit()  # 提交所有未提交的数据
            
            cursor = self.conn.cursor()
            cursor.execute("""
                UPDATE simulations SET end_time = ? WHERE id = ?
            """, (datetime.now(), simulation_id))
            self.conn.commit()
            
            self.logger.info(f"仿真会话完成: ID={simulation_id}")
            
        except sqlite3.Error as e:
            self.logger.error(f"完成仿真失败: {e}")
    
    def get_simulation_history(self, simulation_id: int) -> List[Dict]:
        """
        获取仿真历史数据
        
        Args:
            simulation_id: 仿真ID
            
        Returns:
            状态列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM states WHERE simulation_id = ? ORDER BY time_step
            """, (simulation_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            self.logger.error(f"获取历史数据失败: {e}")
            return []
    
    def get_simulation_alerts(self, simulation_id: int) -> List[Dict]:
        """
        获取仿真告警记录
        
        Args:
            simulation_id: 仿真ID
            
        Returns:
            告警列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts WHERE simulation_id = ? ORDER BY time_step
            """, (simulation_id,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            self.logger.error(f"获取告警记录失败: {e}")
            return []
    
    def get_recent_simulations(self, limit: int = 10) -> List[Dict]:
        """
        获取最近的仿真会话
        
        Args:
            limit: 返回数量
            
        Returns:
            仿真会话列表
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT * FROM simulations 
                ORDER BY start_time DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
            
        except sqlite3.Error as e:
            self.logger.error(f"获取仿真列表失败: {e}")
            return []
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            self.logger.info("数据库连接已关闭")


if __name__ == "__main__":
    # 测试数据库
    from logger import setup_logging
    
    setup_logging({'level': 'INFO', 'console_output': True})
    
    # 使用临时数据库
    db = SimulationDatabase("test.db")
    
    # 创建仿真
    sim_id = db.create_simulation(
        total_hours=50,
        dt=3600.0,
        area=10000.0,
        config={'test': True},
        notes="测试仿真"
    )
    
    # 保存一些状态
    for t in range(10):
        db.save_state(
            sim_id, t, 3.0 + t * 0.1, 5.0, 5.0, 3.0,
            "测试指令", {'Z_ref': 3.0}
        )
    
    # 保存告警
    db.save_alert(sim_id, 5, "WARNING", "测试告警", "这是一个测试", {})
    
    # 完成仿真
    db.finish_simulation(sim_id)
    
    # 读取数据
    history = db.get_simulation_history(sim_id)
    print(f"\n读取到 {len(history)} 条历史记录")
    
    db.close()
    
    # 清理测试文件
    import os
    os.remove("test.db")
    print("测试完成")

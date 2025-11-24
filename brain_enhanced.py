"""
增强版语义解释器
支持模糊匹配和相似度计算
"""

from typing import Dict, List, Tuple, Optional
from difflib import SequenceMatcher
import re
from config_manager import get_config
from logger import get_logger
from exceptions import SemanticError


class EnhancedSemanticInterpreter:
    """
    增强版语义解释器
    
    新功能：
    1. 模糊关键词匹配
    2. 多关键词组合匹配
    3. 相似度评分
    4. 匹配置信度报告
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        初始化解释器
        
        Args:
            similarity_threshold: 相似度阈值（0-1）
        """
        self.config = get_config()
        self.logger = get_logger()
        self.similarity_threshold = similarity_threshold
        
        # 加载场景配置
        self.scenarios = self.config.get_all_scenarios()
        self.default_config = self.config.get_section('default_control')
        
        self.logger.info(f"语义解释器初始化完成，加载 {len(self.scenarios)} 个场景")
    
    def interpret(self, instruction: str) -> Tuple[Dict, float]:
        """
        解释自然语言指令
        
        Args:
            instruction: 自然语言指令
            
        Returns:
            (配置字典, 置信度)
        """
        if not instruction or not instruction.strip():
            raise SemanticError("指令不能为空", instruction=instruction)
        
        instruction = instruction.strip()
        self.logger.debug(f"接收指令: {instruction}")
        
        # 1. 尝试精确匹配（历史兼容）
        for scenario in self.scenarios:
            if self._exact_match(instruction, scenario):
                config = self._merge_config(scenario['config'])
                self.logger.info(
                    f"精确匹配场景: {scenario['name']}",
                    scenario=scenario['name'],
                    confidence=1.0
                )
                return config, 1.0
        
        # 2. 模糊匹配
        best_match, best_score = self._fuzzy_match(instruction)
        
        if best_match and best_score >= self.similarity_threshold:
            config = self._merge_config(best_match['config'])
            self.logger.info(
                f"模糊匹配场景: {best_match['name']} (置信度: {best_score:.2f})",
                scenario=best_match['name'],
                confidence=best_score
            )
            return config, best_score
        
        # 3. 使用默认配置
        self.logger.warning(
            f"无法匹配场景，使用默认配置 (最高相似度: {best_score:.2f})",
            instruction=instruction,
            best_score=best_score
        )
        return self.default_config.copy(), 0.0
    
    def _exact_match(self, instruction: str, scenario: Dict) -> bool:
        """
        精确匹配检查
        
        Args:
            instruction: 指令
            scenario: 场景定义
            
        Returns:
            是否精确匹配
        """
        # 检查是否包含所有关键词
        keywords = scenario.get('keywords', [])
        if not keywords:
            return False
        
        # 构造完整字符串进行比较
        full_text = ' '.join(keywords)
        return instruction == full_text or instruction in keywords
    
    def _fuzzy_match(self, instruction: str) -> Tuple[Optional[Dict], float]:
        """
        模糊匹配
        
        Args:
            instruction: 指令
            
        Returns:
            (最佳匹配场景, 相似度分数)
        """
        best_scenario = None
        best_score = 0.0
        
        for scenario in self.scenarios:
            score = self._calculate_similarity(instruction, scenario)
            
            if score > best_score:
                best_score = score
                best_scenario = scenario
        
        return best_scenario, best_score
    
    def _calculate_similarity(self, instruction: str, scenario: Dict) -> float:
        """
        计算指令与场景的相似度
        
        Args:
            instruction: 指令
            scenario: 场景定义
            
        Returns:
            相似度分数 (0-1)
        """
        keywords = scenario.get('keywords', [])
        if not keywords:
            return 0.0
        
        # 方法1: 关键词命中率
        keyword_hits = sum(1 for kw in keywords if kw in instruction)
        keyword_score = keyword_hits / len(keywords)
        
        # 方法2: 字符串相似度（取最大值）
        string_scores = [
            SequenceMatcher(None, instruction, kw).ratio()
            for kw in keywords
        ]
        max_string_score = max(string_scores) if string_scores else 0.0
        
        # 方法3: 词语重叠度
        instruction_words = set(self._tokenize(instruction))
        keyword_words = set()
        for kw in keywords:
            keyword_words.update(self._tokenize(kw))
        
        if keyword_words:
            overlap_score = len(instruction_words & keyword_words) / len(keyword_words)
        else:
            overlap_score = 0.0
        
        # 综合评分（可调权重）
        final_score = (
            0.4 * keyword_score +
            0.3 * max_string_score +
            0.3 * overlap_score
        )
        
        self.logger.debug(
            f"相似度计算: {scenario['name']} = {final_score:.3f} "
            f"(kw:{keyword_score:.2f}, str:{max_string_score:.2f}, ovl:{overlap_score:.2f})"
        )
        
        return final_score
    
    def _tokenize(self, text: str) -> List[str]:
        """
        简单的中文分词（基于字符）
        
        Args:
            text: 文本
            
        Returns:
            词列表
        """
        # 移除标点符号
        text = re.sub(r'[，。！？、；：""''（）]', '', text)
        # 按字符分割（简单方法）
        # 更好的方法是使用jieba等分词库
        return list(text)
    
    def _merge_config(self, scenario_config: Dict) -> Dict:
        """
        合并场景配置和默认配置
        
        Args:
            scenario_config: 场景配置
            
        Returns:
            合并后的配置
        """
        config = self.default_config.copy()
        config.update(scenario_config)
        return config
    
    def list_scenarios(self) -> List[str]:
        """
        列出所有可用场景
        
        Returns:
            场景名称列表
        """
        return [s['name'] for s in self.scenarios]
    
    def get_scenario_details(self, scenario_name: str) -> Optional[Dict]:
        """
        获取场景详细信息
        
        Args:
            scenario_name: 场景名称
            
        Returns:
            场景详情
        """
        for scenario in self.scenarios:
            if scenario['name'] == scenario_name:
                return scenario
        return None


if __name__ == "__main__":
    # 测试增强版解释器
    from logger import setup_logging
    
    setup_logging({'level': 'DEBUG', 'console_output': True})
    
    interpreter = EnhancedSemanticInterpreter(similarity_threshold=0.5)
    
    test_cases = [
        "保持水位平稳，正常供水。",  # 精确匹配
        "保持水位平稳",  # 部分匹配
        "收到暴雨预警啦！",  # 模糊匹配
        "现在是冰期，不要动",  # 模糊匹配
        "发现污染",  # 模糊匹配
        "这是一条未知指令",  # 无匹配
    ]
    
    print("\n=== 语义解释测试 ===\n")
    for instruction in test_cases:
        config, confidence = interpreter.interpret(instruction)
        print(f"指令: {instruction}")
        print(f"置信度: {confidence:.2f}")
        print(f"目标水位: {config['Z_ref']}m")
        print(f"平滑权重: {config['W_smooth']}")
        print("-" * 50)

# 故障排查指南

## 安装问题

### Q: CVXPY安装失败

**现象：** `pip install cvxpy` 报错或编译失败

**解决方案：**

```bash
# 方案1：使用预编译包
pip install cvxpy --only-binary :all:

# 方案2：先安装编译工具
# Ubuntu/Debian
sudo apt-get install gcc g++ python3-dev

# macOS
xcode-select --install

# Windows
# 安装 Visual C++ Build Tools

# 方案3：使用conda
conda install -c conda-forge cvxpy
```

### Q: scikit-learn或PyTorch安装失败

**说明：** 这两个包是可选依赖，仅Phase 4的机器学习和深度学习检测器需要。系统可以在不安装它们的情况下运行核心功能。

```bash
# 仅安装核心依赖即可运行Phase 1-3
pip install numpy matplotlib cvxpy networkx
```

### Q: 依赖版本冲突

```bash
# 创建干净的虚拟环境
python -m venv venv_clean
source venv_clean/bin/activate
pip install -r requirements.txt
```

## 运行问题

### Q: 找不到模块（ModuleNotFoundError）

**现象：** `ModuleNotFoundError: No module named 'xxx'`

**解决方案：**

```bash
# 确保在项目根目录运行
cd /path/to/e2econtrol
python -m hydroe2e.main

# 如果使用子目录中的脚本，确保已安装包
pip install -e .
python -m hydroe2e.phase4.examples.ml_detection_demo
```

### Q: CVXPY求解失败（SolverError）

**现象：** `cvxpy.error.SolverError: Solver 'xxx' failed`

**解决方案：**

```bash
# 检查已安装的求解器
python -c "import cvxpy; print(cvxpy.installed_solvers())"

# 安装OSQP求解器（推荐）
pip install osqp

# 安装SCS求解器（备选）
pip install scs

# 安装CVXOPT求解器（备选）
pip install cvxopt
```

如果仍然失败，可能是优化问题本身不可行，检查约束条件是否合理。

### Q: 中文显示乱码

**现象：** Matplotlib图表中文显示为方块

**解决方案：**

```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-microhei-fonts

# macOS（通常自带中文字体）

# Windows（通常自带中文字体）
```

如果安装字体后仍乱码：

```python
# 在代码中手动设置字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
```

### Q: 内存不足

**现象：** 运行大规模仿真时内存溢出

**解决方案：**

编辑 `config.yaml` 减小仿真规模：

```yaml
simulation:
  total_hours: 24    # 缩短仿真时长
```

或者减少数字孪生的空间切片数量。

## API问题

### Q: API服务无法启动

**现象：** `Address already in use` 或端口被占用

```bash
# 检查端口占用
# Linux/macOS
lsof -i :5000

# Windows
netstat -ano | findstr :5000

# 使用其他端口
hydroe2e-api --port 8000
```

### Q: API请求超时

**现象：** 仿真请求长时间无响应

**解决方案：** 使用异步模式运行仿真：

```bash
curl -X POST http://localhost:5000/simulation/run \
  -H "Content-Type: application/json" \
  -d '{
    "script": [[0, "保持水位平稳"]],
    "async": true
  }'
```

然后轮询状态：

```bash
curl http://localhost:5000/simulation/{id}/status
```

### Q: 跨域请求被拒绝

**现象：** 浏览器报CORS错误

**说明：** API默认已启用CORS。如果仍遇到问题，检查 `hydroe2e/api.py` 中的CORS配置。

## Docker问题

### Q: Docker构建失败

```bash
# 清理缓存重新构建
docker build --no-cache -t e2econtrol:v1.0 .

# 检查Dockerfile语法
docker build --check .
```

### Q: 容器内无法访问

```bash
# 确认端口映射
docker ps
# 确认容器健康
docker logs <container_id>
```

## 数字孪生模块问题

### Q: 智能感知层导入失败

**现象：** `No module named 'physics.single_channel_fidelity'; 'physics' is not a package`

**原因：** `intelligent_observer.py` 中导入路径错误

**解决方案：** 将第14行的导入语句从：
```python
from physics.single_channel_fidelity import SingleChannelFidelity, PhysicalState, ChannelGeometry
```
修改为：
```python
from hydroe2e.digital_twin.physics.single_channel_fidelity import SingleChannelFidelity, PhysicalState, ChannelGeometry
```

### Q: 物理模型初始化失败

**现象：** `TypeError: __init__() got an unexpected keyword argument 'N'`

**原因：** `SingleChannelFidelity` 需要 `ChannelGeometry` 对象

**解决方案：**

```python
from hydroe2e.digital_twin.physics.single_channel_fidelity import SingleChannelFidelity, ChannelGeometry

geometry = ChannelGeometry(length=20000.0, N=20)
physics = SingleChannelFidelity(geometry=geometry)
```

## 测试问题

### Q: 部分测试被跳过

当前有4个模块在集成测试中被跳过，原因及处理方式：

1. **数字孪生物理模型** - 构造函数参数问题，需使用 `ChannelGeometry`
2. **智能感知层** - 导入路径错误，需修复导入语句
3. **异常检测** - 模块导入路径问题，实际功能可用
4. **故障诊断** - `diagnosis_engine.py` 文件缺失

详见项目 issue tracker。

## 性能问题

### Q: MPC求解速度慢

- 减小预测时域：`config.yaml` 中将 `horizon` 从10改为5
- 使用OSQP求解器（比SCS更快）
- 启用Warm-start

### Q: 仿真速度慢

- 减少空间切片数量
- 缩短仿真时长
- 关闭不必要的可视化输出

## 获取帮助

如果以上方案未能解决你的问题：

1. 查看各模块的 `README.md`
2. 查看代码中的注释和文档字符串
3. 在GitHub Issues中搜索或提交问题

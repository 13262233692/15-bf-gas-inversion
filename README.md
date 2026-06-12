# 高炉红外温度反演系统 (BF Gas Inversion)

重工业冶金高炉炉顶温度场实时监控系统，基于红外相机原始辐射数据进行温度反演与空间插值可视化。

## 项目概述

本系统攻克冶金高炉内部黑盒状态的实时监控难题，通过两条核心工业数据链实现炉顶温度场的高精度还原与可视化：

1. **红外相机原始辐射亮度流反演**：基于TCP端口并发接收工业红外摄像机的14-bit高密原始辐射能量矩阵（128×128像素，10Hz刷新率），根据斯特藩-玻尔兹曼定律与普朗克辐射定律，结合灰度吸收系数校准表，进行高度非线性的温度反演，在内存中还原出炉顶布料面高精摄氏度矩阵。

2. **基于RBF径向基函数的空间等温线网格重构**：反演出的离散温度场经过多维径向基函数（RBF）空间插值，对不规则边界漏点进行高精度平滑，前端利用D3.js动态绘制精细的炉顶二维等温线云图，清晰反映高炉管道中央及边缘的能量场分布。

## 技术栈

**后端**：
- Python 3.12+
- FastAPI - API服务框架
- NumPy - 数值计算
- SciPy - RBF插值算法
- Uvicorn - ASGI服务器

**前端**：
- D3.js v7 - 数据可视化
- Canvas - 热力图渲染
- WebSocket - 实时数据推送

## 项目结构

```
15-bf-gas-inversion/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI主入口
│   │   ├── config.py            # 系统配置
│   │   ├── tcp_server.py        # TCP数据接收服务器
│   │   ├── data_manager.py      # 共享数据管理器
│   │   ├── temperature_inversion.py  # 温度反演模块
│   │   ├── rbf_interpolation.py      # RBF空间插值模块
│   │   └── calib/               # 校准数据目录
│   │       └── absorption_coeffs.npy
│   ├── simulator.py             # 红外相机数据模拟器
│   ├── test_api.py              # API集成测试
│   └── requirements.txt         # Python依赖
├── frontend/
│   ├── index.html               # 主页面
│   ├── css/
│   │   └── style.css            # 样式文件
│   └── js/
│       └── app.js               # 前端可视化逻辑
├── start.bat                    # 服务端启动脚本
├── start_simulator.bat          # 模拟器启动脚本
└── test_modules.py              # 核心模块测试
```

## 快速开始

### 1. 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

### 2. 启动服务

```bash
# Windows
start.bat

# 或手动启动
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

服务启动后访问：
- Web UI: http://localhost:8000/
- API 文档: http://localhost:8000/docs
- TCP 端口: 8888

### 3. 启动数据模拟器

打开新终端：

```bash
# Windows
start_simulator.bat

# 或手动启动
cd backend
python simulator.py --fps 10
```

### 4. 前端操作

- 打开 Web UI 后，点击「启动模拟数据」按钮可直接在前端使用内置模拟数据
- 若启动了 TCP 模拟器，WebSocket 会自动接收实时温度数据

## 核心算法

### 温度反演

**普朗克辐射定律**（默认方法）：
```
L_λ = (c1 / λ^5) / (exp(c2 / (λT)) - 1)
```
其中 c1=1.191×10^8 W·μm^4/(m^2·sr), c2=14387.8 μm·K

**斯特藩-玻尔兹曼定律**：
```
M = σT^4
```
其中 σ=5.67×10^-8 W/(m^2·K^4)

### RBF径向基函数插值

支持的核函数：
- `multiquadric` (默认): φ(r) = √(r² + ε²)
- 其他可配置核函数: `inverse`, `gaussian`, `linear`, `cubic`, `quintic`, `thin_plate_spline`

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/stats` | 实时统计数据 |
| GET | `/api/raw-frame` | 获取原始帧数据 |
| GET | `/api/temperature-frame` | 获取反演温度矩阵 |
| GET | `/api/interpolated-temp` | 获取插值后温度矩阵 |
| GET | `/api/contours` | 获取等温线数据 |
| GET | `/api/calibration` | 获取吸收系数校准表 |
| POST | `/api/invert` | 手动温度反演 |
| POST | `/api/interpolate` | 手动RBF插值 |
| WS | `/ws/stream` | WebSocket实时数据流 |

## 性能指标

- 原始数据：128×128 @ 10Hz, 14-bit
- 插值输出：256×256 高分辨率温度场
- 端到端延迟：< 100ms
- 并发连接：支持多客户端同时接入

## 许可证

MIT License

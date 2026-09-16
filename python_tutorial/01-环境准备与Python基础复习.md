# 第 1 章 环境准备与Python基础复习
本教程以双全回转推进船舶 3DOF 建模项目为背景。

## 1.1 本章学习目标

学完本章，你将能够：

1. 检查本机 Python 版本，并创建独立的虚拟环境（venv 或 uv 两种方式）；
2. 安装本教程需要的第三方包（numpy、matplotlib、pandas、scipy，以及可选的 torch），并会运行一个 `.py` 脚本；
3. 快速复习 Python 标准库的核心语法：变量、字符串、列表、字典、循环、函数、列表推导式、文本文件读写；
4. 说出本项目的研究对象，认识全书统一使用的符号。

## 1.2 为什么学这一章

我们的项目要做这样一件事：先建立一艘"双推进器船"的计算机仿真模型，再用试航数据校准它，最后在仿真里训练强化学习控制器——这一整套工作都用 Python 完成。环境没装好，后面每章的代码都会卡在莫名其妙的报错上；基础语法生疏，学到 NumPy、PyTorch 时会分不清"是新库没学会，还是老语法忘了"。所以本章不引入任何第三方库，只用标准库把环境和语法一次夯实，且所有练习数据都来自船舶场景（转速采样、艏向角、方位角），让你从第一天起就熟悉项目的"语言"。

## 1.3 环境准备

### 1.3.1 确认 Python 版本

打开终端（Windows 可用 PowerShell），输入 `python3 --version`，看到类似 `Python 3.12.13` 的输出即可，本教程要求 **Python 3.10 或更高版本**。Windows 上有些机器要换成 `python --version`；若提示找不到命令，请先到 [python.org](https://www.python.org) 安装（勾选 "Add python.exe to PATH"）。

### 1.3.2 创建虚拟环境

**虚拟环境**（virtual environment）是一个独立的 Python 小房间：在里面安装的包不会污染系统 Python，也不会和其他项目的包版本打架。两种方式任选其一。

**方式一：标准库 venv（无需安装额外工具）**

```bash
python3 -m venv .venv        # 创建名为 .venv 的虚拟环境
source .venv/bin/activate    # 激活（Linux / macOS）
```

Windows PowerShell 的激活命令是 `.venv\Scripts\Activate.ps1`。激活成功后提示符前出现 `(.venv)`，之后的安装都只作用于这个环境；在 Debian/Ubuntu 上若报 `ensurepip is not available`，先 `sudo apt install python3-venv` 再重试。想退出时输入 `deactivate`。

**方式二：uv（更快的现代工具，推荐有条件者使用）**

[uv](https://docs.astral.sh/uv/) 是用 Rust 写的 Python 包管理工具，建环境和装包都快得多：

```bash
uv venv .venv                # 同样生成 .venv 目录
source .venv/bin/activate    # 激活方式相同
```

### 1.3.3 安装本教程所需第三方包

激活虚拟环境后，安装第 2~8 章要用的四个包（用 uv 则换成 `uv pip install ...`）：

```bash
python -m pip install numpy matplotlib pandas scipy
```

用途预告：**numpy** 负责高效数组计算（第 2 章），**matplotlib** 负责画图（第 5 章），**pandas** 处理表格状试航数据（第 6 章），**scipy** 提供最小二乘等优化工具（第 7 章）。

第 9~10 章还会用到深度学习框架 **PyTorch**。它体积较大且现在用不到，可以先不装；没有独立显卡的读者届时安装 CPU 版即可（`--index-url` 指定官方 CPU 专用源，避免拉来巨大的 GPU 版）：

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 1.3.4 运行 .py 脚本的方式

本教程的每个示例都是**完整可运行**的普通脚本。把代码保存成文件（如 `hello_ship.py`）：

```python
print("你好，双全回转推进船！")
```

在终端运行 `python hello_ship.py`，预期输出 `你好，双全回转推进船！`。注意：激活虚拟环境后 `python` 指向环境内的解释器；没激活时系统里可能只有 `python3` 命令，且找不到环境中安装的包。

### 1.3.5 可选：Jupyter 简述

Jupyter 是"边写边看结果"的网页式编程环境：代码切成一个个单元格，每格运行后立刻显示输出和图，适合第 5 章（画图）、第 6 章（看数据表）这类场合。用 `python -m pip install jupyterlab` 安装，终端输入 `jupyter lab` 启动。本教程示例都按普通 `.py` 脚本书写，粘进 Notebook 单元格同样能跑，用不用凭个人习惯。

## 1.4 Python 基础快速复习

### 1.4.1 变量、数值与 f-string

我们的船靠两台**全回转推进器**驱动——全回转推进器（azimuth thruster）是可以绕竖直轴 360° 旋转、从而改变推力方向的推进装置，所以这艘船不需要舵。推进器的**转速**（单位 rpm，转/分钟）是最常记录的量。另一个常用状态量是**艏向角** ψ（读作 "psi"）：船头指向与参考方向（如正北）之间的夹角。给变量赋值、做算术、打印报告：

```python
# 两台推进器的转速，单位 rpm（转/分钟）
n1 = 720.5   # 1 号推进器
n2 = 685.0   # 2 号推进器

n_avg = (n1 + n2) / 2    # 算术平均值；/ 是真除法，结果为浮点数 float
print(n_avg)

psi_deg = 35.2   # 艏向角，单位为度
# f-string：字符串前加 f，用 {表达式:格式} 嵌入变量
report = f"艏向角 psi = {psi_deg:.1f} 度，平均转速 = {n_avg:.1f} rpm"
print(report)
```

预期输出：

```text
702.75
艏向角 psi = 35.2 度，平均转速 = 702.8 rpm
```

赋值为带小数的数就是浮点数；`:.1f` 表示"保留 1 位小数"（`702.75` 四舍五入为 `702.8`），`:.0f` 表示取整。

### 1.4.2 循环与列表

试航时计算机会按固定间隔**采样**（每隔一小段时间记录一次），这个间隔记作 `dt`。用 `for` 循环生成采样时刻，再把采到的一串转速存进**列表**（list，有序的一串值）求平均：

```python
dt = 0.5   # 采样时间间隔，单位秒

for k in range(3):
    print(f"第 {k} 个采样点，时刻 t = {k * dt:.1f} 秒")

# 一次试验中采到的 6 个转速值（rpm）
samples = [698.2, 701.5, 700.1, 699.8, 702.3, 700.9]

total = 0.0
for s in samples:
    total = total + s

print(f"共 {len(samples)} 个采样，平均转速 = {total / len(samples):.2f} rpm")
```

预期输出：

```text
第 0 个采样点，时刻 t = 0.0 秒
第 1 个采样点，时刻 t = 0.5 秒
第 2 个采样点，时刻 t = 1.0 秒
共 6 个采样，平均转速 = 700.47 rpm
```

`range(3)` 产生 0, 1, 2——Python 的计数**从 0 开始**，"第 `k` 个采样点时刻为 `k * dt`"是全书仿真代码反复出现的写法。`len()` 返回列表长度；"遍历一列数算统计量"在试航分析里极常见，第 6 章会用 pandas 做得更快。

### 1.4.3 字典

**字典**（dict）用"键 → 值"存数据，适合给一组有名字的量打包，比如某时刻的船舶状态：

```python
state = {"x": 12.5, "y": -3.0, "psi": 35.2}   # 位置 x, y（米）与艏向角 psi（度）

state["psi"] = 36.0     # 修改已有键（state["psi"] 是按键读取）
state["u"] = 1.5        # 新增一个键：纵荡速度 u（米/秒）

for key, value in state.items():
    print(f"{key} = {value}")
```

预期输出：

```text
x = 12.5
y = -3.0
psi = 36.0
u = 1.5
```

对已有键赋值是修改，对新键赋值是新增；`.items()` 同时遍历键和值。这里 `u` 是**纵荡**速度——纵荡指船沿船长方向（前后）的运动。第 8 章会把这样的状态字典升级成仿真环境类。

### 1.4.4 函数：默认参数与多返回值

函数把可复用的计算打包。下面两个函数分别演示**默认参数**和**一次返回多个值**：

```python
def estimate_thrust(n, k=0.8):
    """教学简化：假设推力 T 与转速 n 的平方成正比。

    参数 n：转速（rpm）；参数 k：比例系数，默认 0.8。
    """
    return k * (n / 100.0) ** 2

print(f"k=0.8 时 T = {estimate_thrust(700.0):.2f}")          # 使用默认 k
print(f"k=1.0 时 T = {estimate_thrust(700.0, k=1.0):.2f}")   # 显式指定 k

def summarize(samples):
    """返回采样的 (平均值, 最大值, 最小值) 三个值。"""
    mean = sum(samples) / len(samples)
    return mean, max(samples), min(samples)

data = [698.2, 701.5, 700.1, 699.8, 702.3, 700.9]
mean_n, max_n, min_n = summarize(data)   # 一次接住三个返回值
print(f"平均 {mean_n:.2f}，最大 {max_n:.2f}，最小 {min_n:.2f}")
```

预期输出：

```text
k=0.8 时 T = 39.20
k=1.0 时 T = 49.00
平均 700.47，最大 702.30，最小 698.20
```

`estimate_thrust` 估计的是推进器的**推力**（推进器向后推水、水反过来推船前进的力）；`k=0.8` 是默认参数，不传就用 0.8。真实的推力–转速关系要在湖试中辨识（本项目的核心问题之一），这里的平方关系只是**教学简化**，数值没有物理意义。`summarize` 一次返回三个值（本质是元组拆包）——第 8、10 章环境的 `step()` 也是这样一次返回"新状态、奖励、是否结束"的。

### 1.4.5 列表推导式

"对列表里每个元素做同一种变换得到新列表"极其常见，列表推导式一行写完：

```python
samples = [698.2, 701.5, 700.1, 699.8, 702.3, 700.9]

thrusts = [0.8 * (n / 100.0) ** 2 for n in samples]   # 沿用教学简化模型 k=0.8

print([round(t, 2) for t in thrusts])
```

预期输出：

```text
[39.0, 39.37, 39.21, 39.18, 39.46, 39.3]
```

`[表达式 for n in samples]` 等价于一个 append 循环，但更紧凑。第 2 章 NumPy 会把这种"整列同算"做得更快更简洁，思想相同。

### 1.4.6 用 with open 读写文本文件

试航数据常以 **CSV**（逗号分隔值：每行一条记录、字段用逗号隔开的纯文本）保存。先把 4 个 (t, n1, n2) 采样点写成 CSV：

```python
samples = [(0.0, 700.1, 698.2), (0.1, 700.5, 698.0),
           (0.2, 699.8, 698.5), (0.3, 700.2, 697.9)]

with open("rpm_log.csv", "w", encoding="utf-8") as f:
    f.write("t,n1,n2\n")                    # 先写一行表头
    for t, n1, n2 in samples:
        f.write(f"{t:.1f},{n1:.2f},{n2:.2f}\n")

print("已写入 rpm_log.csv")
```

运行后脚本所在目录会生成 `rpm_log.csv`（表头加 4 行数据，可用 `cat rpm_log.csv` 查看）。`"w"` 表示写入（覆盖同名旧文件）。`with` 块结束时文件**自动关闭**——这很重要：不关闭的文件，内容可能还留在内存缓冲里没落盘。`\n` 是换行符。

再读回来求平均：

```python
n1_values = []

with open("rpm_log.csv", "r", encoding="utf-8") as f:
    header = f.readline()          # 先读掉表头行
    for line in f:
        t_str, n1_str, n2_str = line.strip().split(",")
        n1_values.append(float(n1_str))

print(n1_values)
print(f"1 号机平均转速 = {sum(n1_values) / len(n1_values):.2f} rpm")
```

预期输出：

```text
[700.1, 700.5, 699.8, 700.2]
1 号机平均转速 = 700.15 rpm
```

从文件读到的每行都是**字符串**：`strip()` 去行尾换行，`split(",")` 切成三段，数字要用 `float()` 转换。这套"读文本 → 切分 → 转数字"是数据处理的起点，第 6 章 pandas 的 `read_csv` 会把三步合成一行。

## 1.5 综合示例：最小采样记录器

把本章知识串起来：模拟生成 1 秒钟双机转速、写成 CSV、读回求平均。保存为 `rpm_logger.py` 运行：

```python
import math   # 标准库：数学函数

# 第 1 步：生成模拟采样
# 教学简化：假设两台推进器的转速在 700/698 rpm 附近按正弦规律波动
dt = 0.1                       # 采样间隔，单位秒
records = []                   # 每个元素是 (t, n1, n2)

for k in range(10):
    t = k * dt
    n1 = 700.0 + 5.0 * math.sin(t)
    n2 = 698.0 + 4.0 * math.cos(t)
    records.append((t, n1, n2))

# 第 2 步：写成 CSV 文本文件
with open("rpm_log.csv", "w", encoding="utf-8") as f:
    f.write("t,n1,n2\n")
    for t, n1, n2 in records:
        f.write(f"{t:.1f},{n1:.2f},{n2:.2f}\n")

# 第 3 步：读回文件，统计 1 号机平均转速
n1_values = []
with open("rpm_log.csv", "r", encoding="utf-8") as f:
    f.readline()               # 跳过表头
    for line in f:
        t_str, n1_str, n2_str = line.strip().split(",")
        n1_values.append(float(n1_str))

mean_n1 = sum(n1_values) / len(n1_values)
print(f"共读取 {len(n1_values)} 个采样点")
print(f"1 号推进器平均转速 = {mean_n1:.2f} rpm")
```

预期输出：

```text
共读取 10 个采样点
1 号推进器平均转速 = 702.09 rpm
```

这个"生成数据 → 存盘 → 读回 → 统计"的骨架就是后续数据流水线的雏形：第 4 章用仿真替代第 1 步的虚构数据，第 6 章用 pandas 替代第 3 步的手工解析。

## 1.6 全书约定

### 1.6.1 项目背景（一段话）

本项目研究一艘由**两台全回转推进器**驱动、无舵、无首侧推的船舶：船上能连续量测的只有两台推进器的转速 `n1, n2` 和方位角 `alpha1, alpha2`（推进器推力指向的角度），而推进器的精确推力特性未知。目标是在这样的信息条件下，建立水平面 **3DOF**（三自由度：只考虑船在水平面内的三种运动——纵荡是前后平移、横荡是左右平移、艏摇是船头左右转动）的"灰箱"仿真模型（物理结构与数据驱动相结合、部分参数已知部分待辨识），用它作为训练强化学习控制器的环境，服务于动力定位（不靠锚、靠推进器把船自动保持在指定位置）、低速靠泊等任务。

### 1.6.2 全书统一符号表

后续章节统一使用以下符号，代码变量名与之一一对应：

| 符号 | 代码变量 | 含义 | 常用单位 |
|---|---|---|---|
| η = (x, y, ψ) | `x, y, psi` | 位置与艏向角（合称"位姿"） | m, m, rad |
| ν = (u, v, r) | `u, v, r` | 纵荡速度、横荡速度、艏摇角速度 | m/s, m/s, rad/s |
| n₁, n₂ | `n1, n2` | 1/2 号推进器转速 | rpm |
| α₁, α₂ | `alpha1, alpha2` | 1/2 号推进器方位角 | rad |
| X, Y, N | `X, Y, N` | 纵向力、横向力、艏摇力矩 | N, N, N·m |
| Δt | `dt` | 时间步长 / 采样间隔 | s |

> 硬性约定：代码里角度一律用弧度（rad）参与计算，展示时再换算成度，第 3 章起会反复强调。

### 1.6.3 学习地图

全书共 11 章，前 8 章打基础，后 3 章进入神经网络与强化学习：

| 章 | 标题 | 你将做的事 |
|---|---|---|
| 01 | 环境准备与Python基础复习 | （本章）装环境，复习标准库 |
| 02 | NumPy数组与向量化 | 用数组表示 η、ν，告别手写循环 |
| 03 | 坐标系与矩阵变换 | 在地面与船体坐标系之间换算 |
| 04 | 数值积分与船舶运动仿真 | 从受力积分出航迹 |
| 05 | Matplotlib可视化 | 画航迹图与转速曲线 |
| 06 | pandas处理试航数据 | 加载、清洗、统计试航 CSV |
| 07 | 最小二乘与参数辨识 | 用数据反推未知参数 |
| 08 | 用类构建仿真环境 | 封装 RL 可用的环境类 |
| 09 | PyTorch神经网络入门 | 用网络拟合未知映射 |
| 10 | 强化学习实战入门 | 训练会控制船的智能体 |
| 11 | 综合项目：端到端流水线 | 串起 1~10 章的完整工作流 |

## 1.7 常见错误

**错误 1：没激活虚拟环境就 pip install。** 症状是脚本里 `import numpy` 报 `ModuleNotFoundError`——包装进了系统 Python，脚本却跑在环境里的解释器（或反之）。务必先激活（提示符前出现 `(.venv)`）再安装：

```bash
source .venv/bin/activate     # 先激活虚拟环境
python -m pip install numpy   # 再安装包
```

**错误 2：f-string 忘记写 f 前缀。** 没有 `f` 的字符串不做替换，`{...}` 会被当成普通文字原样打印。

```python
mean_n1 = 700.15
print("1 号机平均转速 = {mean_n1:.2f} rpm")    # 错误：原样输出大括号
print(f"1 号机平均转速 = {mean_n1:.2f} rpm")   # 正确：输出 700.15
```

**错误 3：列表索引从 1 开始数。** Python 索引从 0 开始，长度为 3 的列表合法下标只有 0、1、2。

```python
samples = [698.2, 701.5, 700.1]
print(samples[1])     # 想取"第一个"，实际输出 701.5（第二个）
print(samples[3])     # 错误：IndexError: list index out of range
print(samples[0])     # 正确：输出 698.2
```

**错误 4：写文件不用 with、忘记关闭。** 缓冲区内容可能没落盘，文件是空的或不完整；`with` 块结束会自动关闭，即使中途报错也安全。

```python
f = open("rpm_log.csv", "w", encoding="utf-8")   # 错误示范
f.write("t,n1,n2\n")
with open("rpm_log.csv", "w", encoding="utf-8") as f:   # 正确写法
    f.write("t,n1,n2\n")
```

## 1.8 动手练习

1. **方位角统计**：1 号推进器方位角的 5 次采样为 `[30.0, 29.5, 30.2, 31.0, 30.8]`（度）。计算并打印平均方位角和"最大波动"（最大值减最小值）。
2. **单位换算函数**：写一个函数 `rpm_to_hz(n)`，把转速从 rpm 换算成 Hz（转/秒，即除以 60），再用列表推导式把 `[600.0, 660.0, 720.0]` 整体换算成 Hz 列表并打印。
3. **角度存盘**：艏向角采样为 `[0.0, 15.0, 30.0, 45.0]`（度）。用 `math.radians()` 换算成弧度，写成两列 CSV（表头 `psi_deg,psi_rad`，弧度保留 4 位小数），再把整个文件读回原样打印。

## 1.9 本章小结与下一章预告

本章完成了三件打地基的事：① 装好并激活虚拟环境，装齐 numpy、matplotlib、pandas、scipy，知道 torch CPU 版如何安装；② 用船舶数据场景复习了标准库核心语法——变量、f-string、列表/字典、循环、函数（默认参数、多返回值）、列表推导式、`with open` 读写 CSV；③ 了解了项目背景、全书符号约定（η、ν、n1/n2、alpha1/alpha2、X/Y/N、dt）与 11 章学习地图。

下一章 **《第 2 章 NumPy数组与向量化》** 将回答一个问题：手写循环一次只能算一个数，而仿真每一步都要对 η、ν 这样的"一组数"整体运算——NumPy 数组如何让"整列同算"既简洁又快上百倍。

## 附：动手练习参考答案

**练习 1**

```python
alpha_samples = [30.0, 29.5, 30.2, 31.0, 30.8]   # 方位角 alpha1 的 5 次采样（度）

mean_a = sum(alpha_samples) / len(alpha_samples)
swing = max(alpha_samples) - min(alpha_samples)

print(f"平均方位角 = {mean_a:.2f} 度")
print(f"最大波动 = {swing:.2f} 度")
```

输出：

```text
平均方位角 = 30.30 度
最大波动 = 1.50 度
```

**练习 2**

```python
def rpm_to_hz(n):
    """把转速从 rpm（转/分钟）换算成 Hz（转/秒）。"""
    return n / 60.0

rpm_list = [600.0, 660.0, 720.0]
hz_list = [rpm_to_hz(n) for n in rpm_list]
print(hz_list)
```

输出：`[10.0, 11.0, 12.0]`

**练习 3**

```python
import math

psi_samples = [0.0, 15.0, 30.0, 45.0]   # 艏向角采样（度）

with open("psi_log.csv", "w", encoding="utf-8") as f:
    f.write("psi_deg,psi_rad\n")
    for psi in psi_samples:
        f.write(f"{psi:.1f},{math.radians(psi):.4f}\n")

with open("psi_log.csv", "r", encoding="utf-8") as f:
    print(f.read())
```

输出：

```text
psi_deg,psi_rad
0.0,0.0000
15.0,0.2618
30.0,0.5236
45.0,0.7854
```

# 第 5 章 Matplotlib可视化

本教程以双全回转推进船舶 3DOF 建模项目为背景，本章教你用 Matplotlib 把仿真和试航数据画成图。

## 本章学习目标

学完本章，你将能够：

1. 说清 Figure（画布）与 Axes（坐标区）的关系，会用 `fig, ax = plt.subplots()` 创建图；
2. 用 `plot` 画时间序列：多条曲线、图例、轴标签、标题、网格、线宽颜色；
3. 用 `subplots` 把多张时间序列图排成子图布局；
4. 画等比例的 x-y 航迹图，并用 `quiver` 在轨迹上抽稀画出艏向箭头；
5. 用 `twinx` 画双 y 轴图，用 `savefig` 把图保存成图片文件，并解决中文乱码与负号方块问题。

## 为什么学这一章

我们的项目最终要回答的问题——"模型辨识准不准""RL 控制器把船开得稳不稳"——几乎都要靠图来回答：艏向角随时间的曲线、x-y 平面里的航迹、推进器转速和方位角的响应。研究文档《03-试航设计与数据需求》里列了 M01–M07 一系列试航机动（例如 M07 矢量 Z 形、M02 回零衰减），每一次机动的数据拿回来，第一件事就是画图检查通道对不对、响应合不合理。不会画图，后面的参数辨识（第 7 章）和 RL 训练（第 10 章）就是盲人摸象。本章假设你已经学过第 2 章的 NumPy 数组，所有数据都用 NumPy 现场造出来，自给自足。

## 5.1 Figure 与 Axes：画布和坐标区

Matplotlib 里最重要的两个概念：

- **Figure**：整张图，相当于一张画布；
- **Axes**：画布上的一块"坐标区"，有 x 轴、y 轴、曲线。一张 Figure 可以放多个 Axes（子图）。

推荐永远用 `fig, ax = plt.subplots()` 显式地创建它们，然后调用 `ax.plot()`、`ax.set_xlabel()` 这样的方法画图。先画一张最简单的图热身——想象一次"回零衰减"试验：船停下来后，艏摇角速度 r（船头左右转动的快慢，单位 deg/s）按指数规律慢慢衰减到 0：

```python
import numpy as np
import matplotlib.pyplot as plt

# 造一条指数衰减曲线：想象船停下来后，艏摇角速度 r 慢慢归零
t = np.arange(0.0, 10.0, 0.1)
r = 6.0 * np.exp(-t / 3.0)

fig, ax = plt.subplots(figsize=(7, 4))   # fig 是整张画布，ax 是画布上的一个坐标区
ax.plot(t, r)                            # 在 ax 上画线
ax.set_xlabel("time t (s)")
ax.set_ylabel("yaw rate r (deg/s)")
ax.set_title("M02 decay test (toy data)")
plt.show()
print("r[0] =", r[0], ", r[-1] =", round(r[-1], 3))
```

运行后弹出（或保存出）一张图：一条从 6 开始、按 e^(-t/3) 衰减的曲线，控制台输出：

```
r[0] = 6.0 , r[-1] = 0.221
```

注意三个细节：`figsize=(7, 4)` 控制画布大小（单位是英寸）；轴标签和图题通过 `ax.set_xxx()` 设置；脚本最后用 `plt.show()` 把图显示出来（在服务器等没有图形界面的环境里，要用 5.8 节的 `savefig` 存成文件）。图题用英文是刻意的，原因见 5.9 节。

## 5.2 准备数据：造一条 Z 形机动轨迹

后面的例子都用同一份数据：一次简化版 **M07 矢量 Z 形机动**。Z 形机动是船舶操纵性的经典试验：让艏向角 ψ（船头指向的角度，0 表示正东，逆时针为正）到达 +20° 就下令反向转，到达 -20° 再反回来，于是船走出一条"之"字形航迹，用来观察船的转向响应和"超越"（换向指令发出后，艏向角因惯性继续冲过阈值的那一段）。

仿真模型是**教学简化**：纵荡速度 u（沿船头方向的速度分量）设为常值 1.5 m/s，艏摇角速度 r 对指令做一阶惯性响应，忽略横荡（船的左右平移）。第 4 章讲过数值积分，这里的循环就是最简单的欧拉积分，重点在后面的画图：

```python
import numpy as np
import matplotlib.pyplot as plt

# ---- 造数据：教学简化的 M07 矢量 Z 形机动（不是真实船舶模型）----
dt = 0.1                                   # 时间步长 (s)
t = np.arange(0.0, 120.0, dt)              # 时间轴
N = len(t)
u = 1.5                                    # 纵荡速度，设为常值 (m/s)
r_max = np.deg2rad(8.0)                    # 目标艏摇角速度 (rad/s)
psi_sw = np.deg2rad(20.0)                  # Z 形换向阈值 (rad)
T_r = 4.0                                  # 艏摇响应时间常数 (s)

x = np.zeros(N); y = np.zeros(N)           # 位置 (m)
psi = np.zeros(N)                          # 艏向角 (rad)
r = np.zeros(N)                            # 艏摇角速度 (rad/s)
rcmd = np.zeros(N)                         # 记录的换向指令
cmd = r_max                                # 初始指令：先向左转
rcmd[0] = cmd
for k in range(1, N):
    if psi[k-1] > psi_sw:                  # 艏向超过 +20 度，下令反向
        cmd = -r_max
    elif psi[k-1] < -psi_sw:               # 艏向低于 -20 度，下令反向
        cmd = r_max
    rcmd[k] = cmd
    r[k] = r[k-1] + dt * (cmd - r[k-1]) / T_r   # 一阶艏摇响应（教学简化）
    psi[k] = psi[k-1] + dt * r[k-1]
    x[k] = x[k-1] + dt * u * np.cos(psi[k-1])   # 航迹积分（忽略横荡，教学简化）
    y[k] = y[k-1] + dt * u * np.sin(psi[k-1])

rng = np.random.default_rng(42)                    # 固定随机种子，结果可复现
n1 = 400 + 60 * rcmd / r_max + rng.normal(0, 3, N) # 1 号推进器转速 (RPM)
n2 = 400 - 60 * rcmd / r_max + rng.normal(0, 3, N) # 2 号推进器转速 (RPM)
alpha1 = 20.0 * rcmd / r_max                       # 1 号推进器方位角指令 (deg)，教学简化

psi_d = np.rad2deg(psi)                    # 画图用角度更直观
r_d = np.rad2deg(r)

print(f"数据点数: {N}")
print(f"航程: x 终点 {x[-1]:.1f} m, 横向偏移幅度 {np.max(np.abs(y)):.1f} m")
print(f"艏向角峰值: {np.max(psi_d):.1f} deg（阈值 20 deg，超出部分就是'超越'）")
print(f"换向次数: {int(np.sum(np.diff(np.sign(rcmd)) != 0))}")
```

预期输出（随机种子固定，你的结果应完全一致）：

```
数据点数: 1200
航程: x 终点 169.2 m, 横向偏移幅度 5.4 m
艏向角峰值: 29.0 deg（阈值 20 deg，超出部分就是'超越'）
换向次数: 10
```

这里双推进器的设定也是教学简化：转向时让两台推进器一快一慢（差动），并叠加一点噪声模拟真实传感器；方位角 alpha1（全回转推进器指向的角度）直接跟随换向指令在 ±20° 间切换。峰值 29° 超过阈值 20°，正是因为 r 响应有惯性——这就是 Z 形试验要记录的"超越量"。

**本章代码是累积的**：从下一节起，每个示例都沿用本节生成的 `t`、`x`、`y`、`psi`、`r`、`n1`、`n2` 等变量，不再重复数据生成代码。把本节代码与后面任一示例按顺序拼接，就是一个完整可运行的脚本。

## 5.3 时间序列：多曲线、图例、标签与样式

试航数据最常见的画法是"某个量随时间变化"。把两台推进器的转速画在一张图上对比，顺便加上图例、网格和样式（接 5.2 节的变量，完整脚本需带上 5.2 节的数据生成代码）：

```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, n1, color="tab:blue", linewidth=1.0, label="n1 (port)")
ax.plot(t, n2, color="tab:orange", linewidth=1.0, linestyle="--", label="n2 (starboard)")
ax.set_xlabel("time t (s)")
ax.set_ylabel("propeller speed (RPM)")
ax.set_title("M07 zigzag: propeller speeds")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
```

你会看到两条阶梯状曲线：换向指令翻转时，n1 在 460 和 340 RPM 之间跳变，n2 正好相反，上面叠着小幅噪声。要点：

- `label="..."` 给曲线起名字，**再调用 `ax.legend()` 图例才会显示**，缺一步都不行；
- `color` 常用 `"tab:blue"`、`"tab:orange"`、`"tab:red"` 等标准色，也可写 `"b"`、`"red"`；
- `linestyle="--"` 画虚线，`linewidth` 控制线宽；
- `ax.grid(True, alpha=0.3)` 加半透明网格，读数更方便。

## 5.4 subplots：一张画布放多张图

一次机动要看 psi、r、转速、方位角好几个通道，一张张弹窗太麻烦。`plt.subplots(2, 2)` 返回的 `axes` 是一个 2×2 的 NumPy 数组，用 `axes[行, 列]` 取每个坐标区分别画图：

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)

axes[0, 0].plot(t, psi_d, color="tab:blue")
axes[0, 0].set_ylabel("psi (deg)")
axes[0, 0].set_title("heading")
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t, r_d, color="tab:red")
axes[0, 1].set_ylabel("r (deg/s)")
axes[0, 1].set_title("yaw rate")
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(t, n1, linewidth=0.8, label="n1")
axes[1, 0].plot(t, n2, linewidth=0.8, label="n2")
axes[1, 0].set_ylabel("RPM")
axes[1, 0].set_xlabel("time t (s)")
axes[1, 0].set_title("propeller speeds")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t, alpha1, color="tab:green")
axes[1, 1].set_ylabel("alpha1 (deg)")
axes[1, 1].set_xlabel("time t (s)")
axes[1, 1].set_title("azimuth command")
axes[1, 1].grid(True, alpha=0.3)

fig.suptitle("M07 zigzag overview (toy data)")   # 整张图的总标题
fig.tight_layout()                                # 自动收紧子图间距，防止标签重叠
plt.show()
```

得到的四联图里：左上是艏向角在 ±29° 之间往返的"锯齿"，右上是艏摇角速度在 ±7°/s 之间切换，左下是两台推进器的差动转速，右下是方位角指令方波。`sharex=True` 让四个子图共用 x 轴刻度，放大其中一张的 x 范围，其余会联动（在交互窗口里）。`axes.shape` 是 `(2, 2)`，如果只要一行子图，`plt.subplots(1, 3)` 返回的是一维数组，用 `axes[0]`、`axes[1]` 索引。

## 5.5 航迹图：x-y 平面与等比例坐标

时间序列看不出船"走到哪里了"。把 `y` 对 `x` 画图，就得到俯视图下的航迹。这里有一个极易踩的坑：**必须加 `ax.set_aspect("equal")`**，让 x 轴和 y 轴的 1 米在屏幕上一样长，否则 Matplotlib 会自动拉伸坐标轴填满画布，"之"字形会被压扁或拉歪，严重时连回转圆都画成椭圆：

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, y, color="tab:blue", linewidth=1.5)
ax.plot(x[0], y[0], marker="o", color="tab:green", markersize=8, label="start")
ax.plot(x[-1], y[-1], marker="s", color="tab:red", markersize=8, label="end")
ax.set_aspect("equal")                     # 关键：x 和 y 单位长度一致，形状不变形
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("M07 zigzag trajectory")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()
```

结果是一条向东推进、上下摆动的蛇形航迹，起点绿圈、终点红方块。船总共走了约 169 m，横向摆动幅度约 5.4 m，每次换向对应蛇形的一个"弯"。`marker="o"` 画圆点、`"s"` 画方块，用来标记特殊位置。

## 5.6 quiver：在轨迹上画艏向箭头

航迹线只告诉你船"在哪里"，看不出船头朝哪。用 `ax.quiver` 可以在指定位置画箭头。1200 个点全画上箭头会糊成一片，所以要**抽稀**——每隔 40 个点（4 秒）取一个。箭头方向就是艏向角的单位方向向量 (cos ψ, sin ψ)（第 3 章坐标变换的老朋友）：

```python
step = 40                                  # 每 40 个点取 1 个，即每 4 秒一个箭头
idx = np.arange(0, N, step)

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, y, color="tab:blue", linewidth=1.5, label="trajectory")
ax.quiver(x[idx], y[idx],                        # 箭头起点：抽稀后的位置
          np.cos(psi[idx]), np.sin(psi[idx]),    # 箭头方向：艏向角的单位向量
          color="tab:red", scale=30, width=0.004, label="heading psi")
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title("M07 zigzag: trajectory with heading arrows")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left")
plt.show()
```

`np.arange(0, N, step)` 生成下标 0, 40, 80, …，一次取出 30 个抽样点。`scale=30` 控制箭头长度（数值越大箭头越短），`width` 控制箭头杆的粗细。画出来红色箭头始终沿着船的艏向：在蛇形的波峰处箭头斜向下，波谷处斜向上——一眼就能看出艏向角领先于航迹拐弯，这正是 Z 形试验要观察的相位关系。RL 项目里画评估 rollout 时，这种"轨迹+艏向箭头"的图是标配。

## 5.7 twinx：单位不同的两条曲线共用 x 轴

艏向角（deg）和转速（RPM）量级差几十倍，画在一个 y 轴上必有一条被压成直线。办法之一是用 `ax.twinx()` 创建一个共享 x 轴、但 y 轴在右侧的新坐标区。换一份独立数据演示——M02 回零衰减：方位角指令保持 20°，第 2 秒阶跃回 0，艏摇角速度随之指数衰减（本例自带 imports，可单独运行）：

```python
import numpy as np
import matplotlib.pyplot as plt

t2 = np.arange(0.0, 20.0, 0.1)
alpha_cmd = np.where(t2 < 2.0, 20.0, 0.0)                    # 方位角指令 (deg)
r2 = np.where(t2 < 2.0, 0.0, 6.0 * np.exp(-(t2 - 2.0) / 3.0))  # 艏摇角速度衰减 (deg/s)

fig, axL = plt.subplots(figsize=(7, 4))
lineL, = axL.plot(t2, alpha_cmd, color="tab:blue", label="alpha cmd")
axL.set_xlabel("time t (s)")
axL.set_ylabel("azimuth alpha (deg)", color="tab:blue")
axL.tick_params(axis="y", labelcolor="tab:blue")

axR = axL.twinx()                              # 共享 x 轴的右 y 轴
lineR, = axR.plot(t2, r2, color="tab:red", linestyle="--", label="yaw rate r")
axR.set_ylabel("yaw rate r (deg/s)", color="tab:red")
axR.tick_params(axis="y", labelcolor="tab:red")

axL.legend(handles=[lineL, lineR], loc="upper right")   # 合并两条曲线的图例
axL.set_title("M02 return-to-zero decay (toy data)")
axL.grid(True, alpha=0.3)
plt.show()
```

左轴蓝色是方位角指令的阶跃，右轴红色虚线是角速度从 6 deg/s 衰减到接近 0。两个细节：`lineL, = axL.plot(...)` 的逗号不能少（`plot` 返回的是列表，逗号是解包）；`twinx` 之后图例要手动合并，把两条曲线的句柄放进 `handles` 列表。双 y 轴适合"指令 vs 响应"的对照，但也容易误导（两边刻度各自独立），报告里慎用，能分子图就分子图。

## 5.8 savefig：把图保存成文件

在服务器上跑批量仿真、或者写报告时，需要把图存成文件而不是弹窗：

```python
fig.savefig("zigzag_track.png", dpi=150, bbox_inches="tight")
```

- 文件名后缀决定格式：`.png` 位图通用，`.pdf` / `.svg` 是矢量图，放大不糊，适合论文和报告；
- `dpi=150` 控制位图分辨率（默认 100）；
- `bbox_inches="tight"` 自动裁掉多余白边，防止轴标签被截掉；
- `savefig` 要写在 `plt.show()` **之前**，因为 `show()` 之后图可能被清空，存出一张空白图。

## 5.9 中文乱码与负号方块

Matplotlib 默认字体不含中文字形，图题写中文常会显示成一排方块 "□□□"。两个办法：

1. **本章示例统一用英文图题和轴标签**（`"heading"`、`"yaw rate r (deg/s)"`），彻底绕开问题，论文投稿通常也要求英文图；
2. 确实需要中文时，在 `import matplotlib.pyplot as plt` 之后设置中文字体：

```python
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False   # 解决负号显示成方块的问题
```

`font.sans-serif` 是候选字体列表，Matplotlib 按顺序找你系统里**已安装**的第一个；列表中的名字要按你的操作系统实际所装字体调整（Linux 常见 Noto Sans CJK SC，Windows 常见 Microsoft YaHei / SimHei，macOS 常见 PingFang SC）。设置后如果仍弹 `Glyph ... missing` 警告，说明这些字体一个都没装上，需要安装字体或换名字。第二行 `axes.unicode_minus = False` 让坐标轴负号用普通短横线显示——否则即使中文正常了，"-20" 的负号也可能变成方块。这两行经常被一起复制，记住它们的分工：第一行管中文，第二行管负号。

## 常见错误

**1. 图例是空的，或弹出警告 "No artists with labels found to put in legend."**

```python
ax.plot(t, n1)          # 错误：忘了写 label
ax.legend()             # 图例里没有内容
```

```python
ax.plot(t, n1, label="n1 (port)")   # 正确：先给曲线命名
ax.legend()                          # 再调用 legend() 显示
```

**2. 航迹图忘了等比例，形状失真**

```python
ax.plot(x, y)           # 错误：Matplotlib 自动拉伸坐标轴，"之"字形被压扁
```

```python
ax.plot(x, y)
ax.set_aspect("equal")  # 正确：x、y 单位长度一致，航迹不变形
```

**3. 把 `plt` 的函数名套到 `ax` 上，报 AttributeError**

```python
ax.xlabel("time t (s)")   # 错误：AttributeError: 'Axes' object has no attribute 'xlabel'
```

```python
ax.set_xlabel("time t (s)")   # 正确：Axes 的方法多带 set_ 前缀
ax.set_ylabel("psi (deg)")
ax.set_title("heading")
```

**4. `savefig` 写在 `plt.show()` 之后，存出空白图**

```python
plt.show()                          # 错误：show() 之后图已被清空
fig.savefig("track.png")
```

```python
fig.savefig("track.png", dpi=150)   # 正确：先保存
plt.show()                          # 再显示
```

## 动手练习

**练习 1（基础）**：用 5.2 节的数据，把艏摇角速度 `r_d` 随时间的曲线画出来，要求：红色虚线、线宽 1.5、带网格、加一条 y=0 的黑色水平参考线（提示：`ax.axhline(0, color="k", linewidth=0.8)`），并保存为 `yaw_rate.png`。

**练习 2（进阶）**：把 5.6 节的艏向箭头改成每 2 秒一个、绿色箭头，并把图保存成 PDF 矢量图 `track.pdf`。

**练习 3（综合）**：用 `twinx` 把艏向角 `psi_d`（左轴，蓝色）和艏摇角速度 `r_d`（右轴，红色）画在一张图上，观察两者的相位关系（r 领先于 psi 换向）。

### 参考答案

```python
# 练习 1
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(t, r_d, color="tab:red", linestyle="--", linewidth=1.5, label="r")
ax.axhline(0, color="k", linewidth=0.8)
ax.set_xlabel("time t (s)")
ax.set_ylabel("yaw rate r (deg/s)")
ax.set_title("M07 zigzag: yaw rate")
ax.grid(True, alpha=0.3)
ax.legend()
fig.savefig("yaw_rate.png", dpi=150)

# 练习 2
idx = np.arange(0, N, 20)            # dt=0.1 s，每 20 个点即每 2 秒
fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(x, y, color="tab:blue", linewidth=1.5)
ax.quiver(x[idx], y[idx], np.cos(psi[idx]), np.sin(psi[idx]),
          color="tab:green", scale=30, width=0.004)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
fig.savefig("track.pdf", bbox_inches="tight")

# 练习 3
fig, axL = plt.subplots(figsize=(9, 4))
lineL, = axL.plot(t, psi_d, color="tab:blue", label="psi")
axL.set_ylabel("psi (deg)", color="tab:blue")
axL.set_xlabel("time t (s)")
axR = axL.twinx()
lineR, = axR.plot(t, r_d, color="tab:red", linestyle="--", label="r")
axR.set_ylabel("r (deg/s)", color="tab:red")
axL.legend(handles=[lineL, lineR], loc="upper right")
axL.set_title("psi vs r (twin y axes)")
plt.show()
```

三个练习都依赖 5.2 节生成的变量，运行时请把数据生成代码放在前面。

## 本章小结

- `fig, ax = plt.subplots()` 创建画布和坐标区，画图调用 `ax` 的方法：`plot`、`set_xlabel`、`set_ylabel`、`set_title`、`grid`、`legend`；
- 多曲线图靠 `label=` + `ax.legend()`，样式靠 `color`、`linestyle`、`linewidth`；
- `plt.subplots(2, 2)` 返回 `axes` 数组，做子图布局；`sharex=True` 共享 x 轴；
- 航迹图必须 `ax.set_aspect("equal")`；`ax.quiver` 配合 `np.arange(0, N, step)` 抽稀画艏向箭头；
- 单位不同用 `twinx` 双 y 轴；`savefig(..., dpi=150, bbox_inches="tight")` 在 `show()` 之前保存；
- 中文图题要么换英文，要么设置 `font.sans-serif`，并记得 `axes.unicode_minus = False` 修负号。

## 下一章预告

本章的数据是我们用 NumPy"造"出来的，真实项目里数据来自试航记录文件（CSV）。下一章 **《第 6 章 pandas处理试航数据》** 将学习用 pandas 读取 CSV、按时间对齐多路传感器、筛选机动片段、计算统计量——并把本章的画图技能用在真实结构上。

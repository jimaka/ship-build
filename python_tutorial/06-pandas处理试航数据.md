# 第 6 章 pandas 处理试航数据

本教程以双全回转推进船舶 3DOF 建模项目为背景。本章教你用 pandas 把"多个采样率、带缺失的试航记录"整理成"干净、对齐、划分好角色的数据集"。

## 本章学习目标

1. 用 `read_csv` 读取试航 CSV，并用 `head`/`describe` 快速了解数据全貌；
2. 把时间戳列解析成真正的时刻并设为索引，会选列、会做布尔筛选；
3. 统计并处理缺失值（`isna`/`interpolate`/`dropna`），懂得为什么不跨长缺口插值；
4. 用 `resample` 降采样，用 `merge_asof` 把采样率不同的通道对齐到同一时间轴；
5. 按连续时间段划分 train/val/test，并解释为什么随机打散时间序列会造成数据泄漏。

## 为什么学这一章

项目的研究文档 `research/03-试航设计与数据需求.md` 对真实数据提出两条硬性要求：一是时标与质量规则（§3.3.3）——船上各通道采样率不同、时间戳异步，必须统一时间基准、保留原始时间戳、记录缺失，且"禁用跨长缺口插值"；二是数据角色与防泄漏（§3.2）——数据必须预先划分为 train/val/test 并在训练前冻结，同一次机动及其派生窗口只能有一个角色。这些要求落到代码上全是 pandas 的活：读 CSV、解析时间、补缺、重采样、对齐、打标签、导出。第 7 章的参数辨识和第 8、10 章的仿真环境与强化学习，都要从本章整理好的数据集出发。本章先用代码生成一份"以假乱真"的模拟试航数据，然后把完整处理流程走一遍。

## 6.1 先造一份模拟试航数据

试航（sea trial）是把船开到试验水域、按预定方案机动并记录各传感器数据的过程。我们模拟一次 60 秒的湖面试航，产生两个 CSV 文件：

- **主通道（10 Hz）**：两台全回转推进器的转速 `n1`、`n2`（单位 RPM，转/分钟）和方位角 `alpha1`、`alpha2`（单位度。全回转推进器可以绕竖直轴旋转，转速决定推力大小、方位角决定推力方向）；三个运动速度：纵荡速度 `u`（沿船长方向）、横荡速度 `v`（沿船宽方向）、艏摇角速度 `r`（船头左右偏转的角速度，单位 rad/s）。故意混入少量缺失值。
- **GNSS 位置通道（1 Hz）**：GNSS 即卫星定位，给出船的位置 `x`、`y`。它的采样率比主通道低，这是船上常见的情况。

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)          # 固定随机种子，保证结果可复现

# ========== 主通道：10 Hz，共 60 秒 ==========
dt = 0.1
n_samples = 600
t = np.arange(n_samples) * dt            # 0.0, 0.1, ..., 59.9 秒
time0 = pd.Timestamp("2026-06-15 09:30:00")
timestamps = time0 + pd.to_timedelta(t, unit="s")

# 两台全回转推进器的转速（RPM）：分档阶跃，模拟"多档定常"机动
n1 = np.where(t < 20, 600.0, np.where(t < 40, 900.0, 750.0))
n2 = np.where(t < 20, 600.0, np.where(t < 40, 700.0, 750.0))  # 中段故意制造差动

# 方位角（度）：前半程正对船艏，30 秒后 1 号机偏转 10 度
alpha1 = np.where(t < 30, 0.0, 10.0)
alpha2 = np.zeros(n_samples)

# 纵荡速度 u：一阶惯性趋近"与总转速成正比"的目标速度（这是教学简化）
u_target = (n1 + n2) / 1200.0 * 2.0
u = np.zeros(n_samples)
for k in range(1, n_samples):
    u[k] = u[k - 1] + dt / 3.0 * (u_target[k] - u[k - 1])
u = u + rng.normal(0, 0.02, n_samples)   # 加一点测量噪声

# 艏摇角速度 r 与横荡速度 v：与差动转速、方位角相关（这是教学简化）
r = 0.00005 * (n1 - n2) + 0.001 * alpha1 + rng.normal(0, 0.001, n_samples)
v = 2.0 * r + rng.normal(0, 0.01, n_samples)

# 由速度积分出艏向角 psi 与位置 (x, y)，供 GNSS 通道使用
psi = np.concatenate([[0.0], np.cumsum(r[:-1]) * dt])
x = np.concatenate([[0.0], np.cumsum((u * np.cos(psi) - v * np.sin(psi))[:-1]) * dt])
y = np.concatenate([[0.0], np.cumsum((u * np.sin(psi) + v * np.cos(psi))[:-1]) * dt])

df_main = pd.DataFrame({
    "timestamp": timestamps,
    "n1": n1, "n2": n2,
    "alpha1": alpha1, "alpha2": alpha2,
    "u": u, "v": v, "r": r,
})

# 故意制造少量缺失值：u 缺 6 个点，alpha1 缺 4 个点
miss_u = rng.choice(n_samples, size=6, replace=False)
miss_a = rng.choice(n_samples, size=4, replace=False)
df_main.loc[miss_u, "u"] = np.nan
df_main.loc[miss_a, "alpha1"] = np.nan

df_main.to_csv("trial_main.csv", index=False)

# ========== GNSS 位置通道：1 Hz，采样率不同 ==========
gnss_rows = np.arange(0, n_samples, 10)  # 每 10 行取 1 行
df_gnss = pd.DataFrame({
    "timestamp": timestamps[gnss_rows],
    "x": x[gnss_rows] + rng.normal(0, 0.05, len(gnss_rows)),  # GNSS 噪声更大
    "y": y[gnss_rows] + rng.normal(0, 0.05, len(gnss_rows)),
})
df_gnss.to_csv("trial_gnss.csv", index=False)

print("主通道行数:", len(df_main), " GNSS 行数:", len(df_gnss))
print("u 缺失点数:", df_main["u"].isna().sum(),
      " alpha1 缺失点数:", df_main["alpha1"].isna().sum())
```

运行输出：

```text
主通道行数: 600  GNSS 行数: 60
u 缺失点数: 6  alpha1 缺失点数: 4
```

两点说明。第一，代码里船速、角速度的"物理模型"只是几个顺手写的公式（教学简化），目的是造出形态合理的数据，不追求物理正确；真实的船模在第 4 章会认真推导。第二，真实数据契约要求分别保存测量时间、接收时间、命令时间等多种时标（研究文档 §3.3.3），本章只保留一个测量时间，同样是教学简化。

## 6.2 读取 CSV 与快速概览

pandas 里装表格数据的对象叫 `DataFrame`（数据表），读 CSV 用 `read_csv`。拿到表的前两个动作永远是 `head`（看前几行长什么样）和 `describe`（看每列的统计概览）：

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")   # 读入 CSV，得到 DataFrame（数据表）

print(df.head(3))              # 看前 3 行
print("----")
print(df.describe().round(3))  # 每列的统计量概览
```

输出（节选）：

```text
                 timestamp     n1     n2  ...         u         v         r
0  2026-06-15 09:30:00.000  600.0  600.0  ...  0.006094 -0.012195  0.000515
1  2026-06-15 09:30:00.100  600.0  600.0  ...  0.045867 -0.006017 -0.000578
2  2026-06-15 09:30:00.200  600.0  600.0  ...  0.146120  0.006751  0.001274
----
            n1       n2   alpha1  alpha2        u        v        r
count  600.000  600.000  596.000   600.0  594.000  600.000  600.000
mean   750.000  683.333    5.000     0.0    2.263    0.016    0.008
max    900.000  750.000   10.000     0.0    2.712    0.061    0.022
```

`describe` 的 `count` 行是"非缺失"计数——注意 `alpha1` 是 596、`u` 是 594，比 600 少，这正是我们故意埋的缺失值，概览阶段就能发现。

## 6.3 选列与布尔筛选

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")

u = df["u"]                     # 选一列：得到一维的 Series
sub = df[["u", "v", "r"]]       # 选多列（注意是双层方括号）：仍是 DataFrame

fast = df[df["n1"] > 800]                       # 布尔筛选：只保留 n1 大于 800 的行
print("n1>800 的行数:", len(fast))
combo = df[(df["n1"] > 800) & (df["u"] > 1.5)]  # 多条件：用 & 且每个条件必须加括号
print("n1>800 且 u>1.5 的行数:", len(combo))
```

输出：

```text
n1>800 的行数: 200
n1>800 且 u>1.5 的行数: 196
```

`df["n1"] > 800` 会得到一列 True/False，把它塞回 `df[...]` 就表示"只保留为 True 的行"。

## 6.4 把时间戳变成"真正的时刻"

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")
print("解析前:", df["timestamp"].dtype)        # str：只是普通字符串

df["timestamp"] = pd.to_datetime(df["timestamp"])   # 解析成真正的时刻
df = df.set_index("timestamp")                      # 设为索引（行标签）
print("解析后:", df.index.dtype)

# 有了时间索引，就能用时刻直接切片
seg = df.loc["2026-06-15 09:30:20":"2026-06-15 09:30:22"]
print("09:30:20~09:30:22 共", len(seg), "行")
```

输出：

```text
解析前: str
解析后: datetime64[us]
09:30:20~09:30:22 共 30 行
```

刚读进来的 `timestamp` 列只是字符串，不能比较先后、不能切片；`pd.to_datetime` 把它解析成时刻，`set_index` 把它变成行标签（`DatetimeIndex`），之后就能用 `df.loc["某时刻":"某时刻"]` 按时间切片——注意切片写到"秒"会包含这一整秒，所以上面切出了 3 秒 × 10 Hz = 30 行。研究文档 §3.3.3 要求"保留原始异步时间戳、统一时间基准"，这一步就是统一时间基准的起点；原始 CSV 文件本身我们从不改动（§3.2 要求原始记录不可变），所有处理都在内存副本上进行。

## 6.5 缺失值：先看看，再决定怎么办

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# 1) 先看缺了多少
print(df.isna().sum())

# 2) 办法一：dropna——直接删掉含缺失的行（丢数据，慎用）
df_dropped = df.dropna()
print("dropna 后剩", len(df_dropped), "行（原 600 行）")

# 3) 办法二：interpolate——按时间线性插值，补上零星缺失
df["u"] = df["u"].interpolate(method="time")
# 4) 研究文档 §3.3.3 要求"禁用跨长缺口插值"→ 用 limit 限制最多连补几个点
df["alpha1"] = df["alpha1"].interpolate(method="time", limit=3)
print("处理后缺失总数:", df.isna().sum().sum())
```

输出：

```text
n1        0
n2        0
alpha1    4
alpha2    0
u         6
v         0
r         0
dtype: int64
dropna 后剩 590 行（原 600 行）
处理后缺失总数: 0
```

怎么选？`dropna` 简单粗暴，但整行删除会把同行其他完好的通道（n1、v、r……）一起丢掉，还可能破坏等间隔时间轴，一般只用于"缺得实在没法用"的情况。`interpolate(method="time")` 按时间间隔做线性插值，适合零星的单点丢失。但要记住 §3.3.3 的规矩：**禁用跨长缺口插值**——传感器掉线 10 秒，插值出来的曲线是你编的，不是测的。`limit=3` 表示最多连续补 3 个点，缺口更长就让它继续缺着，后续按质量规则剔除这一段；真实项目里缺口长度、异常值的退出阈值要在采集前就定好。

## 6.6 重采样：把 10 Hz 变成 1 Hz

`resample` 按时间窗重新分组，再配合聚合函数（`mean`、`max` 等）：

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")

# 10 Hz 降采样到 1 Hz：每秒取平均（顺便压掉一部分噪声）
df_1hz = df.resample("1s").mean()
print(df_1hz.head(3).round(3))
print("1 Hz 数据行数:", len(df_1hz))

# 反过来"升采样"只会重复旧值，不会带来新信息
df_up = df_1hz.resample("0.1s").ffill()
print("升采样后行数:", len(df_up), "（行变多了，信息量没变）")
```

输出：

```text
                        n1     n2  alpha1  alpha2      u      v    r
timestamp
2026-06-15 09:30:00  600.0  600.0     0.0     0.0  0.268 -0.002  0.0
2026-06-15 09:30:01  600.0  600.0     0.0     0.0  0.776 -0.009 -0.0
2026-06-15 09:30:02  600.0  600.0     0.0     0.0  1.127  0.003 -0.0
1 Hz 数据行数: 60
升采样后行数: 591 （行变多了，信息量没变）
```

降采样取均值可以抑制噪声、统一采样率，是常用操作。反过来要警惕：把 1 Hz 数据"升采样"到 10 Hz，行数变多了，但每一行都是旧值的重复——研究文档 §3.3.3 说得很直白："低频数据插值到 10 Hz 不增加带宽"，别让行数骗了你。另外提醒一句：如果重采样的列是角度（如艏向角 psi），±180° 跳变处直接平均会算出假的大角速度，§3.3.3 要求角度先展开再重采样，本章数据没用到，先留个印象。

## 6.7 时间对齐：merge_asof 讲透

现在面对本章最核心的问题：主通道 10 Hz、GNSS 1 Hz，怎么合成一张表？思路是——**以 10 Hz 时间轴为准，每个主通道时刻去"认领"当时最新收到的那条 GNSS**。这正是 `pd.merge_asof`（as-of join，"截至某时刻"的连接）干的事：

```python
import matplotlib.pyplot as plt
import pandas as pd

df_main = pd.read_csv("trial_main.csv")
df_main["timestamp"] = pd.to_datetime(df_main["timestamp"])
df_main = df_main.sort_values("timestamp")   # merge_asof 要求两边都按时间排序

df_gnss = pd.read_csv("trial_gnss.csv")
df_gnss["timestamp"] = pd.to_datetime(df_gnss["timestamp"])
df_gnss = df_gnss.sort_values("timestamp")

# 把 1 Hz 的 GNSS 位置"贴"到 10 Hz 主通道时间轴上：
# 每个主通道时刻，取"当时最新收到的那一条"GNSS（direction="backward"，不偷看未来）
merged = pd.merge_asof(df_main, df_gnss, on="timestamp", direction="backward")
merged = merged.set_index("timestamp")
print(merged[["u", "x", "y"]].head(13).round(3))
print("对齐后总行数:", len(merged), " x 缺失数:", merged["x"].isna().sum())

# tolerance：超过 0.15 秒没有新 GNSS 就不贴，宁缺毋滥
merged_tol = pd.merge_asof(df_main, df_gnss, on="timestamp",
                           direction="backward", tolerance=pd.Timedelta("0.15s"))
print("加 0.15s 容差后 x 缺失数:", merged_tol["x"].isna().sum())
```

输出：

```text
                             u      x      y
timestamp
2026-06-15 09:30:00.000  0.006  0.048  0.012
2026-06-15 09:30:00.100  0.046  0.048  0.012
...
2026-06-15 09:30:00.900  0.509  0.048  0.012
2026-06-15 09:30:01.000  0.593  0.170  0.042
对齐后总行数: 600  x 缺失数: 0
加 0.15s 容差后 x 缺失数: 480
```

看输出就能读懂规则：09:30:00.0 到 00.9 这 10 个时刻认领的都是 1 Hz 在 00.0 秒那一帧位置（x 保持 0.048），直到 01.0 秒新 GNSS 到达才跳变——合成表里低频通道呈"阶梯状"。`direction` 有三个选项：`"backward"`（默认，只看过去的最新值）、`"forward"`（只看未来）、`"nearest"`（时间最近，不管前后）；`tolerance` 限定认领的最大时间差，上例中 0.15 秒的容差让 600 行里 480 行的 x 变成缺失——宁可缺，也不拿一秒前的旧位置冒充当前值，这对应 §3.3.3"记录缺失和质量变化"的质量意识。默认用 `"backward"` 而不用看起来更"准"的 `"nearest"`，是因为 `"nearest"` 会偷看未来：00.6 秒的时刻会认领 01.0 秒才收到的位置，真实系统那时根本还没收到；研究文档 §3.2 也明确了同一原则——离线辨识标签可用双向（非因果）平滑，部署观测只能用因果估计，两者分开存储，所以给"部署/在线"准备的数据集，对齐只能用 `"backward"`。

最后画图直观对比一下（matplotlib 用法见第 5 章）：

```python
# 接上面的代码，画前 8 秒，对比"对齐前"的离散点与"对齐后"的阶梯线
t_end = merged.index[0] + pd.Timedelta(seconds=8)
win_m = merged[merged.index < t_end]
win_g = df_gnss[df_gnss["timestamp"] < t_end]

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(win_g["timestamp"], win_g["x"], "o", ms=8, label="GNSS 原始（1 Hz）")
ax.plot(win_m.index, win_m["x"], "-", label="merge_asof 对齐后（10 Hz）")
ax.set_xlabel("时间")
ax.set_ylabel("x / m")
ax.legend()
fig.tight_layout()
fig.savefig("align_zoom.png", dpi=100)
plt.show()
```

图上每秒一个蓝色圆点（原始 GNSS），橙色阶梯线在每个整秒跳变、其余时间保持上一条的值——这就是"backward 对齐"的几何样子。如果中文标签显示成方块，说明系统缺中文字体，把标签换成英文即可，不影响代码逻辑。

## 6.8 划分 train/val/test：按时间块，训练前冻结

辨识模型（第 7 章）和强化学习（第 10 章）都要把数据分成三种角色，研究文档 §3.2.2 的分工是：**train**（训练集）用于拟合参数和计算归一化统计量；**val**（验证集）用于选模型结构、调超参、选 checkpoint；**test**（测试集）只能在模型和流程全部冻结后做最终评价——被反复查看、反复用来调参的数据，就必须改标为开发数据，不能再当测试集。对时间序列，划分必须**按连续时间段**，而且要在训练开始前定死：

```python
import numpy as np
import pandas as pd

df = pd.read_csv("trial_main.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
df["u"] = df["u"].interpolate(method="time")
df["alpha1"] = df["alpha1"].interpolate(method="time", limit=3)

t0 = df.index[0]
t1 = t0 + pd.Timedelta(seconds=36)   # 前 60% 时间做训练
t2 = t0 + pd.Timedelta(seconds=48)   # 接下来 20% 做验证

# 按连续时间段划分：明确用 < / >=，保证端点不重叠
train = df[df.index < t1]
val   = df[(df.index >= t1) & (df.index < t2)]
test  = df[df.index >= t2]
print("train/val/test 行数:", len(train), len(val), len(test))

# 给整表打"角色"标签列并导出，角色随文件一起冻结
df["role"] = "test"
df.loc[df.index < t1, "role"] = "train"
df.loc[(df.index >= t1) & (df.index < t2), "role"] = "val"
print(df["role"].value_counts())

# 反面教材：随机打散会怎样？
rng = np.random.default_rng(0)
is_test = rng.random(len(df)) < 0.2      # 随机抽 20% 当测试集
test_idx = np.where(is_test)[0]
test_idx = test_idx[(test_idx > 0) & (test_idx < len(df) - 1)]
leak = np.mean((~is_test[test_idx - 1]) | (~is_test[test_idx + 1]))
print(f"随机划分下，{leak:.0%} 的测试样本隔壁就是训练样本")
```

输出：

```text
train/val/test 行数: 360 120 120
role
train    360
val      120
test     120
Name: count, dtype: int64
随机划分下，97% 的测试样本隔壁就是训练样本
```

为什么不能像图像分类那样随机打散？因为时间序列相邻时刻强相关：0.1 秒前后的两个样本，船速、转速几乎一模一样。上面的"反面教材"算了一笔账——随机抽 20% 当测试集时，97% 的测试样本的前一个或后一个样本就在训练集里。模型只要"记住邻居"就能在测试集上拿高分，评估结果虚假地乐观，这就是**数据泄漏**（data leakage）。

三条规矩请带走：第一，划分在训练前冻结——§3.2 要求划分清单带版本号（`split_version`）、训练前冻结，之后要改就保留旧结果和理由；第二，划分单位其实是"整段连续记录"而非"样本"，真实项目里以整条机动/航次（`run_id`/`group_id`）为最小单位，同一次执行及其派生窗口只能有一个角色，本章在单条记录内按时间块划分只是教学演示；第三，块与块之间不许暧昧——§3.2 明确"相邻重叠窗口不得跨角色""禁止跨组平滑"，所以我们用 `<`/`>=` 把端点切干净，更讲究的做法还会在交界处留一小段间隔谁都不用。

## 6.9 导出处理结果

```python
df.to_csv("trial_labeled.csv")   # 时间索引会一并写入文件第一列
print("已导出 trial_labeled.csv")
```

`to_csv` 把打好 `role` 标签的表存盘，后续章节直接读它。重申一次：原始的 `trial_main.csv`、`trial_gnss.csv` 全程只读不写——§3.2 的处理流水线第一步就是"不可变原始记录"，对齐、质检、划分都在副本上进行并另存新文件。

## 常见错误

**1. 没解析时间戳就 `resample`。**

```python
df = pd.read_csv("trial_main.csv")
df.resample("1s").mean()   # TypeError: Only valid with DatetimeIndex... 正确：先 to_datetime 再 set_index（见 6.4 节）
```

**2. 链式赋值 `df["u"][0] = 1.0`：不报错，但值根本没改。**

```python
df["u"][0] = 1.0        # 警告 ChainedAssignmentError：df["u"] 取出的是副本，改副本原表不变
df.loc[0, "u"] = 1.0    # 正确：用 .loc 一步定位"行 + 列"
```

**3. `merge_asof` 前忘了排序。**

两边只要有一边没按时间升序，就会报 `ValueError: left keys must be sorted`。正确写法是像 6.7 节那样先 `sort_values("timestamp")`。

**4. 随机打散时间序列再划分 train/test。**

这是概念性错误：语法上不报错，结果上测试集 97% 的样本能在训练集里找到"双胞胎邻居"，评估失真。正确写法：按连续时间段（真实项目按整条 run/航次）划分，并在训练前冻结。

## 动手练习

1. 读取 `trial_main.csv`，打印每列的缺失数量；把 `u` 的缺失用时间插值补上，打印插值前后 `u` 的均值和标准差，观察变化大不大、想想为什么。
2. 把主通道重采样到 2 Hz（时间窗 `"0.5s"`，取均值），导出为 `trial_main_2hz.csv`，并打印行数。
3. 思考题：船上在线（实时）系统对齐 GNSS 时，为什么只能用 `direction="backward"`？如果把 `tolerance` 设得很小（比如 0.05 秒），合并结果会发生什么变化？

## 练习参考答案

**练习 1**

```python
import pandas as pd

df = pd.read_csv("trial_main.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
print(df.isna().sum())
print("插值前: 均值 %.4f  标准差 %.4f" % (df["u"].mean(), df["u"].std()))
df["u"] = df["u"].interpolate(method="time")
print("插值后: 均值 %.4f  标准差 %.4f" % (df["u"].mean(), df["u"].std()))
```

输出中均值从 2.2632 变到 2.2643、标准差从 0.4916 变到 0.4901，变化很小——零星缺失 6 个点、又用邻近值插补，统计量自然几乎不动；缺失多或缺口长时就不能这么放心了。

**练习 2**

```python
df_2hz = df.resample("0.5s").mean()   # 接练习 1 的 df
df_2hz.to_csv("trial_main_2hz.csv")
print("2 Hz 数据行数:", len(df_2hz))   # 120，即 60 秒 × 2 Hz
```

**练习 3**

在线系统在任何时刻只能使用"已经收到"的数据，`backward` 保证每个时刻认领的都是过去最新的一条，不偷看未来，满足因果性；`nearest` 会用到未来才到达的数据，只能用于离线处理。`tolerance` 设成 0.05 秒后，每秒只有整秒时刻本身（10 Hz 下每秒第 1 个点）落在容差内，其余约 9/10 的行 GNSS 列全是缺失——宁缺毋滥走向极端，数据就没法用了，所以容差要按通道实际采样间隔来定。

## 本章小结与下一章预告

本章完整走了一遍试航数据的整理流程：`read_csv` 读取与 `head`/`describe` 概览 → `to_datetime` + `set_index` 统一时间基准 → `isna`/`interpolate`/`dropna` 处理缺失（不跨长缺口）→ `resample` 降采样 → `merge_asof` 因果对齐多采样率通道 → 按连续时间段划分并冻结 train/val/test → `to_csv` 导出。这些步骤一一对应研究文档 §3.3.3 的时标质量规则与 §3.2 的数据角色防泄漏要求。下一章是**第 7 章 最小二乘与参数辨识**：我们将用本章导出的 `trial_labeled.csv`，拿 train 段拟合船模里的未知参数（比如"转速到推力"的比例系数），用 val 段挑选模型，最后才在 test 段上验收——看最小二乘法如何从试航数据里"称出"模型的参数。

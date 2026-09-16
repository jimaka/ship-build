# 第 9 章 PyTorch神经网络入门
本教程以双全回转推进船舶 3DOF 建模项目为背景。

## 本章学习目标

学完本章，你将能够：

1. 在 NumPy 数组与 PyTorch tensor（张量）之间自由转换，并避开数据类型的坑；
2. 用 autograd（自动求导）让 PyTorch 替你算导数，理解"梯度"是怎么来的；
3. 用 `nn.Linear`、`nn.Sequential`、`nn.Module` 搭建一个小型多层感知机（MLP，最普通的前馈神经网络）；
4. 写出一个完整的训练循环：前向计算 → 算损失 → 反向传播 → 更新参数；
5. 亲手实现项目"MMG + 神经网络残差"路线的最简版本：让网络只学物理模型给不出的那部分力。

前置章节：第 2 章（NumPy 数组）、第 5 章（Matplotlib 画图）、第 8 章（类的用法）。

## 为什么学这一章

我们的项目要建一个"灰箱"模型：物理公式能写清楚的部分（坐标变换、已辨识的线性阻尼）用公式，写不清楚的部分交给神经网络。这不是拍脑袋的决定——研究文档 `research/01-方法对比矩阵.md` 的路线③（MMG + NN 力级残差灰箱）指出：物理骨架承担大部分拟合任务，神经网络只学残差（真实值与模型预测值之间的差），因此比纯黑箱模型更省数据；文献中 Woo 2018 只让网络辨识船舶模型的非线性残差，Nielsen 2022 用网络预测物理模型的速度偏差，都是这一思想。路线④（本项目选定的长期路线）让网络接管更大的子模块，但组装思想相同：解析结构打底，神经网络补物理模型给不出的部分。

灰箱还有一个工程上的好处：当船舶跑出训练数据覆盖的范围时，可以把网络的输出直接置零、退回纯物理模型兜底——这在路线③的文献中正是标准做法。

本章你不需要任何深度学习基础。我们会用一个可以几秒钟跑完的小实验，把"网络只学残差"这件事完整做一遍。需要说明的是：文献中的定量收益（例如某些论文报告的误差下降百分比）都是特定试验条件下的结果，不能提前当作本项目的结论；本章数字同理，只是教学演示。

> 未安装 PyTorch 的读者：`pip install torch`（更小的 CPU 版：`pip install torch --index-url https://download.pytorch.org/whl/cpu`）。
> 本章代码已在 CPU 版 PyTorch 2.14 上实际运行验证，普通笔记本电脑秒级完成。

## 9.1 tensor：PyTorch 里的"NumPy 数组"

tensor（张量）是 PyTorch 的基本数据类型，地位相当于 NumPy 的 ndarray：一维 tensor 像数组，二维 tensor 像矩阵。两者可以互相转换：

```python
import numpy as np
import torch

# 第 2 章的老朋友：numpy 数组。假设这是 3 个横荡速度采样（m/s）
# 横荡：船体在水面内沿左右方向平移的运动
v_np = np.array([0.1, 0.5, -0.3], dtype=np.float32)
print("numpy 数组：", v_np, v_np.dtype)

v_ts = torch.from_numpy(v_np)      # numpy -> tensor
print("tensor：", v_ts, v_ts.dtype)

back_np = v_ts.numpy()             # tensor -> numpy
print("转回 numpy：", back_np, type(back_np).__name__)

v_ts[0] = 9.9                      # 注意：from_numpy 与原数组共享内存！
print("改 tensor 后原 numpy 数组：", v_np)
```

输出：

```
numpy 数组： [ 0.1  0.5 -0.3] float32
tensor： tensor([ 0.1000,  0.5000, -0.3000]) torch.float32
转回 numpy： [ 0.1  0.5 -0.3] ndarray
改 tensor 后原 numpy 数组： [ 9.9  0.5 -0.3]
```

两个要点：

- **数据类型**：NumPy 数组默认是 `float64`，而 PyTorch 网络默认用 `float32`，混用会直接报错（见"常见错误"第 1 条）。养成习惯：进网络前统一转成 `float32`。
- **共享内存**：`torch.from_numpy()` 不做复制，改 tensor 会连累原数组。想要互不影响，用 `torch.tensor(v_np, dtype=torch.float32)`（总是复制一份）。

## 9.2 autograd：让 PyTorch 替你求导

训练神经网络的本质是"调整参数让误差变小"，而调整方向由**导数（梯度）**决定。第 7 章做最小二乘时我们手工推过梯度公式；参数一多（网络里动辄几百个），手工推导就不现实了。PyTorch 的 autograd（自动求导）会记录你对 tensor 做过的每一步运算，然后自动算出梯度：

```python
import torch

c = 2.0
v = torch.tensor(0.5, requires_grad=True)   # 告诉 PyTorch：请跟踪 v
F = c * v ** 2                              # 教学简化的阻尼力模型 F(v) = c·v²
F.backward()                                # 自动求 dF/dv
print("F =", F.item())
print("dF/dv =", v.grad.item(), "（手算验证：2·c·v =", 2 * c * 0.5, "）")
```

输出：

```
F = 0.5
dF/dv = 2.0 （手算验证：2·c·v = 2.0 ）
```

三行关键代码：`requires_grad=True` 声明"这个量要求导"；`F.backward()` 从 `F` 出发沿计算过程**反向传播**（backpropagation），自动算出各量的梯度；求好的梯度存放在 `v.grad` 里。

阻尼（damping）是流体阻碍船体运动的力，可以粗略理解为"水的摩擦力"。上面的 `F = c·v²` 是教学简化，真实船舶阻尼的形式正是本章实战的主角。

## 9.3 nn.Linear 与 nn.Sequential：搭积木

神经网络最基础的积木是**线性层** `nn.Linear(in_features, out_features)`，它做的事情就是矩阵乘法加偏置：`输出 = W·输入 + b`（W、b 是待训练的参数）。

```python
import torch
import torch.nn as nn

torch.manual_seed(0)                     # 固定随机种子，让结果可复现
layer = nn.Linear(1, 8)                  # 1 个输入特征 -> 8 个输出特征
x = torch.tensor([[0.5]])                # 形状 (样本数, 特征数)
print("线性层输出形状：", layer(x).shape)
print("权重形状：", layer.weight.shape, "，偏置形状：", layer.bias.shape)
```

输出：

```
线性层输出形状： torch.Size([1, 8])
权重形状： torch.Size([8, 1]) ，偏置形状： torch.Size([8])
```

注意输入的形状约定：**(样本数, 特征数)**。一个速度样本要写成 `[[0.5]]` 而不是 `[0.5]`。

只有线性层叠再多层也等价于一个线性层，所以层与层之间要插入**激活函数**（非线性变换），网络才能拟合弯曲的曲线。用 `nn.Sequential` 把积木按顺序串起来，就得到一个 MLP：

```python
import torch
import torch.nn as nn

torch.manual_seed(0)         # 固定随机种子，让下面的输出可复现
model = nn.Sequential(
    nn.Linear(1, 16),      # 输入：速度 v
    nn.Tanh(),             # 激活函数，把数值压到 (-1, 1)，曲线平滑
    nn.Linear(16, 16),     # 隐藏层
    nn.Tanh(),
    nn.Linear(16, 1),      # 输出：残差力
)
print(model)
print("参数总数：", sum(p.numel() for p in model.parameters()))
out = model(torch.tensor([[0.5], [-1.2]]))   # 两个样本一起算
print("两个样本的输出：\n", out)
```

输出：

```
Sequential(
  (0): Linear(in_features=1, out_features=16, bias=True)
  (1): Tanh()
  (2): Linear(in_features=16, out_features=16, bias=True)
  (3): Tanh()
  (4): Linear(in_features=16, out_features=1, bias=True)
)
参数总数： 321
两个样本的输出：
 tensor([[0.4052],
        [0.1228]], grad_fn=<AddmmBackward0>)
```

321 个参数 = (1×16+16) + (16×16+16) + (16×1+1)。输出末尾的 `grad_fn` 表示这个结果是可求导的——autograd 一直在默默记录。

## 9.4 用 nn.Module 定义自己的网络

第 8 章我们学会了用类组织代码。PyTorch 里所有网络都继承 `nn.Module`，只需写两个方法：`__init__` 里搭结构，`forward` 里定义数据怎么流动：

```python
import torch
import torch.nn as nn

class ResidualNet(nn.Module):
    """输入横荡速度 v，输出阻尼残差力的小网络。"""

    def __init__(self, n_hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

net = ResidualNet(n_hidden=8)
print(net(torch.tensor([[0.3]])))
```

`net(x)` 会自动调用 `forward`。实战代码我们就用这个类——隐藏层宽度 `n_hidden` 做成参数，方便做实验时调整网络大小。

## 9.5 完整实战：灰箱残差学习（MMG+NN 的最简版）

**场景设定（教学简化）**：只看船舶的横荡方向。真实阻尼力 = 已知线性项 `−Y_v·v` + 未知非线性项 `−Y_vv·v·|v|`（速度越大增长越快的二次项，真实船上很难直接测准）。假设线性系数 `Y_v` 已由第 7 章的最小二乘辨识得到，**网络只学剩下的残差**——这正是研究文档路线③"网络只学物理模型给不出的残差"的思想。MMG 是船舶操纵性建模的经典模块化框架（把船体、推进器等的力分开建模再合成），本例中"已知线性模型"就扮演着 MMG 骨架的角色。

数据从哪来？真实项目里来自试航；本章用一个自给自足的小仿真生成：给船施加来回扫动的激励力（模拟推进器扫频试验），积分出速度，再模拟"带噪声的测量"。

```python
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

np.random.seed(0)
torch.manual_seed(0)

# ========== 1. 简化仿真：生成"试航数据" ==========
Y_v = 4.0        # 已知：线性阻尼系数（假设已辨识）
Y_vv = 8.0       # 未知：非线性阻尼系数（真实船上的"真相"，网络要间接学它）
m = 10.0         # 横荡等效质量（教学简化取值）
dt = 0.05        # 时间步长（s），与全书约定一致

def true_damping(v):
    """真实阻尼力 = 线性项 + 二次项 v|v|（教学简化的物理模型）"""
    return -Y_v * v - Y_vv * v * np.abs(v)

# 施加扫频激励力，正向积分 40 秒得到速度时序（第 4 章的欧拉法）
T = 40.0
steps = int(T / dt)
t = np.arange(steps) * dt
force_in = 30 * np.sin(0.5 * t) + 15 * np.sin(1.7 * t + 0.5)   # 推进器激励

v = 0.0
v_hist = []
for i in range(steps):
    acc = (force_in[i] + true_damping(v)) / m
    v = v + acc * dt
    v_hist.append(v)
v_hist = np.array(v_hist)

# 模拟测量：速度差分得到加速度，叠加测量噪声，反推总阻尼力
a_meas = np.diff(v_hist) / dt + np.random.normal(0, 0.15, steps - 1)
v_mid = v_hist[:-1]                     # 与加速度对齐的速度采样
F_meas = m * a_meas - force_in[:-1]     # 测得的"总阻尼力"（含噪声）

# 灰箱关键一步：已知模型只能给出线性部分，剩下的残差交给网络
F_linear = -Y_v * v_mid                 # 已知模型给出的阻尼力
residual = F_meas - F_linear            # 网络的学习目标：残差

print("样本数：", len(v_mid))
print("速度范围：[%.2f, %.2f] m/s" % (v_mid.min(), v_mid.max()))
print("残差范围：[%.2f, %.2f] N" % (residual.min(), residual.max()))

# ========== 2. 划分训练/验证集并转成 tensor ==========
N = len(v_mid)
idx = np.random.permutation(N)          # 教学简化：随机划分
n_train = int(0.8 * N)
tr, va = idx[:n_train], idx[n_train:]

X_train = torch.tensor(v_mid[tr], dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(residual[tr], dtype=torch.float32).reshape(-1, 1)
X_val = torch.tensor(v_mid[va], dtype=torch.float32).reshape(-1, 1)
y_val = torch.tensor(residual[va], dtype=torch.float32).reshape(-1, 1)

# ========== 3. 网络、损失函数、优化器 ==========
class ResidualNet(nn.Module):
    def __init__(self, n_hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

model = ResidualNet()
loss_fn = nn.MSELoss()                              # 均方误差：预测与目标差的平方的平均
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 最常用的优化器

# ========== 4. 训练循环 + 早停 ==========
n_epochs = 800
patience = 60               # 验证 loss 连续 60 轮不改进就提前停止
best_val = float("inf")
best_state = None
wait = 0
train_losses, val_losses = [], []

for epoch in range(n_epochs):
    # ---- 训练四部曲：前向 -> 算 loss -> 反向 -> 更新 ----
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()       # 清空上一轮累积的梯度
    loss.backward()             # autograd 求所有参数的梯度
    optimizer.step()            # 沿梯度方向更新参数

    # ---- 验证：只前向，不求梯度 ----
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_val), y_val).item()
    train_losses.append(loss.item())
    val_losses.append(val_loss)

    # ---- 早停：记录最好的一版参数 ----
    if val_loss < best_val - 1e-5:
        best_val = val_loss
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("早停触发于第 %d 轮" % (epoch + 1))
            break

model.load_state_dict(best_state)   # 恢复验证表现最好的参数
print("最终训练 loss：%.4f，最佳验证 loss：%.4f" % (train_losses[-1], best_val))

# ========== 5. 对比：只用线性模型 vs 线性模型 + 网络残差 ==========
with torch.no_grad():
    res_pred = model(X_val).numpy().flatten()
y_true = y_val.numpy().flatten()
mse_linear_only = np.mean((y_true - 0.0) ** 2)   # 不用网络 = 把残差全当 0
mse_with_nn = np.mean((y_true - res_pred) ** 2)
print("验证集 MSE（只用线性模型）：%.4f" % mse_linear_only)
print("验证集 MSE（线性模型 + 网络残差）：%.4f" % mse_with_nn)
print("误差下降：%.1f%%" % (100 * (1 - mse_with_nn / mse_linear_only)))

# ========== 6. 画图（第 5 章的技能） ==========
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(train_losses, label="train loss")
axes[0].plot(val_losses, label="val loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("MSE")
axes[0].set_yscale("log")
axes[0].legend()
axes[0].set_title("loss curve")

v_grid = np.linspace(v_mid.min(), v_mid.max(), 200).reshape(-1, 1)
with torch.no_grad():
    curve_nn = model(torch.tensor(v_grid, dtype=torch.float32)).numpy()
axes[1].scatter(v_mid[tr], residual[tr], s=4, alpha=0.4, label="train data")
axes[1].plot(v_grid, -Y_vv * v_grid * np.abs(v_grid), "k-", lw=2, label="true residual")
axes[1].plot(v_grid, curve_nn, "r--", lw=2, label="NN output")
axes[1].set_xlabel("v (m/s)")
axes[1].set_ylabel("residual force (N)")
axes[1].legend()
axes[1].set_title("residual curve")

plt.tight_layout()
plt.show()
```

运行（秒级完成），输出：

```
样本数： 799
速度范围：[-2.08, 2.08] m/s
残差范围：[-36.29, 37.66] N
最终训练 loss：3.0509，最佳验证 loss：3.3115
验证集 MSE（只用线性模型）：310.0754
验证集 MSE（线性模型 + 网络残差）：3.3115
误差下降：98.9%
```

**结果解读**：

- 左图：训练 loss 与验证 loss 一起快速下降并趋于平稳，两条曲线几乎贴合——这是**没有过拟合**的健康形态。
- 右图：红色虚线（网络输出）几乎压在黑色实线（真实残差 `−8·v·|v|`）上。网络从没见过这个公式，只通过带噪声的数据就把这条弯曲的曲线学了出来。
- 验证集上，"线性模型 + 网络残差"的均方误差比"只用线性模型"下降了约 98.9%。重申一次：这是教学简化场景的数字，只说明方法有效，不代表实船收益。

**过拟合与早停的直觉**：网络参数太多、训练太久时，它会把训练数据里的噪声也背下来——训练 loss 继续降，验证 loss 却开始回升，这就是过拟合（overfitting）。应对办法叫**早停**（early stopping）：每轮都检查验证 loss，把表现最好的那版参数存下来，连续若干轮没有改进就停止训练并恢复最好版本。本例数据充足、网络不大，800 轮内验证 loss 仍在缓慢改善，早停没有触发——它更像一根保险丝。想亲眼看到它触发，去做练习 3。

**这和项目的关系**：恭喜你，刚完成了 `research/01-方法对比矩阵.md` 路线③的最简化版本——已知模型（相当于 MMG 骨架）给出线性阻尼，神经网络只补残差，合成力 = 物理项 + 网络项。真实项目把输入从单个 `v` 扩展到 `(u, v, r, n1, n2, α1, α2)` 等更多特征、把网络嵌入完整的 3DOF 仿真，并在数据范围外把网络输出置零、退回物理模型，但核心代码结构与本例完全相同。项目选定的长期路线④则让网络承担更大的子模块，训练方式依然如此。

## 常见错误

**错误 1：float64 与 float32 混用**

```python
x64 = torch.from_numpy(np.array([[0.5]]))   # numpy 默认 float64
model(x64)   # RuntimeError: mat1 and mat2 must have the same dtype, but got Double and Float
```

正确写法：进网络前统一类型——`torch.tensor(x, dtype=torch.float32)`，或对已有 tensor 用 `x64.float()`。

**错误 2：忘记 `optimizer.zero_grad()`**

PyTorch 的梯度默认**累加**。少了这行，每一轮的梯度都叠在上一轮上，loss 会乱跳甚至不降反升。训练四部曲的固定顺序是：`zero_grad() → backward() → step()`（前向算 loss 在它们之前）。

**错误 3：对需要梯度的 tensor 调 `.numpy()`**

```python
F = (v ** 2).sum()
F.numpy()    # RuntimeError: Can't call numpy() on Tensor that requires grad...
```

正确写法：`F.detach().numpy()`，或只要数值时用 `F.item()`（返回 Python 标量，记录 loss 时最常用）。做预测时干脆整段放进 `with torch.no_grad():`，既省事又更快。

**错误 4：目标形状 (N,) 与预测 (N,1) 的广播陷阱**

```python
pred = model(X_train)                              # 形状 (N, 1)
y = torch.tensor(residual, dtype=torch.float32)    # 形状 (N,) ← 少了 reshape
loss_fn(pred, y)   # 不报错（或只给警告），但广播成 N×N 矩阵，loss 是错的！
```

实测同样的数据，两种形状算出的 MSE 分别是 1.2623 和 1.3443——错误的那种不会崩溃，只会悄悄污染训练。正确写法：构造标签时就 `reshape(-1, 1)`，保证 `(N, 1)` 对 `(N, 1)`。

## 动手练习

1. **互转热身**：用 `np.linspace(0, np.pi, 5)` 生成 5 个艏向角 ψ（船头指向的角度，弧度），转成 `float32` 的 tensor，计算每个角度的 sin 和 cos，把结果排成 (5, 2) 的数组转回 NumPy 打印。
2. **autograd 验算**：本章真实残差的形式是 `F(v) = c·v·|v|`。对 `v = -0.5` 用 autograd 求 `dF/dv`，并与手算结果 `2·c·|v|`（取 c=2.0）对比。注意 v 是负数，想想为什么手算公式里是 |v|。
3. **见过拟合**：把 9.5 节代码中的 `n_hidden` 从 32 改成 128、`n_epochs` 从 800 改成 3000，重新运行。观察：早停是否在第几百轮触发？最终训练 loss 与最佳验证 loss 的差距是否变大？用自己的话解释看到的两条 loss 曲线。

## 参考答案

**练习 1**：

```python
import numpy as np
import torch

psi_np = np.linspace(0, np.pi, 5)
psi_ts = torch.tensor(psi_np, dtype=torch.float32)
result = torch.stack([torch.sin(psi_ts), torch.cos(psi_ts)], dim=1).numpy()
print(np.round(result, 4))
```

输出（`[[sin, cos], ...]`，从 0 到 π 均匀采样）：

```
[[ 0.      1.    ]
 [ 0.7071  0.7071]
 [ 1.     -0.    ]
 [ 0.7071 -0.7071]
 [-0.     -1.    ]]
```

**练习 2**：

```python
import torch

c = 2.0
v = torch.tensor(-0.5, requires_grad=True)
F = c * v * torch.abs(v)
F.backward()
print("dF/dv =", v.grad.item(), "（手算：2·c·|v| =", 2 * c * abs(-0.5), "）")
```

输出：`dF/dv = 2.0 （手算：2·c·|v| = 2.0 ）`。因为 `v·|v|` 对 v 求导得 `2|v|`：v 为负时 `v·|v| = −v²`，导数是 `−2v = 2|v|`；v 为正时导数是 `2v = 2|v|`。两种情况统一成 `2|v|`，这正是该形式在阻尼建模中好用的原因——力永远与速度方向相反。

**练习 3**（参考现象）：实测改为 `n_hidden=128`、`n_epochs=3000` 后，早停在第 579 轮左右触发，最终训练 loss 约 2.98、最佳验证 loss 约 3.30，两者差距比原来更明显。解释：网络变大后更容易在训练集上"越钻越深"，而验证集很早就到达瓶颈；早停替我们守住了验证表现最好的那一版参数。如果再叠加减少训练数据（如 `n_train = int(0.2 * N)`），过拟合会出现得更早、更明显。

## 本章小结与下一章预告

- tensor 是 PyTorch 版的 NumPy 数组，`torch.from_numpy` / `.numpy()` 互转，注意 float32 与共享内存；
- `requires_grad=True` + `backward()` + `.grad` 是 autograd 的三件套，梯度不用手推；
- `nn.Linear` 是积木，`nn.Sequential` 串积木，继承 `nn.Module` 写 `forward` 是标准做法；
- 训练循环四部曲：前向算 loss → `zero_grad()` → `backward()` → `step()`；用验证集监控过拟合，用早停保住最好参数；
- 你已经实现了项目"MMG+NN 残差"路线的最简版：物理模型打底，网络只学残差。

下一章《第 10 章 强化学习实战入门》将站在本章的肩膀上：网络不再拟合一条已知的力曲线，而是通过与仿真环境不断交互、试错，自己学会输出推进器指令（n1、n2、α1、α2）让船到达目标点——那时今天写的 `nn.Module` 和训练循环会直接变成 RL 里的"策略网络"。

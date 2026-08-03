# 使用 CNN 提升标签与冻结 VAE 的表示对齐

日期：2026-08-03

## 目标定义

这里的“对齐 VAE”不应定义为让分割、深度或法线 latent 与自然 RGB 图像 latent 具有相同分布，而应定义为：

> 标签经过冻结 VAE 编码和解码以后，原生任务信息仍可被稳定恢复。

强行用 MMD、GAN 或对比损失把标签 latent 拉向 RGB latent，可能破坏离散类别、度量深度和方向结构，不作为第一选择。

## 推荐实施顺序

### 方案 A：只增加 VAE 后置任务解码 CNN

这是最低风险的第一步：

```text
训练目标 y -> deterministic codec C_k -> frozen VAE encoder -> z_0

推理生成 z_hat -> frozen VAE decoder -> task decoder B_k -> y_hat
```

训练 (B_k) 时使用真实标签的 VAE 回环样本：

\[
\hat y = B_k(D_V(E_V(C_k(y))))
\]

优点：

- 不改变当前扩散目标 latent，保留现有主方案和预训练兼容性；
- 推理闭环完整，不需要真值；
- 能学习修复 VAE 平滑边界、通道耦合和小幅数值漂移；
- 实现、消融和失败回退都最简单。

限制：如果 VAE 已彻底丢失某些标签信息，后置 CNN 无法凭空恢复。

### 方案 B：成对的 pre/post-VAE CNN

当方案 A 的 held-out 回环指标仍明显低于无 VAE 的 codec 上限时，再采用：

```text
训练目标：
y -> C_k -> A_k -> frozen VAE encoder -> z_0 -> diffusion

推理输出：
diffusion -> z_hat -> frozen VAE decoder -> B_k -> C_k 空间 -> C_k.decode -> y_hat
```

其中：

- (A_k)：把确定性 codec 表示调整为更容易被冻结 VAE 保存的三通道表示；
- (B_k)：把 VAE 解码结果还原到原来的确定性 codec 空间；
- (A_k) 只在训练目标编码时使用；
- (B_k) 必须在推理时使用，并随 checkpoint 保存和加载。

建议将前置 CNN 限制为小残差：

\[
A_k(c)=\operatorname{clip}\left(c+\alpha_k\tanh(f_k(c)),-1,1\right)
\]

其中 (c=C_k(y))，(f_k) 最后一层零初始化，初始时 (A_k(c)\approx c)。这样可以降低表示坍塌和语义漂移风险。

## 对齐模块的预训练

冻结 VAE，先单独训练 (A_k,B_k)：

\[
c=C_k(y),\quad
r=A_k(c),\quad
\tilde r=D_V(E_V(r)),\quad
\hat c=B_k(\tilde r)
\]

推荐损失：

\[
\mathcal L_{	ext{align}}
=\lambda_c\|\hat c-c\|_1
+\lambda_{task}\mathcal L_{task}(\hat c,y)
+\lambda_{id}\|A_k(c)-c\|_1
+\lambda_{struct}\mathcal L_{struct}
\]

- `codec reconstruction`：保证输出仍能回到确定性 codec 空间；
- `task loss`：直接保护原生任务语义；
- `identity/residual loss`：限制前置 CNN 不要发明任意编码；
- `structure loss`：按任务保护边界、梯度或方向。

只有 held-out 数据的任务回环指标提升、输出范围合法且无坍塌后，才冻结 (A_k,B_k) 并训练扩散模型。第一轮不建议把 CNN、VAE 和扩散网络全部联合更新。

## 三类任务的 CNN 输出和损失

### Query-conditioned segmentation

```text
B_seg output: [B,1,H,W] logits
loss: BCEWithLogits + Dice + boundary loss
```

训练使用 soft probability，评测时才阈值化。不要在训练图中使用不可导的 hard threshold，也不要把整数 class ID 直接送入 CNN。

### Depth

```text
B_depth output: [B,1,H,W] normalized depth
loss: masked L1/Charbonnier + gradient loss
```

继续使用数据集级固定的 `min_depth/max_depth` 和固定 inverse/log/linear 变换。禁止使用推理时拿不到的单张真值均值、方差或 min/max。

### Surface normal

```text
B_normal output: [B,3,H,W]
postprocess: normalize to unit vector
loss: 1 - cosine_similarity + optional gradient consistency
```

必须固定坐标系、通道顺序和水平翻转规则。

## 可选的低成本前置检查

在增加 CNN 之前，先统计三任务 target latent 的每通道均值、标准差、极值和有效区域方差。如果主要问题只是各任务 latent 尺度不一致，可先尝试使用训练集固定统计量的可逆仿射归一化：

\[
u_k=(z_k-\mu_k)/(\sigma_k+\epsilon),\qquad
z_k=u_k\sigma_k+\mu_k
\]

统计量必须来自训练集并固定保存，不能在推理时从未知目标计算。该方案比可学习 CNN 更容易验证和反变换。

## 实验顺序

1. `deterministic codec + frozen VAE`：当前基线。
2. `基线 + post-VAE decoder B_k`：推荐的首个 CNN 实验。
3. `paired A_k/B_k`：仅在方案 2 无法达到任务回环要求时启用。
4. `paired A_k/B_k + diffusion`：先冻结对齐模块训练扩散。
5. 可选小学习率联合微调：只有前四步稳定后再尝试。

每组固定数据划分、分辨率、VAE、seed 和评测器，至少报告：

- codec-space MSE/MAE；
- 分割 IoU、边界指标；
- 深度 AbsRel、RMSE；
- 法线平均/中位角误差；
- latent 每通道统计量；
- 原始 codec、VAE 重建、CNN 修复和误差图。

## 当前项目建议

当前正式配置继续保持 `target_adapter.enabled: false`。新增 CNN 时，优先实现推理端可用的 `TaskPostVAEDecoderBank`，完成三任务真实样本回环消融后，再决定是否把现有 `ResidualPreVAEAdapter` 升级为成对的 `TaskPreVAEEncoderBank + TaskPostVAEDecoderBank`。

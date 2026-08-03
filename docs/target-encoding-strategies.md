# 三任务目标编码策略

状态：代码契约已实现；真实 VAE 保真度数值需按 `docs/run-cookbook.md` 在服务器生成。

范围固定为语义分割、单目深度和表面法线，不包含光流。

## 1. 统一契约

所有原生目标先由确定性的 `TargetCodec` 转成 `float32 [3,H,W]`、值域 `[-1,1]` 的伪 RGB 表示，同时保留 `bool [1,H,W]` 有效掩码。随后 `VisualLatentPathway` 使用同一个冻结 SD2 VAE 编码 RGB 图像和任务目标：

```text
native target -> TargetCodec -> optional bounded pre-VAE residual adapter
              -> frozen SD2 VAE encoder -> target latent
```

codec 负责语义、单位、通道和值域；VAE 只负责连续图像压缩。禁止把整数 class ID 直接交给无约束 CNN 或 VAE。

## 2. 分割：离散目标

默认训练策略是 class-query 二值掩码。对查询类别 `k`：

```text
m_k(x) = 1[label(x) = k]
encoded(x,c) = 2 * m_k(x) - 1, c in {R,G,B}
```

三通道完全复制，VAE 重建后先对三通道求均值，再映射到 `[0,1]` 分数并按阈值解码。标准 ADE20K 推理必须对数据集完整 150 类词表生成查询，再在全部类别分数图上取最大值；候选类别不得读取当前样本 GT。

训练 query 与测评候选必须分开理解：`uniform`/`fixed` 从完整数据集词表独立采样，`mixed` 仅可在训练期读取 GT 来平衡正负监督；后者是标签采样策略，绝不能进入验证或推理。极小 overfit gate 为避免每轮目标变化，预先固定为词表 class 0，并通过保真度报告检查前景占比。所有验证/推理候选只由数据集词表生成并覆盖全部 150 类。

离散目标的主要风险是 VAE 平滑边界以及产生介于前景/背景之间的连续值。palette 和 normalized-ID codec 仅用于诊断消融；normalized ID 的邻近数值不代表语义相近，因此不是默认方案。

## 3. 深度：单通道连续目标

深度先裁剪到配置范围 `[d_min,d_max]`，可选择 `linear`、`inverse` 或 `log` 变换 `f(d)`：

```text
u = 2 * (f(d) - f_min) / (f_max - f_min) - 1
encoded = repeat(u, 3 channels)
```

解码时三通道求均值、裁剪至 `[-1,1]`、执行相反的仿射映射和 `f^-1`，最后恢复米制深度。无效深度不进入扩散损失和评测。

深度是连续但具有度量结构的目标。VAE 可能损失细边缘和小尺度变化；裁剪饱和会造成不可逆损失，必须把配置的深度范围写入实验记录。

## 4. 法线：三通道连续方向目标

法线采用 camera-space `xyz` 三通道单位向量，天然位于 `[-1,1]`：

```text
encoded = n / ||n||_2
decoded = reconstructed / ||reconstructed||_2
```

解码后的再次单位化是必要步骤。坐标系、通道顺序、Y 轴方向和水平翻转规则必须由数据配置固定。VAE 的通道耦合及边界平滑会转化为角度误差。

## 5. 可学习 pre-VAE adapter

可选 `ResidualPreVAEAdapter` 只在确定性 codec 后学习有界三通道残差，最后一层零初始化并裁剪到 `[-1,1]`。它不拥有任务语义，也不能替代 codec。

该 adapter 当前没有声明可逆解码器，因此标准训练/推理配置默认关闭；端到端推理会拒绝启用了该模块的配置。它仅用于 VAE 表示能力消融，直到设计并验证成对的逆变换。

## 6. 保真度报告

`scripts/analyze_vae_reconstruction.py` 对三任务输出：

- 公共 codec-image 指标：MSE、MAE、PSNR；
- 分割：二值 IoU、像素准确率；
- 深度：AbsRel、RMSE；
- 法线：平均/中位角度误差；
- 每样本 `原编码 / VAE 重建 / 放大误差` 可视化；
- JSON、CSV 和 Markdown 汇总。

只有公共 encoded-space MSE 可以机械排序三种表示的 VAE 误差，但它不能代替任务指标。实际“信息损失最大目标”必须由同一模型、分辨率、样本预算和有效掩码下的运行报告确定，不能在运行前预设结论。

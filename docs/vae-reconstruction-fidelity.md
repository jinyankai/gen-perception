# 编解码保真度分析：信息损失最大的目标及原因

## 1. 结论速览

同一冻结 VAE 下，三任务目标经 encode→decode 往返的信息损失排序（大→小）：

**法线（normal）≫ 分割（segmentation）≫ 深度（depth）**

以公共 codec-image MSE 衡量，法线约为分割的 33 倍、深度的 800 倍。但公共 MSE 只能机械排序，真正的可用性要看各任务原生指标。

## 2. 三任务横向对比

| 任务 | encoded MSE | PSNR (dB) | 任务原生指标 | 分辨率 |
|---|---|---|---|---|
| depth | 7.2e-05 | 48.70 | AbsRel 0.010, RMSE 0.038 m | 480×640 |
| segmentation | 1.7e-03 | 35.42 | binary IoU 0.873, PixAcc 0.9995 | ADE20K |
| normal | 5.8e-02 | 18.49 | mean 19.48°, median 15.90° | 576×768 |

> 提示：公共 MSE 跨任务可比，因为都在同一 `[-1,1]` 伪 RGB 空间度量；原生指标（IoU / AbsRel / 角度误差）跨任务不可直接比较。

可视化：各任务三联图整理在 `output/vis/<task>/visualizations/` 下,每样本一张 `原编码 | VAE重建 | 放大误差(×4)`。下面各小节各选一张代表图说明损失的空间分布。

## 3. 逐任务分析

### 3.1 深度 — 近无损

PSNR 48.7 dB、AbsRel 仅 1.0%。深度经 codec 归一化为**单通道**信号后复制到三通道（`DepthCodec.encode`，`codecs/depth.py:69`），是低频、通道完全相关、色彩多样性极低的图像——正是自然图像 VAE 最擅长压缩的形态。

残余损失来自两处：细边缘的高频细节被平滑；深度范围 `[0.1, 10.0]` 裁剪饱和造成的不可逆损失（配置 `configs/depth/nyuv2.yaml:14-15`）。两者在 NYUv2 室内尺度下都很小。**VAE 基本不构成深度任务的上限。**

![深度重建三联图（0003）](../../output/vis/depth/visualizations/depth__nyuv2__train__0003.png)

*深度 `0003`：重建与原编码几乎无差，误差×4 图整体接近纯黑,仅在物体细边缘有微弱残留。*

### 3.2 分割 — 内部无损，边界损失在复杂场景显著

PixAcc 0.9995、IoU 0.873。像素准确率极高但 IoU“只有”0.87，二者的差距揭示了损失的位置：**边界**。默认 codec 是 class-query 二值掩码，`{0,1}→{-1,+1}` 复制三通道（`SegmentationBinaryMaskCodec`，`codecs/segmentation.py:36-38`）。

- VAE 的 8× 下采样把 mask 边缘糊成前景/背景之间的连续值，解码时按 0.5 阈值切分（`decode`，`codecs/segmentation.py:57`）在边界产生偏移。
- IoU 对边界敏感，PixAcc 被大面积正确的内部区域拉高，因此二者分化。
- 前景占比 0.35，目标不大，少量边界误分即可拉低 IoU。

![分割重建三联图（ADE_train_00000001）](../../output/vis/seg/visualizations/segmentation__ade20k__training__ADE_train_00000001.png)

*分割 `ADE_train_00000001`：大块内部全黑,误差只在细长结构轮廓上勾出亮线。*

可视化实测（`ADE_train_00000001` 三联图）证实了损失的分布：大块区域（天空、地面、墙面）重建几乎无差，误差×4 图中内部区域全黑，**所有误差集中在物体轮廓**。但该场景细长结构密集（成排电线杆、树干、树冠、前景零散小物体），边界总长度大，误差在视觉上"铺满"全图——即单幅指标（IoU 0.87）与主观"差距较大"的印象是一致的：**损失不是大面积错分，而是边界损失在边缘密集场景中被放大，且恰好落在语义关键的细长结构上**。这类结构是分割在冻结 VAE 下最脆弱的部分，靠提高分辨率与边界后处理缓解，而非更换 VAE。

离散目标的固有风险：VAE 产生介于类别之间的连续值。这也是默认用二值掩码、而非 `SegmentationPaletteCodec` / `SegmentationIdCodec` 的原因——后两者的邻近数值/颜色不代表语义相近，VAE 扰动会导致语义上任意的类别串换，仅作诊断消融（`codecs/segmentation.py:158-163`）。

### 3.3 法线 — 信息损失最大

PSNR 18.5 dB、mean 19.48°、median 15.90°，是三者中唯一损失显著的目标。原因是法线的信号形态与自然图像最不匹配：

1. **天然三通道且强耦合**：法线把方向 `(x,y,z)` 编码进 RGB 三通道（`NormalCodec`，`codecs/normal.py`），三通道各自独立且高频，不像 depth/seg 的三通道是复制出来的冗余。VAE 的通道耦合与边界平滑直接转化为角度误差。
2. **要求单位向量精度**：解码后必须重新单位化（`codecs/normal.py:44`），较小的逐分量 MSE 会被放大成可观的角度误差。
3. **分辨率/长宽比敏感**（见第 4 节）。

![法线重建三联图（0003）](../../output/vis/normal/visualizations/normal__nyuv2__train__0003.png)

*法线 `0003`：误差×4 图里墙面/桌面整片布满彩色噪点,平坦大色块反而低——损失弥漫在高频纹理面上,不勾边界。*

可视化实测（`0003` 三联图）揭示了损失的空间分布，与分割相反：**误差并非集中在物体边界，而是弥漫在整个高频法向纹理表面**。误差×4 图中，布满细密起伏的墙面、桌面整片呈彩色噪点，而少数真正平坦的大色块区域误差反而低。重建图可见大色块的主方向大致保留，但 GT 中清晰的斜向细纹在重建里被抹平、方向漂移。

即：VAE 系统性地**抹除高频法向细节**，这正是法线信号（高频、三通道独立、要求单位向量精度）与自然图像 VAE 最不匹配之处的直接体现，也解释了为何 mean(19.48) > median(15.90)——纹理越密的样本损失越大，拉出长尾。这与我此前基于数值的“掠射角/深度跳变伪影”推断不同：主因是纹理区的高频抹除，而非局部几何伪影。

## 4. 法线损失可被分辨率大幅缓解

对 NYUv2 法线做的分辨率对照（16 样本，SD2-base 冻结 VAE，mean angular error）：

| 分辨率 | 长宽比 | mean | median | encoded MSE |
|---|---|---|---|---|
| 480×640 | 3:4 原生 | 22.40° | 18.47° | 0.0725 |
| 512×512 | 1:1 方形（畸变） | 24.23° | 19.82° | 0.0832 |
| 576×768 | 3:4 原生 | 19.48° | 15.90° | 0.0581 |
| 768×768 | 1:1 方形 | 17.95° | 14.56° | 0.0508 |

三点结论：

1. **提高分辨率显著降低损失**：从 480 到 768，mean 下降约 4–6°。VAE 的 8× 下采样是主因，像素越多、单个 latent cell 覆盖的角度变化越少。
2. **方形裁剪会引入畸变**：512×512 反而比 480×640 差，因为把原生 4:3 的 NYUv2 强拉成正方形扭曲了几何，法线对此敏感。故法线一律用保长宽比配置（`configs/normal/nyuv2_res576x768.yaml`）。768×768 虽也是方形，但分辨率增益盖过了畸变损失，所以仍最低。
3. **576×768 介于 480 与 768 之间**：说明纯分辨率增益略微压过长宽比畸变，但未完全抵消——若要最低天花板需同时高分辨率 + 保长宽比。

**关键含义**：这些角度误差是**上界**。即使下游 latent 预测完美，解码后法线的角度误差也降不到这个数以下。576×768 的 19.5°（768×768 的 18.0°）已落在 Marigold 端到端结果（NYU 约 18–20°）附近——说明冻结的 SD2 VAE 并非灾难性瓶颈，之前 24° 的印象主要是低分辨率造成的假象。

## 5. 对训练策略的影响

- **深度、分割**：VAE 不是瓶颈，可直接在冻结 latent 空间训练 U-Net。分割的边界损失可通过更高分辨率与后处理缓解。
- **法线**：不必微调或更换 VAE。Marigold 自身所有 trainer 也冻结 VAE、只训 U-Net（`marigold_normals_trainer.py:102`），其优秀结果来自 U-Net 在冻结 latent 空间学习映射，而非更好的 VAE。参见记忆 `freeze-vae-train-unet`。
- 若法线仍不理想，`model.yaml:36` 预留的 pre-VAE `target_adapter`（当前 `enabled: false`）是**不动 VAE** 前提下改善法线表征的入口；但它尚无配对的可逆解码器，端到端推理会拒绝启用它的配置（[target-encoding-strategies.md](target-encoding-strategies.md) 第 5 节）。

## 6. 复现

```bash
python scripts/analyze_vae_reconstruction.py --config configs/depth/nyuv2.yaml --task depth --output-dir outputs/vae_recon/depth --limit 16 --precision fp32 --overwrite
python scripts/analyze_vae_reconstruction.py --config configs/segmentation/ade20k.yaml --task segmentation --output-dir outputs/vae_recon/seg --limit 16 --precision fp32 --overwrite
python scripts/analyze_vae_reconstruction.py --config configs/normal/nyuv2_res576x768.yaml --task normal --output-dir outputs/vae_recon/normal_576x768 --limit 16 --precision fp32 --overwrite
```

局限：每任务仅 16 样本、诊断性而非正式实验；法线 GT 由深度现算，含掠射角/深度跳变伪影；分割为 ADE20K class-query 二值掩码，多类整图指标需单独评测。正式结论须在固定模型、分辨率、样本预算与有效掩码下重跑。

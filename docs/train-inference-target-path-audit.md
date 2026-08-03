# 训练—推理目标路径审查

日期：2026-08-03
审查范围：当前三任务统一生成式感知方案、默认配置、训练前向、latent 采样、任务 codec 与可学习 pre-VAE CNN 消融支路。

## 结论

1. **默认主方案不存在“推理时拿不到训练目标”的目标泄漏问题。**训练真值只用于构造干净目标 latent 和加噪样本；去噪器没有把干净目标或 CNN 真值特征作为额外条件。
2. **可学习 pre-VAE CNN 支路确实没有完整的推理解码闭环。**它改变了 codec 输出，但当前没有对应逆变换；因此只能作为分析消融，不能作为正式训练—推理方案。
3. 当前实现已经采取两项保护：默认配置关闭该 CNN；端到端推理在检测到它启用时直接报错。

## 默认主方案的数据流

### 训练

```text
输入图像 x ─────────────────> frozen VAE encoder ─> z_img ───────────┐
                                                                    │
真值目标 y -> deterministic TargetCodec -> frozen VAE encoder -> z_0
                                                         │          │
                                                         └-> 加噪 -> z_t

SharedUNet(z_img, z_t, timestep, task token, optional text) -> 预测噪声
```

这里的 (z_0) 是训练监督空间，类似分类训练中的标签。它只用于生成 (z_t) 和计算噪声预测损失，不作为独立条件送入 U-Net。

### 推理

```text
输入图像 x -> frozen VAE encoder -> z_img
随机噪声 z_T -> SharedUNet 多步去噪（条件为 z_img、task token、可选 text）
              -> 预测 target latent -> frozen VAE decoder
              -> deterministic TargetCodec.decode -> 任务输出 y_hat
```

推理时不知道真值内容是正常的；只需要知道任务类型，例如 segmentation、depth 或 normal。目标 latent 从随机噪声开始生成，不需要调用目标 VAE encoder。

## 代码证据

- `configs/base/model.yaml:36-37`：`target_adapter.enabled: false`，默认使用确定性 codec 路径。
- `perception_diffusion/models/visual_latent.py:133-153`：训练目标经过 target adapter 后由 VAE 编码为目标 latent。
- `perception_diffusion/training/unified_trainer.py:135-144`：U-Net 输入是图像 latent、加噪目标 latent、时间步和任务/文本条件；没有额外输入干净目标特征。
- `perception_diffusion/inference/latent_sampler.py:61-105`：推理目标 latent 从随机噪声初始化并迭代去噪。
- `perception_diffusion/inference/runner.py:90-100`：生成 latent 只经过 VAE decoder 恢复到 codec 图像空间。
- `perception_diffusion/inference/runner.py:125-129`：若启用无逆变换的可学习 pre-VAE adapter，推理直接拒绝运行。
- `perception_diffusion/codecs/depth.py:49-85`：深度编解码使用配置固定的深度上下界和可逆变换，不依赖单张推理真值的统计量。
- `perception_diffusion/codecs/segmentation.py:23-57`、`perception_diffusion/codecs/normal.py:21-45`：分割与法线均有明确的确定性 encode/decode。

## 可学习 pre-VAE CNN 的实际问题

消融支路当前为：

```text
y -> TargetCodec -> CNN adapter A_k -> frozen VAE encoder -> z_0
```

而现有推理后半段只有：

```text
z_hat -> frozen VAE decoder -> A_k(codec(y)) 的近似表示
```

缺少：

```text
A_k(codec(y)) -> codec(y)
```

因此不能直接调用原来的 `TargetCodec.decode` 得到可靠任务输出。此外，该 CNN 若只由扩散噪声损失端到端更新，没有重建、任务指标或成对逆变换约束，可能学习到有利于降低扩散损失、却不再保留任务语义的表示，甚至发生表示收缩或坍塌。

## 建议

### 当前阶段

- 正式单任务和多任务实验保持 `model.target_adapter.enabled: false`。
- `configs/ablations/learnable_pre_vae_adapter.yaml` 仅用于表示能力分析，不用于报告正式推理指标。
- 每次实验保存解析后的完整配置，并在启动时记录 `target_adapter.enabled`，避免误用消融配置。
- 先完成确定性 codec 的 `codec -> VAE -> codec decode` 回环保真度，再进入长训练。

### 如果以后确实需要可学习 CNN

必须改成成对模块：

```text
训练：y -> Codec -> A_k -> VAE encoder -> diffusion
推理：diffusion -> VAE decoder -> B_k -> Codec.decode -> y_hat
```

其中 (A_k) 是任务编码器，(B_k) 是任务解码器。先单独训练并验证：

\[
B_k(\operatorname{VAEdec}(\operatorname{VAEenc}(A_k(C_k(y))))) \approx C_k(y)
\]

需要同时使用 encoded-space 重建损失和原生任务损失；回环通过后冻结或严格约束 (A_k,B_k)，再训练扩散模型。W1/W2 阶段不建议增加这项复杂度。

## 验证记录

2026-08-03 在本地运行：

```text
python -m unittest discover -s tests -p "test_*.py"
```

结果：61 项测试全部通过。该结果证明现有接口、配置保护和结构测试通过，但不能代替真实 SD2 权重与真实数据上的 VAE 回环、过拟合和端到端推理验证。

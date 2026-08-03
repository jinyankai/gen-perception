# 真实训练与推理实现

## 模块边界

- `models/pretrained.py`：只从本地 SD2 目录加载 tokenizer、CLIP、VAE、U-Net 和 scheduler，不隐式联网。
- `models/visual_latent.py`：冻结 VAE 的图像/目标编码、解码、scaling factor 和保守有效掩码下采样。
- `models/conditioning.py`：task token、task adapter、可选 text token，以及保持 token 长度不变的条件开关。
- `training/noise.py`：Marigold 风格多分辨率噪声和按 timestep 线性退火的强度。
- `training/runner.py`：三任务共用的 DataLoader 调度、AMP、梯度累积、优化、日志和 checkpoint 循环。
- `inference/runner.py`：checkpoint 加载、DDIM/DDPM latent 采样、VAE 解码、task codec 解码和证据输出。

## 训练数据流

```text
raw batch
  -> deterministic codec target [-1,1]
  -> frozen CLIP prompt tokens
  -> frozen VAE image/target latents
  -> annealed multi-resolution target noise
  -> concat(image latent, noisy target latent)
  -> one task-token-conditioned shared U-Net
  -> masked epsilon MSE
  -> optimizer / checkpoint / JSONL + TensorBoard (+ optional W&B)
```

多任务配置在 batch 之间 round-robin；一个 batch 内始终同任务。训练 runner 当前是单进程实现，DDP 会显式拒绝，直到服务器 NCCL gate 通过。

## 条件消融

Conditioner 和推理 CLI 支持：

| 模式 | Task token | Text token |
| --- | --- | --- |
| `full` | 开 | 开 |
| `task_only` | 开 | 置零 |
| `text_only` | 置零 | 开 |
| `unconditional` | 置零 | 置零 |

关闭条件时 token 数量和 cross-attention 形状保持不变，避免把序列长度变化混入消融结果。`scripts/validate_conditions.py` 固定图像、初始噪声、采样步数和随机种子，保存各模式/任务/prompt 的输出与差异报告。

## Checkpoint 与日志

checkpoint 保存共享 denoiser、目标 adapter、优化器、LR scheduler、GradScaler、专用 generator，以及 Python/NumPy/PyTorch RNG 状态。恢复后继续写入原实验目录。

每个训练目录包含：

```text
config.yaml
command.txt
git_commit.txt
environment.txt
train.log
metrics.jsonl
metrics.json
tensorboard/
checkpoints/
predictions/
visualizations/
```

W&B 默认关闭并使用可选依赖；JSONL 与 TensorBoard 是离线证据主路径。

## 证据边界

代码和本地测试只能证明接口、数值检查、checkpoint 与 tiny 结构路径成立。loss 收敛、VAE 真实目标保真度、任务/prompt 语义正确性、过拟合预测和正式指标必须由服务器实际运行产物证明。执行顺序见 `docs/run-cookbook.md`。

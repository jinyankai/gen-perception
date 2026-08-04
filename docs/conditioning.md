# 任务 / 文本条件注入（Task / Text Conditioner）

本文件说明 W4 交付的条件注入模块：架构、开关接口、正确性验证方法与消融建议。
代码位于 `perception_diffusion/models/conditioning.py`、`adapters.py`，
消融脚本为 `scripts/validate_conditions.py`。

## 1. 架构

单一共享 U-Net 靠 `encoder_hidden_states` 区分任务。该条件张量由
`TaskTokenConditioner` 构建，流程：

```
task_name --> 可学习 task token 组 (nn.Embedding, 每任务 num_task_tokens 个)
          --> 每任务 ResidualConditionAdapter (TaskAdapterBank 按 task_name 分派)
optional text prompt --> 冻结 CLIP text encoder --> Linear 投影到 cross_attention_dim
          --> concat([task_tokens, text_tokens], dim=1)
          --> encoder_hidden_states  [B, num_task_tokens(+L_text), cross_attention_dim]
```

- **task token**：每个任务拥有独立的 `num_task_tokens` 个 embedding（默认 4）。
  即使深度与法线都用空文本，其条件状态也不同，模型据此区分任务。
- **task adapter**：`ResidualConditionAdapter` 是一个 LayerNorm→down→SiLU→up 的
  残差瓶颈，`residual_scale` 初始为 0（启动时等价恒等，训练中再长出任务特化）。
- **text token**：可选。冻结 CLIP 的 `last_hidden_state` 经 `text_projection`
  （维度相同则为 `nn.Identity`）投影后拼在 task token 之后。开放词表分割用它
  携带类别名。

## 2. 多尺度注入

条件是**单一** `encoder_hidden_states` 张量，`UnifiedPerceptionDenoiser` 将其送入
共享 U-Net。Diffusers 的 `UNet2DConditionModel` 会在 **down / mid / up 各分辨率**的
`attn2`（cross-attention）层重复消费同一条件——因此这就是 v1 的"多尺度注入"：
任务信息在每个尺度都参与，而非只在入口注入一次。

真·**per-scale 独立条件**（每个 U-Net 尺度用不同条件张量，需自定义 `AttnProcessor`
或 per-block 投影 / LoRA）属于框架文档中的 **v2 Option C**，是负迁移的补救方案，
本模块**不实现**。参见 `docs/unified-perception-framework.md` 方案对比表 C 行。

## 3. 条件开关接口

两个布尔开关可分别启停任务条件与文本条件，用于消融。它们在 config 里设默认值
（`configs/base/model.yaml`），也可在每次 forward / sample 时覆盖：

- `use_task_condition`：False 时把 task token 段整体置零。
- `use_text_condition`：False 时把 text token 段整体置零（不影响 task token）。

置零而非删除 token，保证条件张量形状不变、U-Net 前向不受影响。开关沿
`TaskTokenConditioner.forward` → `UnifiedPerceptionDenoiser.forward` →
`UnifiedLatentSampler.sample` → `run_inference` 全程透传。

`scripts/infer.py` / `scripts/validate_conditions.py` 用命名模式表达两个开关的组合，
由 `condition_mode_switches`（`perception_diffusion/inference/runner.py`）解析：

| mode | use_task_condition | use_text_condition | 含义 |
| --- | --- | --- | --- |
| `full` | True | True | 任务 + 文本条件全开（基线） |
| `task_only` | True | False | 仅任务 token，文本置零 |
| `text_only` | False | True | 仅文本，任务 token 置零 |
| `unconditional` | False | False | 两者全关；输出与任务名无关，可复现 |

非法 mode 抛 `ValueError`。

## 4. 正确性验证

### 4.1 单元测试（无需训练，本地 CPU 即可）

随机初始化即可证明**接线是活的**，与权重好坏无关：

- `tests/test_models.py::ConditioningTest`
  - task/text token 拼接与形状；
  - 全关时条件张量为零；
  - 只关文本时 task token 段保留、text 段为零。
- `tests/test_training_and_inference.py::ConditionSwitchTest`
  - `condition_mode_switches` 四模式映射 + 非法模式报错；
  - 关任务条件使采样输出端到端改变；
  - `unconditional` 下同图同噪声、不同任务名输出一致（可复现）。

运行：

```bash
python -m unittest tests.test_models tests.test_training_and_inference
```

### 4.2 语义验证脚本（需要 checkpoint）

```bash
python scripts/validate_conditions.py \
  --config configs/multitask/stage1_shared_unet.yaml \
  --checkpoint <CKPT> --task depth \
  --output-dir <OUT> --num-steps 4 \
  --prompt "a different textual concept"
```

产物：每个变体一张 PNG、一张横向 `comparison.png`、以及 `condition_report.json`
（含各变体相对 `full` 的 `latent_l2_from_full` / `latent_cosine_distance_from_full` /
`pixel_l1_from_full`）。

**解读边界（脚本 `qualification` 字段也写明）**：非零差异只证明"条件被消费"，
**不**证明语义正确。因此：

- 只想看**敏感性**（条件确实生效）：任意 checkpoint，哪怕几十步 overfit 都行。
- 想看**语义正确**（不同 prompt/task 给出各自正确结构）：需要正式或充分 overfit
  的 checkpoint。

## 5. 消融建议

- **task token 容量**：`num_task_tokens` 取 1 / 4 / 8 做参数量与效果消融。设为 1 即
  "严格单 task token"，接口不变。
- **零信号对照**：把 task token 全部置零（`use_task_condition=False`）联合训练，
  多任务性能应明显掉点；否则说明模型没在用任务信号。
- **固定图像+噪声、仅换 task token**：输出应变化，验证任务条件真正驱动生成。
- **adapter 学习**：监控 `residual_scale` 与 adapter 梯度；长期为 0 说明任务特化没长出来。


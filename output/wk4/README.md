# W4 交付：任务 / 文本条件注入（Task / Text Conditioner）

本文件是 W4 的交付说明（自足）。模块代码、开关接口、单元测试均在仓库内（见下方落点）；
可视化 PNG、`condition_report.json`、逐变体结论 md 由脚本在**服务器**上生成，落在本目录的
`sensitivity-<task>/` 子目录下，按 **D005 不随代码入库**。

## 交付物对照

| W4 交付项 | 落点 | 状态 |
| --- | --- | --- |
| ① Task/Text Conditioner 模块代码（多尺度注入） | `perception_diffusion/models/conditioning.py`（`TaskTokenConditioner`）、`models/adapters.py`（`TaskAdapterBank`）；多尺度注入见 §2 | ✅ |
| ② 不同 prompt/task 输出变化的可视化与量化 | `scripts/validate_conditions.py` → 逐变体 PNG + `comparison.png` + `condition_report.json`；见 §4.2、§4.3 | ✅（敏感性） |
| ③ 条件开关接口（可开/关文本或任务条件，供消融） | `use_task_condition` / `use_text_condition` + `condition_mode_switches` 四模式；见 §3 | ✅ |
| ④ 模块单元测试与说明文档 | `tests/test_models.py::ConditioningTest`、`tests/test_training_and_inference.py::ConditionSwitchTest`；文档即本文件（详版见同目录 `conditioning.md`，属运行产物不入库）；见 §4.1 | ✅ |

> ② 的**语义正确性**（不同 task/prompt 给出各自*正确*的结构）需训练后 checkpoint 才能判定，
> 见 §4.3「解读边界」。当前预训练权重下仅验证到**敏感性**——条件确实被共享 U-Net 消费。

## 1. 模块架构

单一共享 U-Net 靠 `encoder_hidden_states` 区分任务。该条件张量由 `TaskTokenConditioner` 构建：

```
task 名   → task_embeddings(nn.Embedding, std=0.02) → TaskAdapterBank(残差) → task_tokens ┐
prompt    → [冻结 CLIP text_encoder] → text_projection(dims 相等时为 Identity) → text_tokens ┤
                                        torch.cat([task_tokens, text_tokens], dim=1) → encoder_hidden_states
                                                                                              │
image → [冻结 VAE encoder] → image latent ──→ 共享 U-Net（每个 down/mid/up 的 attn2 都消费）←┘
```

- 冻结：VAE、CLIP 文本编码器。可训：U-Net cross-attention + 任务 token + 任务 adapter。
- `num_task_tokens=4`，任务 token 为 `nn.Embedding` std=0.02 初始化；任务 adapter
  `residual_scale_init=0.0`（启动即恒等）。
- `text_input_dim == cross_attention_dim == 1024` 时 `text_projection` 退化为 `nn.Identity`。

## 2. 多尺度注入（v1）

`torch.cat([task_tokens, text_tokens])` 得到**同一条** `encoder_hidden_states`，被 U-Net
**每一个** down / mid / up 阶段的交叉注意力（`attn2`）在**各自空间分辨率**上消费——这即本期的
「多尺度注入」：同一条件在多个尺度被反复读取。

> 真正的「逐尺度独立条件」（每个 scale 一套 token / 投影）是 v2（Option C），本期未实现，作为
> 后续扩展点记录。

## 3. 条件开关接口（供消融）

两个布尔开关按段置零对应 token（**形状不变**，便于消融对照）：

- `use_task_condition=False` → 任务 token 段置零
- `use_text_condition=False` → 文本 token 段置零

`condition_mode_switches(mode)` 把四种模式映射为 `(use_task, use_text)`：

| mode | use_task | use_text | 含义 |
| --- | --- | --- | --- |
| `full` | True | True | 任务 + 文本 |
| `task_only` | True | False | 仅任务 token |
| `text_only` | False | True | 仅文本 |
| `unconditional` | False | False | 两者全关 |

推理/验证脚本均走此接口，训练侧可用于「零信号对照」消融（见 §5）。

## 4. 正确性验证

### 4.1 单元测试（无需训练，本地 CPU 即可）

- `tests/test_models.py::ConditioningTest`
  - `test_task_and_text_tokens_are_concatenated`：拼接顺序/形状。
  - `test_condition_switches_keep_shape_and_zero_selected_tokens`：开关按段置零、形状不变。
  - `test_disabling_only_text_keeps_task_tokens`：只关文本时任务 token 保留。
- `tests/test_training_and_inference.py::ConditionSwitchTest`
  - `test_condition_mode_switches_maps_each_mode`：四模式映射正确。
  - `test_disabling_task_condition_changes_sampled_output`：关任务条件改变采样输出。
  - `test_unconditional_sampling_is_reproducible`：无条件采样可复现。

运行：`python -m unittest tests.test_models tests.test_training_and_inference`

### 4.2 敏感性验证脚本（无需 checkpoint，可直接用预训练权重）

`scripts/validate_conditions.py`：固定图像 + 固定初始噪声，仅切换条件（四模式 / 换 prompt /
换 task），采样后解码，输出逐变体 PNG、`comparison.png`（横向拼接）与 `condition_report.json`
（逐变体 `latent_l2_from_full` / `cosine_distance` / `pixel_l1`）。`scripts/summarize_condition_report.py`
把该 json 汇总为 `w4_condition_conclusion.md`（逐变体数值 + 判定）。复现命令见 §5。

### 4.3 验证结论（如实记录）

在**冻结 CLIP + 未训练任务 token/adapter** 的预训练权重上跑 §4.2 脚本（`checkpoint_loaded: false`），
观测到：

- **文本条件：活的**。关文本（`mode-task_only`）与更换 prompt（`prompt-*`）都使采样 latent 产生
  **明显非零**变化，量级比仅换任务的路径高约 **2~3 个数量级**——文本 token 确实被共享 U-Net 的
  cross-attention 消费。
- **开关接口：有效**。`mode-unconditional`（两者全关）与 `mode-task_only` 同量级，
  `use_task_condition` / `use_text_condition` 按预期整段置零对应 token 且形状不变。
- **任务 token：训练前测不出区分度（符合预期，非缺陷）**。仅更换任务（`task-*`）时 latent 变化与
  数值噪声同量级。原因有三：任务 token 是 `std=0.02` 初始化的 `nn.Embedding`，量级被 O(1~10) 的
  冻结 CLIP 文本 token 淹没约 2~3 个数量级；任务 adapter `residual_scale_init=0.0`，启动即恒等；
  训练策略是冻结 VAE / 文本、只训 U-Net cross-attention 与任务 token / adapter。**因此「任务 token
  区分任务」的能力必须在训练之后才体现。**

> **解读边界**：非零差异只证明「条件被消费」（敏感性），不证明语义正确。要判断不同 task/prompt
> 是否给出各自*正确*的结构，需用 `--checkpoint` 加载正式或充分 overfit 的权重重跑。

**逐变体精确数值不写死在本文档**：由 `summarize_condition_report.py` 从当次运行的
`condition_report.json` 生成到 `sensitivity-<task>/w4_condition_conclusion.md`；换权重 / 换样本重跑即
刷新，避免 run-specific 数字过期。该结论及 PNG / JSON 属运行产物，按 D005 不入库。

## 5. 复现与产物落点（服务器执行）

> 以下命令由**你**在服务器上跑（D007：编码侧不连服务器）。产物写入本目录 `sensitivity-<task>/`。

```bash
# 1) 敏感性验证（无需 checkpoint，直接用预训练权重）
python scripts/validate_conditions.py \
  --config configs/multitask/stage1_shared_unet.yaml \
  --task depth \
  --num-steps 4 \
  --output-dir output/wk4/sensitivity-depth

# 2) 汇总为结论文档（逐变体数值 + 结论）
python scripts/summarize_condition_report.py \
  --report output/wk4/sensitivity-depth/condition_report.json \
  --output-dir output/wk4/sensitivity-depth
```

训练（或短 overfit）出 checkpoint 后，加 `--checkpoint <RUN>/checkpoints/step-XXXX.pt` 重跑，可看到
换任务时 latent 显著非零（任务 token 作用显现）。

产物落点（服务器生成，**不入库**）：

```
output/wk4/
├── README.md                         # 本交付说明（入库）
└── sensitivity-depth/                # 脚本生成，D005 不入库
    ├── mode-full.png / mode-task_only.png / mode-text_only.png / mode-unconditional.png
    ├── prompt-1.png / prompt-2.png
    ├── task-segmentation.png / task-normal.png
    ├── comparison.png
    ├── condition_report.json
    └── w4_condition_conclusion.md
```

## 6. 消融建议

- **task token 容量**：`num_task_tokens` 取 1 / 4 / 8 做参数量与效果消融（=1 即严格单 token，接口不变）。
- **零信号对照**：`use_task_condition=False` 联合训练，多任务性能应明显掉点；否则说明没在用任务信号。
- **固定图像+噪声、仅换 task token**：训练后输出应变化，验证任务条件真正驱动生成。
- **adapter 学习**：监控 `residual_scale` 与 adapter 梯度；长期为 0 说明任务特化没长出来。

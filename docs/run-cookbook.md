# 三任务真实运行 Cookbook

本手册由用户在研究服务器执行。工作目录固定为 `/home/jinyankai/gen-perception`；会读取 `/home/jinyankai/data` 和 `/home/jinyankai/models`，并向 `/home/jinyankai/outputs` 写入日志、checkpoint、预测和可视化。不要在其他用户占用 GPU 时运行，不要删除失败目录。

## 0. 环境与代码预检

```bash
cd /home/jinyankai/gen-perception
source /home/jinyankai/miniconda3/bin/activate gen-perception

export DATA_ROOT=/home/jinyankai/data
export MODEL_CACHE=/home/jinyankai/models
export OUTPUT_ROOT=/home/jinyankai/outputs

python evals/smoke_eval.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_configs.py
python scripts/train.py --config configs/smoke.yaml --dry-run
```

预期：全部命令退出码为 0；dry-run 报告 `annealed_multi_scale_noise: true` 和 `formal_training_started: false`。失败时停止，不启动 GPU 训练，并返回完整输出。

## 1. 数据 gate

资源：CPU、磁盘读取；NYUv2 预处理会持续写入 `/home/jinyankai/data/nyuv2/processed`。

```bash
cd /home/jinyankai/gen-perception
python scripts/data/preprocess_nyuv2.py \
  --root "$DATA_ROOT/nyuv2" \
  --split all

python scripts/data/smoke_test_datasets.py \
  --task all \
  --data-root "$DATA_ROOT" \
  --nyuv2-source auto \
  --num-workers 8 \
  --batch-size 2
```

预期：ADE20K、NYUv2 depth、derived normal 均报告合法 shape/range/valid fraction。出现缺文件、零有效像素、法线非单位向量或 worker 异常时停止。

## 2. 三任务 VAE 重建与保真度

资源：单 GPU，主要为 VAE 前向；输出目录必须不存在或为空。

```bash
cd /home/jinyankai/gen-perception
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_vae_reconstruction.py \
  --config configs/multitask/stage1_shared_unet.yaml \
  --task all \
  --device cuda \
  --precision fp16 \
  --limit 16 \
  --output-dir "$OUTPUT_ROOT/diagnostics/vae-reconstruction-v1"
```

预期：`VAE_RECONSTRUCTION_ANALYSIS_COMPLETED`，并生成 `report.json`、`report.csv`、`records.json`、`fidelity.md` 和三任务可视化。若出现非有限 latent/重建、指标缺失或显存错误，停止并返回报告与 traceback。

## 3. 单任务小样本过拟合

每次只使用一张空闲 GPU。三个命令分别写入独立目录；先运行分割，确认 loss 和 checkpoint 正常后再运行深度、法线。

```bash
cd /home/jinyankai/gen-perception
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/overfit/segmentation.yaml \
  --device cuda --precision fp16 \
  --experiment-name overfit-seg-v1

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/overfit/depth.yaml \
  --device cuda --precision fp16 \
  --experiment-name overfit-depth-v1

CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/overfit/normal.yaml \
  --device cuda --precision fp16 \
  --experiment-name overfit-normal-v1
```

预期：每项输出 `TRAINING_COMPLETED`、200 steps 和 `step-00000200.pt`；`metrics.jsonl` 中 loss 有总体下降趋势且无 NaN/Inf。分割 overfit 使用数据集词表中预先固定的 class 0，选择过程不读取当前样本 GT；先从 VAE 报告的 `foreground_fraction` 确认这 8 个样本不是全空掩码。真实推理和测评仍覆盖完整 150 类词表。若第一步失败、梯度为零、loss 连续非有限或显存不足，停止该任务并保留目录。

查看本地 TensorBoard（服务器只监听回环地址）：

```bash
tensorboard \
  --logdir "$OUTPUT_ROOT" \
  --host 127.0.0.1 \
  --port 6006
```

需要 W&B 时先执行 `pip install -r requirements-wandb.txt`，保持配置 `mode: offline`，训练命令追加 `--wandb`；离线目录后续由用户自行同步。

## 4. Checkpoint 恢复 gate

以下命令从分割 step 200 恢复并扩展到 220 steps，不覆盖旧 checkpoint：

```bash
cd /home/jinyankai/gen-perception
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/overfit/segmentation.yaml \
  --device cuda --precision fp16 \
  --max-steps 220 \
  --resume "$OUTPUT_ROOT/segmentation/overfit-seg-v1/checkpoints/step-00000200.pt"
```

预期：同一 run 目录生成 `step-00000220.pt`，日志 step 单调增加。若 checkpoint/config 不匹配或恢复后第一步非有限，停止并保留原文件。

## 5. 真实推理、可视化与评测

深度示例：

```bash
cd /home/jinyankai/gen-perception
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --config configs/overfit/depth.yaml \
  --checkpoint "$OUTPUT_ROOT/depth/overfit-depth-v1/checkpoints/step-00000200.pt" \
  --task depth \
  --device cuda --precision fp16 \
  --num-steps 10 --limit 8 \
  --output-dir "$OUTPUT_ROOT/diagnostics/infer-depth-v1"

python scripts/evaluate.py \
  --config configs/overfit/depth.yaml \
  --predictions "$OUTPUT_ROOT/diagnostics/infer-depth-v1/predictions" \
  --targets "$OUTPUT_ROOT/diagnostics/infer-depth-v1/targets" \
  --valid-masks "$OUTPUT_ROOT/diagnostics/infer-depth-v1/valid_masks" \
  --output-dir "$OUTPUT_ROOT/diagnostics/eval-depth-v1"
```

法线将 config/checkpoint/task/output 名替换为 `normal`。分割推理会对 ADE20K 全部 150 类执行封闭集查询，计算量约为 `样本数 × 150 × denoise steps`，首次只使用 `--limit 1 --num-steps 2`：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer.py \
  --config configs/overfit/segmentation.yaml \
  --checkpoint "$OUTPUT_ROOT/segmentation/overfit-seg-v1/checkpoints/step-00000200.pt" \
  --task segmentation \
  --device cuda --precision fp16 \
  --num-steps 2 --limit 1 \
  --output-dir "$OUTPUT_ROOT/diagnostics/infer-seg-v1"
```

预期：每个推理目录包含严格配对的 `predictions/`、`targets/`、`valid_masks/`、`visualizations/` 和 `inference.json`。这些 overfit 指标只验证 pipeline，不得登记为正式 benchmark。

## 6. Task/Text 条件正确性

先训练一个极小多任务 checkpoint：

```bash
cd /home/jinyankai/gen-perception
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
  --config configs/overfit/multitask.yaml \
  --device cuda --precision fp16 \
  --experiment-name overfit-multitask-v1
```

固定样本、初始噪声和 seed，对比 full/task-only/text-only/unconditional、替换 prompt 和替换 task token：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/validate_conditions.py \
  --config configs/overfit/multitask.yaml \
  --checkpoint "$OUTPUT_ROOT/multitask/overfit-multitask-v1/checkpoints/step-00000300.pt" \
  --task depth \
  --device cuda --precision fp16 \
  --num-steps 2 \
  --output-dir "$OUTPUT_ROOT/diagnostics/conditions-v1"
```

预期：生成 `comparison.png`、每种条件的 PNG 和 `condition_report.json`。非零差异只证明模型对条件敏感；必须结合过拟合后的语义结果判断条件是否正确。

## 7. 返回证据

请返回以下文件或完整文本，不要只返回“成功”：

1. 各命令终端输出和退出码；
2. `metrics.json`、loss 对应的 `metrics.jsonl` 片段；
3. VAE `report.json` 与 `fidelity.md`；
4. 条件 `condition_report.json` 和 `comparison.png`；
5. 每任务至少一张推理 panel；
6. `git rev-parse HEAD`、GPU 型号、运行时长和峰值显存。

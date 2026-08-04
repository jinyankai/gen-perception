# Baseline RUNBOOK — 三任务判别式对照

面向服务器操作者的**可复制粘贴**命令链。编码 agent 不在服务器执行任何命令（D007）；
你在服务器逐段运行并回传日志。所有权重/数据/输出只落 `/home/jinyankai/{models,data,outputs}`，
**不进 Git**（D005）。

## 0. 约定

```bash
export REPO=/home/jinyankai/gen-perception          # 本仓库 checkout
export MODELS=/home/jinyankai/models
export DATA=/home/jinyankai/data
export OUT=/home/jinyankai/outputs/baselines
export HF_ENDPOINT=https://hf-mirror.com            # D006：显式镜像端点
cd "$REPO"
```

数据流：`export_targets.py` 写 GT + `image_manifest.jsonl` → `run_*.py` 读 manifest 写预测 `.npy`
→ `scripts/evaluate.py` 按相对路径 stem 配对预测/GT 出指标。**每步先 `--limit 5` 冒烟，再全量。**

## 1. 建环境（每个 baseline 独立，ABI 冲突不可共享）

```bash
# 逐个创建；env yaml 头部有等价 pip 命令
conda env create -f scripts/baselines/envs/marigold.yaml
conda env create -f scripts/baselines/envs/mask2former.yaml
conda env create -f scripts/baselines/envs/depthanything.yaml   # 可选
conda env create -f scripts/baselines/envs/dsine.yaml           # 可选
```

导出 GT 只需 gen-perception 主环境（有 h5py/scipy/PIL）。

## 2. 下载权重（复用 scripts/hf_download.py，解析不可变 SHA + 写 manifest，D006）

```bash
# 深度（主）
HF_ENDPOINT=https://hf-mirror.com python scripts/hf_download.py \
  --repo-id prs-eth/marigold-depth-v1-1 --repo-type model \
  --local-dir "$MODELS/marigold-depth-v1-1"
# 法线（主）
HF_ENDPOINT=https://hf-mirror.com python scripts/hf_download.py \
  --repo-id prs-eth/marigold-normals-v1-1 --repo-type model \
  --local-dir "$MODELS/marigold-normals-v1-1"
# 分割（主）
HF_ENDPOINT=https://hf-mirror.com python scripts/hf_download.py \
  --repo-id facebook/mask2former-swin-small-ade-semantic --repo-type model \
  --local-dir "$MODELS/mask2former-swin-small-ade-semantic"
# 深度（可选对照）
HF_ENDPOINT=https://hf-mirror.com python scripts/hf_download.py \
  --repo-id depth-anything/Depth-Anything-V2-Base-hf --repo-type model \
  --local-dir "$MODELS/depth-anything-v2-base"
```

DSINE 无稳定 HF repo：`git clone` 官方仓库到 `$MODELS/dsine-repo` 并 `git checkout <PINNED_COMMIT>`，
权重手动放入 `$MODELS/dsine`，再用 `scripts/hf_download.py --register-existing` 记录 SHA manifest。
运行前必须核对 DSINE 的 torch.hub 入口名与预处理是否与 pinned commit 一致（见脚本 docstring）。

预期产物：每个 `--local-dir` 下含权重文件 + `download_manifest.json`（含 resolved SHA）。
失败回退：镜像超时→重试或换 `--revision <sha>`；gated 许可→先在 hf-mirror 页面接受许可。

## 3. 导出 GT + image manifest（gen-perception 主环境）

```bash
# 分割：ADE20K validation → targets/(0..149,255=ignore) + images 引用源 JPEG
python scripts/baselines/export_targets.py --task segmentation \
  --config configs/segmentation/ade20k.yaml --output-dir "$OUT/seg/gt" --limit 5
# 深度：NYUv2 test → targets/(metric metres) + valid_masks/ + images/(解码 PNG)
python scripts/baselines/export_targets.py --task depth \
  --config configs/depth/nyuv2.yaml --output-dir "$OUT/depth/gt" --limit 5
# 法线：NYUv2 test → targets/(3xHxW unit) + valid_masks/ + images/
python scripts/baselines/export_targets.py --task normal \
  --config configs/normal/nyuv2.yaml --output-dir "$OUT/normal/gt" --limit 5
```

冒烟无误后去掉 `--limit` 跑全量（seg 2000 张，depth/normal 各 654 张）。
预期产物：`$OUT/<task>/gt/{targets,valid_masks,images,image_manifest.jsonl}`。
**分割 GT 已映射到 0..149，故 evaluate 无需 `--target-label-offset`。**

## 4. 推理（各自 conda 环境；GPU 空闲后再跑，勿动占用 8 卡的进程）

```bash
# 深度主 baseline（conda activate baseline-marigold）
python scripts/baselines/run_marigold_depth.py \
  --image-manifest "$OUT/depth/gt/image_manifest.jsonl" \
  --weights-dir "$MODELS/marigold-depth-v1-1" \
  --output-dir "$OUT/depth/pred_marigold" --limit 5
# 法线主 baseline（同环境）
python scripts/baselines/run_marigold_normal.py \
  --image-manifest "$OUT/normal/gt/image_manifest.jsonl" \
  --weights-dir "$MODELS/marigold-normals-v1-1" \
  --output-dir "$OUT/normal/pred_marigold" --limit 5
# 分割主 baseline（conda activate baseline-mask2former）
python scripts/baselines/run_mask2former_seg.py \
  --image-manifest "$OUT/seg/gt/image_manifest.jsonl" \
  --weights-dir "$MODELS/mask2former-swin-small-ade-semantic" \
  --output-dir "$OUT/seg/pred_mask2former" --limit 5

# 可选对照
python scripts/baselines/run_depthanything_depth.py \
  --image-manifest "$OUT/depth/gt/image_manifest.jsonl" \
  --weights-dir "$MODELS/depth-anything-v2-base" \
  --output-dir "$OUT/depth/pred_depthanything" --invert --limit 5   # --invert 必须！视差→深度
python scripts/baselines/run_dsine_normal.py \
  --image-manifest "$OUT/normal/gt/image_manifest.jsonl" \
  --weights-dir "$MODELS/dsine" --dsine-repo "$MODELS/dsine-repo" \
  --output-dir "$OUT/normal/pred_dsine" --limit 5
```

冒烟通过后去掉 `--limit`。预期产物：`pred_*/<sample_id>.npy` + `runtime.json`（GPU/显存/时延）。

## 5. 评测（gen-perception 主环境，统一 evaluator）

```bash
# 分割 mIoU（dataset-level 混淆矩阵；无需 label offset）
python scripts/evaluate.py --config configs/segmentation/ade20k.yaml --task segmentation \
  --predictions "$OUT/seg/pred_mask2former" --targets "$OUT/seg/gt/targets" \
  --output-dir "$OUT/seg/metrics_mask2former" --limit 5
# 深度 AbsRel/delta1/RMSE（逐样本 affine 对齐，min/max 由 config）
python scripts/evaluate.py --config configs/depth/nyuv2.yaml --task depth \
  --predictions "$OUT/depth/pred_marigold" --targets "$OUT/depth/gt/targets" \
  --valid-masks "$OUT/depth/gt/valid_masks" \
  --output-dir "$OUT/depth/metrics_marigold" --limit 5
# 法线角误差（unit 编码；内部 re-normalize）
python scripts/evaluate.py --config configs/normal/nyuv2.yaml --task normal \
  --predictions "$OUT/normal/pred_marigold" --targets "$OUT/normal/gt/targets" \
  --valid-masks "$OUT/normal/gt/valid_masks" \
  --prediction-normal-encoding unit --target-normal-encoding unit \
  --output-dir "$OUT/normal/metrics_marigold" --limit 5
```

冒烟通过后去掉 `--limit` 出正式指标。预期产物：`metrics/{metrics.json,metrics.csv}`。

## 6. 合理性区间（偏离则排查配对/编码/--invert）

| 指标 | 参考量级（NYUv2/ADE20K） |
| --- | --- |
| Marigold depth AbsRel | ~0.05–0.07（affine 对齐后） |
| Marigold depth delta1 | ~0.94–0.96 |
| Mask2Former (Swin-S) mIoU | ~0.47–0.50 |
| Marigold normals mean angular err | ~15°–20° |

## 7. 回传

把每个 `metrics.json` + `runtime.json` 回传，用于登记到 EXPERIMENTS.md（状态 COMPLETED/FAILED）。
指标只作 baseline 对照，非生成式框架复现结果。

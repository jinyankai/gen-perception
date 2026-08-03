#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "AGENTS.md",
    "docs/agent-harness/index.md",
    "docs/agent-harness/quality.md",
    "docs/agent-harness/tools.md",
    "docs/agent-harness/server-operator-contract.md",
    "docs/agent-harness/review.md",
    "agents/skills/README.md",
    "STATUS.md",
    "BLOCKERS.md",
    "DECISIONS.md",
    "EXPERIMENTS.md",
    "configs/smoke.yaml",
    "perception_diffusion/codecs/segmentation.py",
    "perception_diffusion/codecs/depth.py",
    "perception_diffusion/codecs/normal.py",
    "perception_diffusion/evaluation/segmentation.py",
    "perception_diffusion/evaluation/depth.py",
    "perception_diffusion/evaluation/normal.py",
    "scripts/train.py",
    "scripts/validate_configs.py",
    "scripts/hf_mirror_env.sh",
    "scripts/hf_download.py",
    "scripts/operator/sync_server_repo.sh",
    "scripts/operator/probe_hf_mirror.sh",
    "docs/huggingface-mirror.md",
    "docs/unified-perception-framework.md",
    "configs/multitask/stage1_shared_unet.yaml",
    "perception_diffusion/tasks.py",
    "perception_diffusion/models/conditioning.py",
    "perception_diffusion/models/adapters.py",
    "perception_diffusion/models/unet.py",
    "perception_diffusion/models/unified_denoiser.py",
    "perception_diffusion/models/builder.py",
    "perception_diffusion/models/target_adapters.py",
    "perception_diffusion/task_specs.py",
    "perception_diffusion/data/segmentation_vocabulary.py",
    "perception_diffusion/inference/segmentation_queries.py",
    "perception_diffusion/inference/latent_sampler.py",
    "perception_diffusion/training/unified_trainer.py",
    "scripts/framework_smoke.py",
]


def main() -> int:
    missing = [p for p in REQUIRED if not (ROOT / p).exists()]
    if missing:
        print("Missing harness files:")
        for path in missing:
            print(f"- {path}")
        return 1
    print("Harness smoke eval passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

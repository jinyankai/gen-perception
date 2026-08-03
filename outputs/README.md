# Experiment outputs

Generated experiment files are ignored by Git. Each experiment must have:

```text
outputs/<task>/<experiment_name>/
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

Do not delete failed experiment directories. Record their status in `EXPERIMENTS.md`.

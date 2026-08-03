# Stage-One 3-4 Week Plan

## Outcome

Deliver one configuration-driven latent-diffusion framework with shared training, inference, and evaluation entry points for segmentation, depth, and surface normals. Each task must have a tested codec, dataset adapter, evaluator, visualization, formal configuration, checkpoint/log trail, and a reasonable comparison baseline.

## Gate sequence

No task may start long or multi-GPU training until it passes, in order:

1. single-sample codec round-trip;
2. one-batch forward;
3. one-batch backward with finite gradients;
4. 10-step loss check;
5. 100-500 step small-sample overfit;
6. checkpoint save/load/resume;
7. inference decode;
8. metric test against known answers;
9. task visualization;
10. provenance capture.

## Milestones

| Window | Milestone | Evidence |
| --- | --- | --- |
| D0-D3 | server, repo, environment, protocol base | `STATUS.md`, environment files, checks |
| D2-D7 | three codecs and evaluators | tests, round-trip metrics, reconstruction report |
| D5-D11 | VAE, conditioner, denoiser, scheduler | component load test and shape checks |
| D9-D14 | three small overfit loops | logs, checkpoints, inference, visualizations |
| D13-D22 | formal runs and baselines | traceable per-task experiment directories |
| D19-D26 | minimum ablations | condition, noise scale, step/latency tables |
| D22-D28 | report and clean reproduction | `REPORT.md`, reproduce script, clean smoke test |

## Scope control

Detection, classification, optical flow, large hyperparameter sweeps, and unvalidated innovation modules are out of stage-one scope. If the schedule compresses, reduce baselines and ablations, not the shared framework or three-task closure.

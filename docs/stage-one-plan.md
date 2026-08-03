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

## Current gate status

| Gate | Local structural status | Real-component status |
| --- | --- | --- |
| codec round-trip and known-answer metrics | evaluator unit cases and the unified JSON/CSV CLI pass synthetic known answers | ADE20K/NYUv2 sample codec checks passed; provenance-bound real prediction evaluation and task-target VAE reconstruction pending |
| one-batch forward/backward | passed for all three tasks with an injected tiny U-Net/scheduler | real SD2 component load and CPU forward passed in S002; real backward and CUDA pending |
| 10-step loss and small-sample overfit | not run | pending |
| checkpoint/resume and decoded inference | not implemented end to end | pending |
| visualization and provenance capture | protocol documented | real-run artifacts pending |

Structural passes verify interfaces and gradient flow only; they do not advance the corresponding real-model gate.

## Milestones

| Window | Milestone | Evidence | Status |
| --- | --- | --- | --- |
| D0-D3 | server, repo, environment, protocol base | `STATUS.md`, environment files, checks | environment and asset roots confirmed; exact server commit binding still pending |
| D2-D7 | three codecs and evaluators | tests, round-trip metrics, reconstruction report | unit-level implementation and dataset sample checks complete; task-target VAE reconstruction pending |
| D5-D11 | VAE, conditioner, denoiser, scheduler | component load test and shape checks | real components and project-wrapped CPU forward validated; reusable training loader/backward pending |
| D9-D14 | three small overfit loops | logs, checkpoints, inference, visualizations | pending |
| D13-D22 | formal runs and baselines | traceable per-task experiment directories | pending |
| D19-D26 | minimum ablations | condition, noise scale, step/latency tables | configs prepared; runs pending |
| D22-D28 | report and clean reproduction | `REPORT.md`, reproduce script, clean smoke test | report scaffold and local smoke available; formal evidence pending |

## Scope control

Detection, classification, optical flow, large hyperparameter sweeps, and unvalidated innovation modules are out of stage-one scope. If the schedule compresses, reduce baselines and ablations, not the shared framework or three-task closure.

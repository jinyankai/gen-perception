# Stage-One Technical Report

Status: the three-task real training/inference implementation is code-ready, and user-operated CPU plus CUDA forward evidence is recorded. Real backward, overfit, decoded inference, and formal metrics have not been produced.

## Verified implementation snapshot

- A canonical three-task registry binds segmentation, depth, and surface-normal codecs/evaluators to explicit task names.
- Learned task tokens, task-specific condition adapters, and optional text tokens form the shared U-Net cross-attention condition.
- One 8-channel U-Net wrapper, mask-aware latent diffusion-loss core, and latent sampler are shared across all three tasks.
- ADE20K closed-set queries come from the dataset taxonomy rather than the current sample's ground truth.
- An optional bounded, zero-initialized pre-VAE residual CNN is isolated as an ablation; deterministic codecs remain responsible for task semantics and target ranges.
- A frozen Visual Latent Pathway connects real targets to SD2 VAE latents, while a Marigold-style timestep-annealed multi-resolution noise sampler feeds the shared epsilon-prediction loss.
- A single configuration-driven runner provides real DataLoaders, AMP optimization, checkpoint/resume, JSONL/TensorBoard/W&B logging, decoded inference, reconstruction fidelity, and condition-switch diagnostics for all three tasks.

## Evidence currently available

At commit `3ea54b8`, 33 unit tests passed, six configs validated, and the tiny torch-only framework smoke produced finite loss, gradients, and samples for all three tasks. The subsequent user-operated S002 smoke loaded local SD2 components offline on CPU and checked ADE20K/NYUv2 samples; S003 passed the real 8-channel SD2 forward and VAE decode on GPU 0. See `docs/validation/stage1-assets-2026-08-03.md` and `docs/validation/segmentation-cuda-forward-2026-08-03.md`.

For the current local working tree, 61 unit tests (including an injected-component optimizer/log/checkpoint runner integration), 10 configuration validations, the harness smoke, training dry-run, bytecode compilation, and the three-task structural forward/backward/sample smoke pass. This is code evidence only: it is not evidence that the new reusable runner has executed against real weights/data, that loss converges, or that decoded predictions are correct. The ordered server gates are in `docs/run-cookbook.md`.

## Report outline

1. Project background
2. Stage-one research questions
3. Literature review
4. Taxonomy of generative perception methods
5. Unified framework design
6. Visual Latent Pathway
7. Task / Text Conditioner
8. Consistent Denoising U-Net
9. Datasets and evaluation protocols
10. Experimental setup
11. Segmentation results
12. Depth results
13. Surface normal results
14. Initial generative versus discriminative comparison
15. Denoising steps, quality, and latency
16. Failure cases
17. Current limitations
18. Stage-one conclusions
19. Stage-two plan
20. Reproduction instructions

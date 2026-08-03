# Stage-One Technical Report

Status: framework design, local structural evidence, and a user-operated real-asset CPU integration smoke are recorded. No formal metrics have been produced.

## Verified implementation snapshot

- A canonical three-task registry binds segmentation, depth, and surface-normal codecs/evaluators to explicit task names.
- Learned task tokens, task-specific condition adapters, and optional text tokens form the shared U-Net cross-attention condition.
- One 8-channel U-Net wrapper, mask-aware latent diffusion-loss core, and latent sampler are shared across all three tasks.
- ADE20K closed-set queries come from the dataset taxonomy rather than the current sample's ground truth.
- An optional bounded, zero-initialized pre-VAE residual CNN is isolated as an ablation; deterministic codecs remain responsible for task semantics and target ranges.

## Evidence currently available

At commit `3ea54b8`, 33 unit tests passed, six configs validated, the repository smoke and configuration dry-run passed, and the tiny torch-only framework smoke produced finite loss, gradients, and samples for all three tasks. In the subsequent user-operated S002 server smoke, the local SD2 tokenizer/text encoder/VAE/U-Net loaded offline on CPU, ADE20K and NYUv2 samples passed structural/codec checks, and the project 8-channel denoiser produced a finite output. The detailed evidence is in `docs/validation/stage1-assets-2026-08-03.md`. This is not evidence of CUDA execution, training, decoded task predictions, formal metrics, latency/memory performance, or baseline comparisons.

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

# Blockers

## B001 - No writable high-capacity project storage

- Status: OPEN
- Evidence: system filesystem had about 39 GiB free; `/data` had capacity but its root was not writable.
- Impact: model cache, ADE20K/NYUv2, checkpoints, predictions, and formal outputs must not be placed under the system filesystem.
- Minimum resolution: allocate a writable directory such as `/data/jinyankai/gen-perception` or provide another approved high-capacity path.
- Non-blocked work: repository, unit tests, codecs, evaluators, configs, and CPU smoke tests.

## B002 - All GPUs occupied at inventory time

- Status: OPEN
- Evidence: every RTX 4090 had active compute processes and at least about 16 GiB memory in use.
- Impact: CUDA smoke tests and training cannot be run safely without interfering with other users.
- Minimum resolution: provide a GPU availability window or confirm which GPU IDs are allocated to this project.
- Non-blocked work: CPU implementation, configuration, tests, data protocol design, and documentation.

## B003 - Hugging Face unreachable from server

- Status: MITIGATION IMPLEMENTED, CONNECTIVITY UNVERIFIED
- Evidence: HTTPS probe to `huggingface.co` timed out; GitHub and PyPI succeeded.
- Impact: Stable Diffusion, Marigold, and many baseline checkpoints cannot download through the default Hub route.
- Current mitigation: standard `HF_ENDPOINT=https://hf-mirror.com` support plus an immutable-revision downloader and manifest.
- Minimum resolution: verify mirror connectivity from the server; otherwise use an approved proxy, pre-populated shared cache, or offline transfer of exact checkpoints and licenses.
- Non-blocked work: package installation from PyPI, source code, synthetic smoke tests, and evaluator implementation.

## B004 - Required datasets not located

- Status: OPEN
- Evidence: limited user-home inventory found no ADE20K/PASCAL Context or NYUv2 directories.
- Impact: dataset adapters can be implemented against documented layouts, but real-data smoke and formal metrics remain blocked.
- Minimum resolution: approved dataset paths or permission to download and prepare the datasets in B001 storage.

# Tool and MCP Constraints

## Allowed Tooling

- Prefer local repository files, tests, scripts, and official docs.
- Use MCP tools only when they are configured and relevant to the task.
- Do not assume a tool exists. Discover it from local config or the active Codex session.

## Safety

- Do not expose secrets, tokens, credentials, or private data.
- Do not connect to or operate the research server directly. Server commands are user-operated under `server-operator-contract.md`.
- For each requested server action, provide the exact working directory, reviewed command or repository script, expected effects, expected evidence, and any relevant recovery step.
- Treat user-returned logs as the only evidence that a server command completed; never infer completion from an interrupted or timed-out session.
- Ask for approval before destructive operations, external writes, production access, or broad network actions.
- Keep generated artifacts inside the repository unless the user asks for personal/global configuration.
- Do not call `sudo`; the server account has no sudo access.
- Do not stop or interfere with other users' GPU processes.
- Do not write datasets, model weights, or environments to the nearly full system filesystem once a high-capacity project path is available.
- Hugging Face is currently unreachable from the server. Use the documented `HF_ENDPOINT` workflow in `docs/huggingface-mirror.md` or an approved offline checkpoint transfer.

## Evidence

When using tools, record the important outputs in the final response.

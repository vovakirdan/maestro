Minimal multi-agent orchestration system (Python standard library only).

Commands:

    python3 main.py setup
    python3 main.py run

Workspace layout created by `setup`:

    <workspace>/
      orchestrator/
        pipeline.json
        packets/
          agent_1/
            ROLE.md
            TARGET.md
            RULES.md
            CONTEXT.md
            REPORT_FORMAT.md
            INPUTS.md
            NOTES.md
          agent_2/
          ...
        runs/
          <run_id>/
            state.json
            timeline.jsonl
            packets/        # run-local packet copies
            steps/          # per-step artifacts

Providers:
- `deterministic`: offline provider for predictable runs.
- `codex_cli`: shells out to a configurable Codex CLI command and parses JSONL from stdout (recommended: `codex exec --json --full-auto`).

Example (offline):

    python3 main.py run --workspace example/workspace

Checks / lint (stdlib-only):

    python3 tools/check.py

Optional (if installed):

    python3 tools/check.py --with-ruff
    python3 tools/check.py --with-black

Minimal multi-agent orchestration system (Python standard library only).

Commands:

    python3 main.py setup
    python3 main.py run

Timeouts:
- The provider has two timeouts: `timeout_s` (hard wall-clock) and `idle_timeout_s` (no stdout/stderr).
- Set either to `0` to disable it.
- For long-running Codex runs, prefer `timeout_s=0` (no hard limit) and keep a reasonable `idle_timeout_s`.

Setup presets:
- `crt` (coder -> reviewer -> tester): runs reviewer/tester validation and routes FAILED feedback back to coder.
  Stops and escalates after `max_returns` returns to the coder.
- `cr` (coder -> reviewer): runs reviewer validation and routes FAILED feedback back to coder.
  Stops and escalates after `max_returns` returns to the coder.

Setup also asks for a task type:
- `feature`: prioritize backwards-compatibility and regression protection.
- `bug`: prioritize minimal scope, repro, root cause, and regression coverage.
- `bootstrap`: prioritize explicit scope/acceptance criteria and a runnable baseline.

Before `run`, the CLI asks about planning:
- `auto`: generate a plan via the configured provider and inject it into INPUTS.md for all steps.
- `user`: use a user-provided plan (paste or file path).
- `none`: run with packets only.

Before `run`, the CLI can also apply a git safety policy (if the workspace is a git repo root):
- `branch`: require a clean working tree and checkout a new branch `orch/<run_id>` (configurable prefix).
- `check`: require a clean working tree but do not switch branches.
- `off`: no git checks.

When `Git safety=branch`, the CLI can also auto-commit after coder steps (optional). This makes review loops
more pragmatic by turning each coder output into a concrete commit for the reviewer to inspect.

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

Run overrides:

    # Disable hard timeout, keep idle timeout at 10 minutes.
    python3 main.py run --workspace /path/to/workspace --timeout-s 0 --idle-timeout-s 600

Checks / lint (stdlib-only):

    python3 tools/check.py

Optional (if installed):

    python3 tools/check.py --with-ruff
    python3 tools/check.py --with-black

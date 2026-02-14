Minimal multi-agent orchestration system (Python standard library only).

This repo provides:
- Packet-based, provider-agnostic actors (ROLE/TARGET/RULES/CONTEXT/REPORT_FORMAT/INPUTS/NOTES).
- Pluggable providers: `deterministic` (offline), `codex_cli`, `gemini_cli`, `claude_cli`.
- A sequential orchestrator with an optional linear "return-on-failure" loop.
- An interactive `setup` wizard (human prompts) plus an optional AI setup-wizard stage.
- Optional plan injection, git safety, auto-commit, and merge request instructions.

Commands:

    python3 main.py setup --workspace /path/to/workspace
    python3 main.py run --workspace /path/to/workspace
    python3 main.py run --workspace /path/to/workspace --task <task_id>

Workspace layout created by `setup`:

    <workspace>/
      orchestrator/
        CURRENT_TASK             # task id used by default in `run`
        tasks/
          <task_id>/
            TASK.json            # task metadata (goal, kind, timestamps)
            pipeline.json
            BRIEF.md             # only when AI setup wizard runs
            setup_artifacts/     # provider artifacts from setup wizard runs
            packets/
              <actor_id>/
                ROLE.md
                TARGET.md
                RULES.md
                CONTEXT.md
                REPORT_FORMAT.md
                INPUTS.md
                NOTES.md         # append-only; managed by orchestrator during runs
            runs/
              <run_id>/
                state.json
                timeline.jsonl
                final.txt
                final_report.json
                NEEDS_INPUT.md   # only when run stops with NEEDS_INPUT
                plan/            # auto/user plan artifacts (if enabled)
                escalation/      # escalation wizard output (if enabled/available)
                packets/         # run-local packet copies (inputs are modified here)
                steps/           # per-step artifacts (attempts, validation, commits)
                  <nn_actor_id>/
                    final.txt
                    report.json
                    commit.json  # only if auto-commit created a commit
                    attempt_1/
                    attempt_2/
                delivery/        # MR instructions (if enabled)

Legacy layout is still supported:
- <workspace>/orchestrator/pipeline.json
- <workspace>/orchestrator/packets/
- <workspace>/orchestrator/runs/

Setup flow:
- Task type (feature/bug/bootstrap/other).
- Workflow preset (crt/cr/c/r/t/d/custom).
- Optional: configure how the orchestrator reacts to reviewer findings (status-only vs tag/threshold based return vs escalation).
- Provider selection:
  - The CLI only offers providers that are installed (found on PATH).
  - You can use one provider for everything (wizard + all agents), or split it:
    - choose a wizard provider
    - choose one provider for all agents, or per-agent providers (overrides)
- Optional: "setup wizard (AI)" (any non-deterministic provider):
  - `wizard:analyze` writes `orchestrator/BRIEF.md` (high-signal decomposition).
  - `wizard:packet:<actor_id>` rewrites packet docs to be detailed and executable.
  - Packet docs are written for isolated agents (no orchestration/other-agent mentions).
  - Setup artifacts are saved under `orchestrator/setup_artifacts/` for debugging.
- If task type is `bootstrap` and the workspace is not a git repo, setup can optionally run `git init`.
- Each `setup` creates a new task folder under `orchestrator/tasks/<task_id>/` and updates `orchestrator/CURRENT_TASK`.

Workflow presets:
- `crt`: coder -> reviewer -> tester. If reviewer/tester returns `FAILED`, the run routes back to coder.
  Escalates to NEEDS_INPUT after `max_returns`.
- `cr`: coder -> reviewer. If reviewer returns `FAILED`, the run routes back to coder.
  Escalates to NEEDS_INPUT after `max_returns`.
- `c`: coder only.
- `r`: reviewer only.
- `t`: tester only.
- `d`: devops only.
- `custom`: 1-6 stages, choose templates (`coder/reviewer/tester/devops/custom`) and optionally configure
  a linear return-on-failure loop.
- `other` (custom stage template): requires a role description (multiline) so the setup wizard can produce good packets.
  Use this when the built-ins do not fit but you still want structured setup.

Run flow:
- Plan mode (auto/user/none):
  - `auto`: generate a plan via the first non-deterministic provider in the pipeline and inject it into INPUTS.md.
  - `user`: use a user plan (paste or file path).
  - `none`: no plan injection.
- Git safety (branch/check/off), if the workspace is a git repo root:
  - `branch`: require clean tree (ignores changes under `orchestrator/`), create a run branch, optional auto-commit.
  - `check`: require clean tree (ignores changes under `orchestrator/`), do not switch branches.
  - `off`: no git checks.
- Each step:
  - Runs the actor, validates REPORT_FORMAT.md, and retries once if the output format is invalid.
    "attempt 2" means a report-format retry for the same step.
  - Writes upstream handoff into the next step's `INPUTS.md` (in the run-local packet copy).
  - If `git safety=branch` and auto-commit is enabled, commits workspace changes (excluding `orchestrator/`).
- If the run stops with `NEEDS_INPUT`:
  - The run writes `NEEDS_INPUT.md` for the user.
  - If a non-deterministic provider is available, an escalation wizard writes `runs/<run_id>/escalation/escalation.md`
    with suggested resolutions and subtasks.

Progress output:
- When stderr is a TTY, `run` shows a single-line spinner status on stderr.
- Otherwise (IDE output panels, logs), it prints periodic `progress:` lines instead.

Delivery (merge request instructions):
- If `setup` enabled "Merge request at end: instructions" AND the run used `git safety=branch`,
  the run writes:
  - `orchestrator/runs/<run_id>/delivery/mr_instructions.md`
  - `orchestrator/runs/<run_id>/delivery/mr_body.md`

Timeouts:
- Providers use `timeout_s` (hard wall-clock) and `idle_timeout_s` (no stdout/stderr activity).
- Set either to `0` to disable it.
- For long Codex runs, prefer `--timeout-s 0` and keep a reasonable `--idle-timeout-s`.

Providers:
- `deterministic`: offline provider for predictable runs.
- `codex_cli`: shells out to a configurable Codex CLI command and parses JSONL from stdout.
  Recommended command: `codex exec --json --full-auto`.
- `gemini_cli`: shells out to `gemini` (Gemini CLI). Recommended command:
  `gemini --output-format text --approval-mode yolo -p " "`.
- `claude_cli`: shells out to `claude` (Claude Code CLI). Recommended command:
  `claude -p --output-format text --input-format text --permission-mode acceptEdits`.

Example (offline):

    python3 main.py run --workspace example/workspace

Run overrides:

    # Disable hard timeout, keep idle timeout at 10 minutes.
    python3 main.py run --workspace /path/to/workspace --timeout-s 0 --idle-timeout-s 600

Checks:

    python3 tools/check.py
    python3 tools/check.py --with-ruff
    python3 tools/check.py --with-black

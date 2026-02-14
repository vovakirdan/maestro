from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.packet import write_packet_documents
from src.core.runtime import (
    ActorConfig,
    OrchestrationConfig,
    PipelineConfig,
    ProviderConfig,
    TaskConfig,
    TaskKind,
    orchestrator_root,
    save_pipeline,
)
from src.core.types import JsonDict
from src.orchestrator.templates import get_template, list_template_ids
from src.providers.base import Provider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider


def _prompt_line(label: str, *, default: str | None = None) -> str:
    if default is None:
        suffix = ""
    else:
        suffix = f" [{default}]"
    while True:
        v = input(f"{label}{suffix}: ").strip()
        if v:
            return v
        if default is not None:
            return default


def _prompt_optional_line(label: str) -> str:
    return input(f"{label}: ").strip()


def _prompt_int(label: str, *, min_value: int, max_value: int, default: int) -> int:
    while True:
        raw = _prompt_line(label, default=str(default))
        try:
            n = int(raw)
        except ValueError:
            print(f"Please enter an integer between {min_value} and {max_value}.")
            continue
        if min_value <= n <= max_value:
            return n
        print(f"Please enter an integer between {min_value} and {max_value}.")


def _prompt_yes_no(label: str, *, default: bool) -> bool:
    d = "y" if default else "n"
    while True:
        raw = _prompt_line(label + " (y/n)", default=d).lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _prompt_choice(label: str, *, choices: list[str], default: str) -> str:
    choice_map = {c.lower(): c for c in choices}
    while True:
        raw = _prompt_line(label, default=default).strip().lower()
        if raw in choice_map:
            return choice_map[raw]
        print(f"Please choose one of: {', '.join(choices)}")


def _prompt_multiline_optional(label: str) -> str:
    print(f"{label} (optional). End input with a single '.' line:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


class _SetupRefineUI:
    def __init__(self, *, label: str) -> None:
        self._label = label
        self._frames = ["|", "/", "-", "\\"]
        self._frame_i = 0
        self._status_len = 0
        self._started = time.monotonic()
        self._last_summary = ""
        self._last_line = 0.0

    def _clear(self) -> None:
        if self._status_len <= 0:
            return
        sys.stderr.write("\r" + (" " * self._status_len) + "\r")
        sys.stderr.flush()
        self._status_len = 0

    def log_line(self, msg: str) -> None:
        self._clear()
        sys.stderr.write(msg.rstrip() + "\n")
        sys.stderr.flush()

    def _summarize(self, ev: JsonDict) -> str | None:
        t = ev.get("type")
        if not isinstance(t, str) or not t:
            return None
        if t == "heartbeat":
            elapsed_s = ev.get("elapsed_s")
            idle_s = ev.get("idle_s")
            if isinstance(elapsed_s, (int, float)) and isinstance(idle_s, (int, float)):
                return f"heartbeat elapsed={elapsed_s:.1f}s idle={idle_s:.1f}s"
            return "heartbeat"
        if t == "provider":
            sub = ev.get("event")
            if isinstance(sub, str) and sub:
                return f"provider:{sub}"
            return "provider"
        if t == "item.completed":
            item = ev.get("item")
            if isinstance(item, dict):
                it = item.get("type")
                if isinstance(it, str) and it:
                    return f"item.completed:{it}"
            return "item.completed"
        return t

    def on_event(self, ev: JsonDict) -> None:
        try:
            summary = self._summarize(ev)
            if summary is None:
                return
            self._last_summary = summary
            now = time.monotonic()
            if summary.startswith("heartbeat") and (now - self._last_line) >= 5.0:
                self._last_line = now
                self.log_line(f"setup: {self._label} | {summary}")
            self.render()
        except Exception:
            pass

    def render(self) -> None:
        elapsed = time.monotonic() - self._started
        if sys.stderr.isatty():
            self._frame_i = (self._frame_i + 1) % len(self._frames)
            frame = self._frames[self._frame_i]
            line = f"[{frame}] {self._label} | {elapsed:.1f}s | last={self._last_summary}"
            pad = ""
            if len(line) < self._status_len:
                pad = " " * (self._status_len - len(line))
            sys.stderr.write("\r" + line + pad)
            sys.stderr.flush()
            self._status_len = len(line)
            return

        # Non-interactive: print at most once every ~5s unless the summary changes.
        now = time.monotonic()
        if (now - self._last_line) < 5.0 and not self._last_summary.startswith("item.completed"):
            return
        self._last_line = now
        sys.stderr.write(f"setup: {self._label} | {elapsed:.1f}s | {self._last_summary}\n")
        sys.stderr.flush()

    def finish(self) -> None:
        self._clear()


def _default_report_format() -> str:
    return (
        "FORMAT: ORCH_JSON_V1\n"
        "\n"
        "Return exactly one JSON object (no markdown fences, no extra text) with:\n"
        '- status: "OK" | "NEEDS_INPUT" | "FAILED"\n'
        "- output: string (your report)\n"
        "- next_inputs: string (use empty string when not needed)\n"
        "- artifacts: optional list of strings (e.g. file paths you changed/created)\n"
        "\n"
        "Status semantics:\n"
        "- OK: you completed your step.\n"
        "- FAILED: changes are required (use next_inputs for actionable change requests / repro steps).\n"
        "- NEEDS_INPUT: you are blocked; ask precise questions in next_inputs.\n"
        "\n"
        "If you cannot proceed, set status=NEEDS_INPUT and describe what you need in next_inputs.\n"
    )


def _extract_first_json_object(text: str) -> JsonDict | None:
    decoder = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            idx = text.find("{", idx + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


@dataclass(frozen=True)
class AgentSpec:
    actor_id: str
    template_id: str  # coder/reviewer/tester/custom
    specialization: str
    custom_role: str


def _task_kind_title(kind: TaskKind) -> str:
    if kind == "feature":
        return "Feature implementation"
    if kind == "bug":
        return "Bug fix"
    if kind == "bootstrap":
        return "Bootstrap / greenfield"
    return kind


def _task_rules_appendix(kind: TaskKind, *, template_id: str) -> str:
    if kind == "feature":
        if template_id == "coder":
            return (
                "Task-specific guardrails (feature):\n"
                "- Do not break existing functionality; treat regressions as failures.\n"
                "- Prefer backwards-compatible changes; call out any breaking risk explicitly.\n"
                "- Add or update validation that protects existing behavior (tests/checks).\n"
                "- If the feature has risky rollout, propose a gradual enablement strategy.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific guardrails (feature):\n"
                "- Look for regressions, subtle behavior changes, and backwards-compat risks.\n"
                "- Check that validation covers both the new behavior and key existing flows.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific guardrails (feature):\n"
                "- Include regression coverage for existing user flows near the change.\n"
                "- Prefer a mix of automated checks and a targeted manual validation plan.\n"
            )
    if kind == "bug":
        if template_id == "coder":
            return (
                "Task-specific guardrails (bug):\n"
                "- Keep the fix scope minimal and focused on the root cause.\n"
                "- Prefer to capture a repro and a regression test; avoid masking symptoms.\n"
                "- Clearly state expected vs actual behavior and how the fix changes it.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific guardrails (bug):\n"
                "- Verify the change addresses the root cause, not just symptoms.\n"
                "- Watch for overly broad fixes and hidden behavior changes.\n"
                "- Ensure there is a regression test or a clear repro/verification plan.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific guardrails (bug):\n"
                "- Capture repro steps and verify the fix closes them.\n"
                "- Add regression scenarios around the fix boundary.\n"
            )
    if kind == "bootstrap":
        if template_id == "coder":
            return (
                "Task-specific guardrails (bootstrap):\n"
                "- Define clear scope: what is in vs out.\n"
                "- Prefer a minimal, complete, runnable baseline over lots of partial ideas.\n"
                "- Provide run/validate instructions and explicit acceptance criteria.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific guardrails (bootstrap):\n"
                "- Check that requirements and acceptance criteria are explicit and testable.\n"
                "- Look for missing glue: docs, run instructions, and basic validation.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific guardrails (bootstrap):\n"
                "- Define acceptance tests that prove the project works end-to-end.\n"
                "- Prefer small, high-signal smoke tests over exhaustive coverage.\n"
            )
    return ""


def _task_target_appendix(kind: TaskKind, *, template_id: str) -> str:
    if kind == "feature":
        if template_id == "coder":
            return (
                "Task-specific deliverables (feature):\n"
                "- Call out backwards-compat considerations and how regressions are avoided.\n"
                "- Provide validation steps that cover both new behavior and key existing behavior.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific deliverables (feature):\n"
                "- If changes are required, write a concise checklist in next_inputs.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific deliverables (feature):\n"
                "- Include regression scenarios; focus on highest-risk paths.\n"
            )
    if kind == "bug":
        if template_id == "coder":
            return (
                "Task-specific deliverables (bug):\n"
                "- Include a root-cause summary and a minimal fix description.\n"
                "- Include repro/verification steps and (if feasible) a regression test.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific deliverables (bug):\n"
                "- If rejecting, include the minimal repro/verification evidence needed.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific deliverables (bug):\n"
                "- Provide a crisp repro and a boundary-focused regression set.\n"
            )
    if kind == "bootstrap":
        if template_id == "coder":
            return (
                "Task-specific deliverables (bootstrap):\n"
                "- Provide a runnable baseline and clear run/validate instructions.\n"
                "- Define explicit acceptance criteria and what is out of scope.\n"
            )
        if template_id == "reviewer":
            return (
                "Task-specific deliverables (bootstrap):\n"
                "- Call out missing requirements, unclear acceptance criteria, or unsafe defaults.\n"
            )
        if template_id == "tester":
            return (
                "Task-specific deliverables (bootstrap):\n"
                "- Provide smoke tests / acceptance tests for the baseline.\n"
            )
    return ""


def _task_context_appendix(task: TaskConfig | None) -> str:
    if task is None:
        return ""
    parts: list[str] = []
    parts.append("## Task type")
    parts.append("")
    parts.append(_task_kind_title(task.kind))
    if task.details_md.strip():
        parts.append("")
        parts.append("## Task details")
        parts.append("")
        parts.append(task.details_md.strip())
    parts.append("")
    return "\n".join(parts)


def _base_packet_docs(
    *,
    workspace_dir: Path,
    goal: str,
    task: TaskConfig | None,
    agent: AgentSpec,
    is_first: bool,
) -> dict[str, str]:
    if agent.template_id == "custom":
        role_md = "# ROLE\n\n" + agent.custom_role.strip() + "\n"
        rules_md = "# RULES\n\nFollow ROLE, TARGET, and REPORT_FORMAT.\n"
        context_md = "# CONTEXT\n\nYou have access to a workspace/repository to modify.\n"
        target_md = "# TARGET\n\n" f"Goal:\n{goal}\n\nDescribe what you will do in this step.\n"
    else:
        tmpl = get_template(agent.template_id)
        role_md = tmpl.base_role_md
        if agent.specialization.strip():
            role_md = (
                role_md.rstrip() + "\n\n" + f"Specialization: {agent.specialization.strip()}\n"
            )
        rules_md = tmpl.base_rules_md
        context_md = tmpl.base_context_md
        target_md = tmpl.build_target_md(goal, agent.actor_id, None)

    task_kind = task.kind if task is not None else "feature"
    task_target_extra = _task_target_appendix(task_kind, template_id=agent.template_id)
    if task_target_extra.strip():
        target_md = target_md.rstrip() + "\n\n" + task_target_extra.rstrip() + "\n"

    task_rules_extra = _task_rules_appendix(task_kind, template_id=agent.template_id)
    if task_rules_extra.strip():
        rules_md = rules_md.rstrip() + "\n\n" + task_rules_extra.rstrip() + "\n"

    context_md = (
        context_md.rstrip()
        + "\n\n"
        + f"Goal: {goal}\n"
        + f"Workspace root: {workspace_dir.as_posix()}\n"
        + "\n"
    )
    task_ctx = _task_context_appendix(task)
    if task_ctx.strip():
        context_md = context_md.rstrip() + "\n\n" + task_ctx.rstrip() + "\n"

    if is_first:
        inputs_md = (
            "# INPUTS\n\n"
            f"Initial request:\n{goal}\n\n"
            f"Task type: {_task_kind_title(task_kind)}\n\n"
            "Additional constraints:\n"
            "- (none provided)\n\n"
            "Notes:\n"
            "- Use this file for any extra requirements or clarifications.\n"
        )
        if task is not None and task.details_md.strip():
            inputs_md = (
                inputs_md.rstrip()
                + "\n\n"
                + "Task details:\n\n"
                + task.details_md.strip()
                + "\n"
            )
    else:
        inputs_md = (
            "# INPUTS\n\n"
            "This file will contain any material you should act on for this step.\n"
            "If it is empty/insufficient, set status=NEEDS_INPUT.\n"
        )

    return {
        "ROLE": role_md.rstrip() + "\n",
        "TARGET": target_md.rstrip() + "\n",
        "RULES": rules_md.rstrip() + "\n",
        "CONTEXT": context_md.rstrip() + "\n",
        "REPORT_FORMAT": _default_report_format(),
        "INPUTS": inputs_md,
        "NOTES": "# NOTES\n\n(append-only; managed by orchestrator)\n",
    }


def _refine_packet_docs_via_provider(
    provider: Provider,
    *,
    artifacts_dir: Path,
    goal: str,
    task: TaskConfig | None,
    agent: AgentSpec,
    base_role_md: str,
    base_target_md: str,
    base_rules_md: str,
    base_context_md: str,
    timeout_s: float,
    idle_timeout_s: float,
    on_event: Callable[[JsonDict], None] | None = None,
) -> tuple[dict[str, str] | None, JsonDict]:
    specialization = agent.specialization.strip() or "(none)"
    task_kind = task.kind if task is not None else "feature"
    task_details = task.details_md.strip() if task is not None else ""
    task_details_block = ""
    if task_details:
        task_details_block = f"Task details:\n{task_details}\n\n"

    prompt = "".join(
        [
            "You are refining packet documents (ROLE/TARGET/RULES/CONTEXT) for an isolated agent.\n\n",
            "Goal: produce high-quality instructions with a strong quality bar.\n",
            "The documents should be practical, specific, and deterministic.\n\n",
            "Hard constraints:\n",
            "- Keep the top-level headings exactly: # ROLE, # TARGET, # RULES, # CONTEXT.\n",
            "- Do not mention other agents, pipeline order, or orchestration.\n",
            "- Keep rules language-agnostic (no language/framework-specific advice).\n",
            "- Do not output markdown fences.\n",
            "- Do not change the output protocol: the agent must follow REPORT_FORMAT.md.\n\n",
            "Content guidelines (make this substantially more detailed than the base docs):\n",
            "- ROLE: responsibilities, non-goals, decision making, definition of done.\n",
            "- TARGET: explicit deliverables and success criteria for this step.\n",
            "- RULES: correctness, minimal change, avoiding hallucinations, validation expectations.\n",
            "- CONTEXT: how to use INPUTS.md, how to handle ambiguity, where to look in workspace.\n\n",
            "Return exactly one JSON object with keys:\n",
            "- role_md\n",
            "- target_md\n",
            "- rules_md\n",
            "- context_md\n",
            "No extra keys. Values must be non-empty strings.\n\n",
            f"Goal:\n{goal}\n\n",
            f"Task type: {_task_kind_title(task_kind)}\n\n",
            task_details_block,
            f"Agent id: {agent.actor_id}\n",
            f"Template: {agent.template_id}\n",
            f"Specialization: {specialization}\n\n",
            "REPORT_FORMAT.md (do not edit, but your docs must be consistent with it):\n",
            f"{_default_report_format()}\n",
            "\n",
            "Current ROLE.md:\n",
            f"{base_role_md}\n\n",
            "Current TARGET.md:\n",
            f"{base_target_md}\n\n",
            "Current RULES.md:\n",
            f"{base_rules_md}\n\n",
            "Current CONTEXT.md:\n",
            f"{base_context_md}\n",
        ]
    )

    res = provider.run(
        prompt,
        artifacts_dir=artifacts_dir,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
        on_event=on_event,
    )

    obj = _extract_first_json_object(res.final_text)
    expected = {"role_md", "target_md", "rules_md", "context_md"}
    if not isinstance(obj, dict) or set(obj.keys()) != expected:
        return None, res.metadata
    if not all(isinstance(obj[k], str) and obj[k].strip() for k in expected):
        return None, res.metadata

    return (
        {
            "ROLE": obj["role_md"].rstrip() + "\n",
            "TARGET": obj["target_md"].rstrip() + "\n",
            "RULES": obj["rules_md"].rstrip() + "\n",
            "CONTEXT": obj["context_md"].rstrip() + "\n",
        },
        res.metadata,
    )


@dataclass(frozen=True)
class SetupResult:
    workspace_dir: Path
    orchestrator_dir: Path
    pipeline_path: Path


def run_interactive_setup(*, workspace_dir: Path) -> SetupResult:
    workspace_dir = Path(_prompt_line("Workspace path?", default=str(workspace_dir))).expanduser()
    workspace_dir = workspace_dir.resolve()
    orch_dir = orchestrator_root(workspace_dir)
    packets_dir = orch_dir / "packets"
    runs_dir = orch_dir / "runs"
    pipeline_path = orch_dir / "pipeline.json"

    if orch_dir.exists() and (pipeline_path.exists() or packets_dir.exists()):
        overwrite = _prompt_yes_no(
            "Overwrite existing orchestrator config?",
            default=False,
        )
        if not overwrite:
            print("Setup aborted (no changes made).")
            return SetupResult(
                workspace_dir=workspace_dir,
                orchestrator_dir=orch_dir,
                pipeline_path=pipeline_path,
            )
        if packets_dir.exists():
            shutil.rmtree(packets_dir)

    print("Quick setup will create:")
    print(f"- {packets_dir}")
    print(f"- {runs_dir}")
    print("")

    task_kind = _prompt_choice(
        "Task type (feature/bug/bootstrap)",
        choices=["feature", "bug", "bootstrap"],
        default="feature",
    )
    goal_label = "What is the goal of orchestration?"
    if task_kind == "feature":
        goal_label = "What feature do you want to implement?"
    elif task_kind == "bug":
        goal_label = "What bug do you want to fix?"
    elif task_kind == "bootstrap":
        goal_label = "What do you want to build from scratch?"
    goal = _prompt_line(goal_label)

    print("")
    if task_kind == "feature":
        print("Task details hint (optional): acceptance criteria, constraints, rollout/compat notes.")
    elif task_kind == "bug":
        print("Task details hint (optional): repro steps, expected vs actual, scope constraints.")
    else:
        print("Task details hint (optional): requirements, non-goals, constraints, acceptance criteria.")
    details = _prompt_multiline_optional("Task details")
    task = TaskConfig(kind=task_kind, details_md=details)
    print("")

    preset = _prompt_choice(
        "Orchestration preset (crt/cr/custom)",
        choices=["crt", "cr", "custom"],
        default="crt",
    )
    max_returns = 3
    if preset == "crt":
        max_returns = _prompt_int(
            "Max returns (review/test -> coder)",
            min_value=0,
            max_value=10,
            default=3,
        )
        num_agents = 3
    elif preset == "cr":
        max_returns = _prompt_int(
            "Max returns (review -> coder)",
            min_value=0,
            max_value=10,
            default=3,
        )
        num_agents = 2
    else:
        num_agents = _prompt_int(
            "How many agents do you want? (2-4)", min_value=2, max_value=4, default=2
        )

    use_codex = _prompt_yes_no("Do you want Codex CLI as provider?", default=False)
    if use_codex:
        cmd_raw = _prompt_line(
            "Codex command (space-separated)",
            default="codex exec --json --full-auto",
        )
        command = tuple(cmd_raw.split())
        if "--json" not in command:
            print("WARN: Codex command does not include '--json'; JSONL parsing may fail.")
        provider_cfg = ProviderConfig(
            type="codex_cli",
            command=command,
            # Default to no hard timeout; rely on idle timeout for safety.
            # This avoids killing long-running but active runs (e.g. 30+ minutes).
            timeout_s=0.0,
            idle_timeout_s=600.0,
        )
    else:
        provider_cfg = ProviderConfig(type="deterministic")

    timeout_s = float(
        _prompt_line(
            "Hard timeout seconds (0=disabled)", default=str(int(provider_cfg.timeout_s))
        )
    )
    idle_timeout_s = float(
        _prompt_line("Idle timeout seconds (0=disabled)", default=str(int(provider_cfg.idle_timeout_s)))
    )
    provider_cfg = ProviderConfig(
        type=provider_cfg.type,
        command=provider_cfg.command,
        timeout_s=timeout_s,
        idle_timeout_s=idle_timeout_s,
    )

    refine_now = False
    if provider_cfg.type == "codex_cli":
        refine_now = _prompt_yes_no("Use provider to refine packet documents now?", default=True)

    # Choose templates and specializations.
    agents: list[AgentSpec] = []
    if preset == "crt":
        for template_id in ("coder", "reviewer", "tester"):
            actor_id = template_id
            specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
            agents.append(
                AgentSpec(
                    actor_id=actor_id,
                    template_id=template_id,
                    specialization=specialization,
                    custom_role="",
                )
            )
    elif preset == "cr":
        for template_id in ("coder", "reviewer"):
            actor_id = template_id
            specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
            agents.append(
                AgentSpec(
                    actor_id=actor_id,
                    template_id=template_id,
                    specialization=specialization,
                    custom_role="",
                )
            )
    else:
        template_ids = list_template_ids() + ["custom"]
        for i in range(1, num_agents + 1):
            actor_id = f"agent_{i}"
            default_template = "custom"
            if i == 1:
                default_template = "coder"
            elif i == 2:
                default_template = "reviewer"
            elif i == 3:
                default_template = "tester"

            template_id = _prompt_choice(
                f"Template for {actor_id} ({'/'.join(template_ids)})",
                choices=template_ids,
                default=default_template,
            )
            if template_id == "custom":
                role = _prompt_line(f"Role for {actor_id}")
                agents.append(
                    AgentSpec(
                        actor_id=actor_id,
                        template_id="custom",
                        specialization="",
                        custom_role=role,
                    )
                )
            else:
                specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
                agents.append(
                    AgentSpec(
                        actor_id=actor_id,
                        template_id=template_id,
                        specialization=specialization,
                        custom_role="",
                    )
                )

    # Create base directories deterministically.
    runs_dir.mkdir(parents=True, exist_ok=True)
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Provider instance for optional refinement.
    provider: Provider
    if provider_cfg.type == "codex_cli":
        assert provider_cfg.command is not None
        provider = CodexCLIProvider(command=provider_cfg.command, cwd=workspace_dir)
    else:
        provider = DeterministicProvider()

    actor_cfgs: list[ActorConfig] = []
    for i, agent in enumerate(agents, start=1):
        actor_cfgs.append(
            ActorConfig(
                actor_id=agent.actor_id,
                packet_dir=agent.actor_id,
                include_paths_in_prompt=True,
            )
        )

        docs = _base_packet_docs(
            workspace_dir=workspace_dir,
            goal=goal,
            task=task,
            agent=agent,
            is_first=i == 1,
        )

        if refine_now and provider_cfg.type == "codex_cli":
            gen_dir = orch_dir / "setup_artifacts" / agent.actor_id
            gen_dir.mkdir(parents=True, exist_ok=True)
            ui = _SetupRefineUI(label=f"refine:{agent.actor_id}")
            refined, meta = _refine_packet_docs_via_provider(
                provider,
                artifacts_dir=gen_dir,
                goal=goal,
                task=task,
                agent=agent,
                base_role_md=docs["ROLE"],
                base_target_md=docs["TARGET"],
                base_rules_md=docs["RULES"],
                base_context_md=docs["CONTEXT"],
                timeout_s=provider_cfg.timeout_s,
                idle_timeout_s=provider_cfg.idle_timeout_s,
                on_event=ui.on_event,
            )
            ui.finish()
            if refined is not None:
                docs.update(refined)
            else:
                timed_out = bool(meta.get("timed_out"))
                idle_timed_out = bool(meta.get("idle_timed_out"))
                hint = ""
                if timed_out or idle_timed_out:
                    hint = f" (timed_out={timed_out}, idle_timed_out={idle_timed_out})"
                print(
                    "WARN: provider refinement failed for "
                    f"{agent.actor_id}; using base templates.{hint}"
                )
                print(f"WARN: see artifacts in {gen_dir}")

        packet_dir = packets_dir / agent.actor_id
        write_packet_documents(packet_dir, docs=docs)

    orchestration = None
    if preset == "crt":
        orchestration = OrchestrationConfig(preset="crt_v1", max_returns=max_returns)
    elif preset == "cr":
        orchestration = OrchestrationConfig(preset="cr_v1", max_returns=max_returns)
    pipeline = PipelineConfig(
        version=1,
        provider=provider_cfg,
        actors=tuple(actor_cfgs),
        orchestration=orchestration,
        goal=goal,
        task=task,
    )
    save_pipeline(pipeline_path, pipeline)

    print("")
    print("Setup complete.")
    print(f"Workspace: {workspace_dir}")
    print(f"Pipeline:  {pipeline_path}")

    return SetupResult(
        workspace_dir=workspace_dir,
        orchestrator_dir=orch_dir,
        pipeline_path=pipeline_path,
    )

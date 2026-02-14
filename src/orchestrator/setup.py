from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.core.packet import write_packet_documents
from src.core.runtime import (
    ActorConfig,
    DeliveryConfig,
    GitDefaultsConfig,
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
from src.orchestrator.spec import AgentSpec
from src.orchestrator.wizard import run_setup_wizard_analyze, run_setup_wizard_write_packet
from src.providers.base import Provider
from src.providers.claude_cli import ClaudeCLIProvider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider
from src.providers.gemini_cli import GeminiCLIProvider


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


def _task_kind_title(kind: TaskKind) -> str:
    if kind == "feature":
        return "Feature implementation"
    if kind == "bug":
        return "Bug fix"
    if kind == "bootstrap":
        return "Bootstrap / greenfield"
    if kind == "other":
        return "Other"
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
        "Task type (feature/bug/bootstrap/other)",
        choices=["feature", "bug", "bootstrap", "other"],
        default="feature",
    )
    goal_label = "What is the goal of orchestration?"
    if task_kind == "feature":
        goal_label = "What feature do you want to implement?"
    elif task_kind == "bug":
        goal_label = "What bug do you want to fix?"
    elif task_kind == "bootstrap":
        goal_label = "What do you want to build from scratch?"
    elif task_kind == "other":
        goal_label = "What do you want to do?"
    goal = _prompt_line(goal_label)

    print("")
    if task_kind == "feature":
        print("Task details hint (optional): acceptance criteria, constraints, rollout/compat notes.")
    elif task_kind == "bug":
        print("Task details hint (optional): repro steps, expected vs actual, scope constraints.")
    elif task_kind == "bootstrap":
        print("Task details hint (optional): requirements, non-goals, constraints, acceptance criteria.")
    else:
        print("Task details hint (optional): constraints, acceptance criteria, non-goals, risks.")
    details = _prompt_multiline_optional("Task details")
    task = TaskConfig(kind=task_kind, details_md=details)
    print("")

    if task_kind == "bootstrap" and not (workspace_dir / ".git").exists():
        init_git = _prompt_yes_no(
            "Workspace is not a git repository. Initialize git here (git init)?",
            default=True,
        )
        if init_git:
            p = subprocess.run(
                ["git", "init"],
                cwd=str(workspace_dir),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if p.returncode != 0:
                out = (p.stdout or "").strip()
                print(f"WARN: git init failed (exit={p.returncode}): {out}")
            else:
                print("git: initialized repository")
                print("")

    preset = _prompt_choice(
        "Workflow preset (crt/cr/c/r/t/d/custom)",
        choices=["crt", "cr", "c", "r", "t", "d", "custom"],
        default="crt",
    )

    # Choose templates and specializations.
    agents: list[AgentSpec] = []
    orchestration: OrchestrationConfig | None = None

    if preset == "crt":
        max_returns = _prompt_int(
            "Max returns (review/test -> coder)",
            min_value=0,
            max_value=10,
            default=3,
        )
        for template_id in ("coder", "reviewer", "tester"):
            specialization = _prompt_optional_line(f"Specialization for {template_id} (optional)")
            agents.append(
                AgentSpec(
                    actor_id=template_id,
                    template_id=template_id,
                    specialization=specialization,
                )
            )
        orchestration = OrchestrationConfig(
            preset="linear_v1",
            max_returns=max_returns,
            return_to="coder",
            return_from=("reviewer", "tester"),
        )

    elif preset == "cr":
        max_returns = _prompt_int(
            "Max returns (review -> coder)",
            min_value=0,
            max_value=10,
            default=3,
        )
        for template_id in ("coder", "reviewer"):
            specialization = _prompt_optional_line(f"Specialization for {template_id} (optional)")
            agents.append(
                AgentSpec(
                    actor_id=template_id,
                    template_id=template_id,
                    specialization=specialization,
                )
            )
        orchestration = OrchestrationConfig(
            preset="linear_v1",
            max_returns=max_returns,
            return_to="coder",
            return_from=("reviewer",),
        )

    elif preset == "c":
        specialization = _prompt_optional_line("Specialization for coder (optional)")
        agents.append(AgentSpec(actor_id="coder", template_id="coder", specialization=specialization))

    elif preset == "r":
        specialization = _prompt_optional_line("Specialization for reviewer (optional)")
        agents.append(
            AgentSpec(actor_id="reviewer", template_id="reviewer", specialization=specialization)
        )

    elif preset == "t":
        specialization = _prompt_optional_line("Specialization for tester (optional)")
        agents.append(AgentSpec(actor_id="tester", template_id="tester", specialization=specialization))

    elif preset == "d":
        specialization = _prompt_optional_line("Specialization for devops (optional)")
        agents.append(
            AgentSpec(actor_id="devops", template_id="devops", specialization=specialization)
        )

    else:
        stage_count = _prompt_int(
            "How many stages do you want? (1-6)",
            min_value=1,
            max_value=6,
            default=2,
        )
        template_ids = list_template_ids() + ["custom"]
        used_actor_ids: set[str] = set()

        for i in range(1, stage_count + 1):
            template_id = _prompt_choice(
                f"Template for stage {i} ({'/'.join(template_ids)})",
                choices=template_ids,
                default=("coder" if i == 1 else "reviewer"),
            )

            default_actor_id = template_id if template_id != "custom" else f"agent_{i}"
            while True:
                actor_id = _prompt_line(f"Actor id for stage {i}", default=default_actor_id)
                actor_id = actor_id.strip()
                if not actor_id:
                    continue
                if any(ch.isspace() for ch in actor_id):
                    print("Actor id must not contain whitespace.")
                    continue
                if actor_id in used_actor_ids:
                    print("Actor id must be unique.")
                    continue
                used_actor_ids.add(actor_id)
                break

            if template_id == "custom":
                role = _prompt_line(f"Role for {actor_id}")
                agents.append(AgentSpec(actor_id=actor_id, template_id="custom", custom_role=role))
            else:
                specialization = _prompt_optional_line(f"Specialization for {actor_id} (optional)")
                agents.append(
                    AgentSpec(
                        actor_id=actor_id,
                        template_id=template_id,
                        specialization=specialization,
                    )
                )

        if stage_count >= 2:
            enable_returns = _prompt_yes_no("Enable return-on-failure loop?", default=True)
            if enable_returns:
                max_returns = _prompt_int(
                    "Max returns (FAILED -> return_to)",
                    min_value=0,
                    max_value=10,
                    default=3,
                )
                default_return_to = "coder" if any(a.actor_id == "coder" for a in agents) else agents[0].actor_id
                return_to = _prompt_line("Return to actor_id", default=default_return_to).strip()
                if return_to not in {a.actor_id for a in agents}:
                    print("WARN: return_to is not a valid actor_id; using first stage actor_id.")
                    return_to = agents[0].actor_id
                default_from = ",".join(a.actor_id for a in agents if a.actor_id != return_to)
                raw_from = _prompt_line("Return from actor_ids (comma separated)", default=default_from)
                return_from = tuple(x.strip() for x in raw_from.split(",") if x.strip())
                return_from = tuple(x for x in return_from if x != return_to)
                orchestration = OrchestrationConfig(
                    preset="linear_v1",
                    max_returns=max_returns,
                    return_to=return_to,
                    return_from=return_from,
                )

    interaction_notes = _prompt_multiline_optional("Workflow/interaction notes")

    # Provider selection (only offer installed CLIs).
    available: list[str] = ["deterministic"]
    if shutil.which("codex") is not None:
        available.append("codex_cli")
    if shutil.which("gemini") is not None:
        available.append("gemini_cli")
    if shutil.which("claude") is not None:
        available.append("claude_cli")

    def _preferred_ai_provider() -> str:
        for t in ("codex_cli", "gemini_cli", "claude_cli"):
            if t in available:
                return t
        return "deterministic"

    def _prompt_provider_cfg(*, label: str, default_type: str) -> ProviderConfig:
        provider_type = _prompt_choice(
            f"{label} ({'/'.join(available)})",
            choices=available,
            default=default_type if default_type in available else "deterministic",
        )

        if provider_type == "codex_cli":
            cmd_raw = _prompt_line(
                "Codex command (shell syntax)",
                default="codex exec --json --full-auto",
            )
            command = tuple(shlex.split(cmd_raw))
            if "--json" not in command:
                print("WARN: Codex command does not include '--json'; JSONL parsing may fail.")
            return ProviderConfig(
                type="codex_cli",
                command=command,
                timeout_s=0.0,
                idle_timeout_s=600.0,
            )

        if provider_type == "gemini_cli":
            cmd_raw = _prompt_line(
                "Gemini command (shell syntax)",
                default='gemini --output-format text --approval-mode yolo -p \" \"',
            )
            command = tuple(shlex.split(cmd_raw))
            return ProviderConfig(
                type="gemini_cli",
                command=command,
                timeout_s=0.0,
                idle_timeout_s=600.0,
            )

        if provider_type == "claude_cli":
            cmd_raw = _prompt_line(
                "Claude command (shell syntax)",
                default="claude -p --output-format text --input-format text --permission-mode acceptEdits",
            )
            command = tuple(shlex.split(cmd_raw))
            return ProviderConfig(
                type="claude_cli",
                command=command,
                timeout_s=0.0,
                idle_timeout_s=600.0,
            )

        return ProviderConfig(type="deterministic")

    one_provider = _prompt_yes_no("Use one provider for wizard + all agents?", default=True)
    provider_overrides: dict[str, ProviderConfig] = {}

    wizard_provider_cfg: ProviderConfig
    pipeline_provider_cfg: ProviderConfig

    if one_provider:
        base_default = _preferred_ai_provider()
        pipeline_provider_cfg = _prompt_provider_cfg(label="Provider", default_type=base_default)
        wizard_provider_cfg = pipeline_provider_cfg
    else:
        wizard_provider_cfg = _prompt_provider_cfg(
            label="Wizard provider",
            default_type=_preferred_ai_provider(),
        )
        one_agents = _prompt_yes_no("Use one provider for all agents?", default=True)
        if one_agents:
            pipeline_provider_cfg = _prompt_provider_cfg(
                label="Agent provider",
                default_type=wizard_provider_cfg.type,
            )
        else:
            last_type = wizard_provider_cfg.type
            if last_type == "deterministic":
                last_type = _preferred_ai_provider()
            for a in agents:
                cfg = _prompt_provider_cfg(label=f"Provider for {a.actor_id}", default_type=last_type)
                provider_overrides[a.actor_id] = cfg
                last_type = cfg.type
            pipeline_provider_cfg = provider_overrides[agents[0].actor_id]

    selected_cfgs = [wizard_provider_cfg, pipeline_provider_cfg, *provider_overrides.values()]
    any_cli = any(c.type != "deterministic" for c in selected_cfgs)
    default_timeout_s = "0" if any_cli else "120"
    default_idle_timeout_s = "600" if any_cli else "30"

    timeout_s = float(_prompt_line("Hard timeout seconds (0=disabled)", default=default_timeout_s))
    idle_timeout_s = float(
        _prompt_line("Idle timeout seconds (0=disabled)", default=default_idle_timeout_s)
    )

    def _with_timeouts(cfg: ProviderConfig) -> ProviderConfig:
        return ProviderConfig(
            type=cfg.type,
            command=cfg.command,
            timeout_s=timeout_s,
            idle_timeout_s=idle_timeout_s,
        )

    wizard_provider_cfg = _with_timeouts(wizard_provider_cfg)
    pipeline_provider_cfg = _with_timeouts(pipeline_provider_cfg)
    provider_overrides = {k: _with_timeouts(v) for k, v in provider_overrides.items()}

    default_git_mode = "off"
    if (workspace_dir / ".git").exists():
        default_git_mode = "branch"
    git_mode = _prompt_choice(
        "Default git safety for runs (branch/check/off)",
        choices=["branch", "check", "off"],
        default=default_git_mode,
    )
    git_defaults: GitDefaultsConfig | None = None
    branch_pref = ""
    commit_pref = ""
    if git_mode == "off":
        git_defaults = GitDefaultsConfig(mode="off")
    else:
        branch_prefix = _prompt_line("Git branch prefix", default="orch/")
        auto_commit = _prompt_yes_no("Auto-commit after implementer steps?", default=True)
        branch_pref = _prompt_optional_line(
            "Branch naming preference (optional, e.g. 'JIRA-123/feat-{slug}')"
        )
        commit_pref = _prompt_optional_line(
            "Commit message preference (optional, e.g. 'conventional commits')"
        )
        git_defaults = GitDefaultsConfig(
            mode=git_mode, branch_prefix=branch_prefix, auto_commit=auto_commit
        )

    mr_mode = _prompt_choice(
        "Merge request at end (off/instructions)",
        choices=["off", "instructions"],
        default="off",
    )
    delivery: DeliveryConfig | None = None
    if mr_mode == "instructions":
        remote = _prompt_line("MR remote", default="origin")
        target_branch = _prompt_line("MR target branch", default="main")
        delivery = DeliveryConfig(mr_mode="instructions", remote=remote, target_branch=target_branch)

    # Create base directories deterministically.
    runs_dir.mkdir(parents=True, exist_ok=True)
    packets_dir.mkdir(parents=True, exist_ok=True)

    # Provider instance for wizarding/refinement.
    provider: Provider
    if wizard_provider_cfg.type == "deterministic":
        provider = DeterministicProvider()
    elif wizard_provider_cfg.type == "codex_cli":
        assert wizard_provider_cfg.command is not None
        provider = CodexCLIProvider(command=wizard_provider_cfg.command, cwd=workspace_dir)
    elif wizard_provider_cfg.type == "gemini_cli":
        assert wizard_provider_cfg.command is not None
        provider = GeminiCLIProvider(command=wizard_provider_cfg.command, cwd=workspace_dir)
    elif wizard_provider_cfg.type == "claude_cli":
        assert wizard_provider_cfg.command is not None
        provider = ClaudeCLIProvider(command=wizard_provider_cfg.command, cwd=workspace_dir)
    else:
        raise ValueError(f"Unsupported provider type: {wizard_provider_cfg.type!r}")

    run_wizard = False
    wizard_parallel = False
    if wizard_provider_cfg.type != "deterministic":
        run_wizard = _prompt_yes_no("Run setup wizard (AI) now?", default=True)
        if run_wizard:
            wizard_parallel = _prompt_yes_no("Wizard: generate per-agent packets in parallel?", default=True)

    actor_cfgs: list[ActorConfig] = []
    docs_by_actor: dict[str, dict[str, str]] = {}
    for i, agent in enumerate(agents, start=1):
        override = provider_overrides.get(agent.actor_id)
        if override is not None and override == pipeline_provider_cfg:
            override = None
        actor_cfgs.append(
            ActorConfig(
                actor_id=agent.actor_id,
                packet_dir=agent.actor_id,
                include_paths_in_prompt=True,
                provider=override,
            )
        )

        base_docs = _base_packet_docs(
            workspace_dir=workspace_dir,
            goal=goal,
            task=task,
            agent=agent,
            is_first=i == 1,
        )

        docs_by_actor[agent.actor_id] = base_docs

    brief_md = ""
    if run_wizard:
        notes = interaction_notes.strip()
        if branch_pref.strip():
            notes = (notes + "\n\n" if notes else "") + f"Branch naming preference: {branch_pref.strip()}\n"
        if commit_pref.strip():
            notes = (notes + "\n\n" if notes else "") + f"Commit message preference: {commit_pref.strip()}\n"

        wiz_dir = orch_dir / "setup_artifacts" / "_wizard"
        wiz_dir.mkdir(parents=True, exist_ok=True)
        ui = _SetupRefineUI(label="wizard:analyze")
        wiz, meta = run_setup_wizard_analyze(
            provider,
            workspace_dir=workspace_dir,
            goal=goal,
            task=task,
            agents=tuple(agents),
            orchestration=orchestration,
            git_defaults=git_defaults,
            delivery=delivery,
            interaction_notes=notes,
            artifacts_dir=wiz_dir,
            timeout_s=wizard_provider_cfg.timeout_s,
            idle_timeout_s=wizard_provider_cfg.idle_timeout_s,
            on_event=ui.on_event,
        )
        ui.finish()
        if wiz is None:
            timed_out = bool(meta.get("timed_out"))
            idle_timed_out = bool(meta.get("idle_timed_out"))
            hint = ""
            if timed_out or idle_timed_out:
                hint = f" (timed_out={timed_out}, idle_timed_out={idle_timed_out})"
            print(f"WARN: setup wizard analyze failed; using base templates.{hint}")
            print(f"WARN: see artifacts in {wiz_dir}")
        else:
            brief_md = wiz.brief_md
            (orch_dir / "BRIEF.md").write_text(brief_md, encoding="utf-8")
            if wiz.git is not None:
                git_defaults = wiz.git
            if wiz.delivery is not None:
                delivery = wiz.delivery
            if wiz.orchestration is not None:
                orchestration = wiz.orchestration

            def _refine_one(agent: AgentSpec) -> tuple[str, dict[str, str] | None, JsonDict]:
                gen_dir = orch_dir / "setup_artifacts" / f"wizard_{agent.actor_id}"
                gen_dir.mkdir(parents=True, exist_ok=True)
                refined, meta2 = run_setup_wizard_write_packet(
                    provider,
                    goal=goal,
                    task=task,
                    agent=agent,
                    brief_md=brief_md,
                    base_docs=docs_by_actor[agent.actor_id],
                    artifacts_dir=gen_dir,
                    timeout_s=wizard_provider_cfg.timeout_s,
                    idle_timeout_s=wizard_provider_cfg.idle_timeout_s,
                    on_event=None,
                )
                if refined is None:
                    return agent.actor_id, None, meta2
                return agent.actor_id, refined.docs, meta2

            results: dict[str, dict[str, str]] = {}
            failures: dict[str, JsonDict] = {}
            if wizard_parallel and len(agents) > 1:
                with ThreadPoolExecutor(max_workers=min(4, len(agents))) as ex:
                    futs = [ex.submit(_refine_one, a) for a in agents]
                    for fut in as_completed(futs):
                        actor_id, refined_docs, meta2 = fut.result()
                        if refined_docs is None:
                            failures[actor_id] = meta2
                        else:
                            results[actor_id] = refined_docs
            else:
                for a in agents:
                    ui2 = _SetupRefineUI(label=f"wizard:packet:{a.actor_id}")
                    refined, meta2 = run_setup_wizard_write_packet(
                        provider,
                        goal=goal,
                        task=task,
                        agent=a,
                        brief_md=brief_md,
                        base_docs=docs_by_actor[a.actor_id],
                        artifacts_dir=(orch_dir / "setup_artifacts" / f"wizard_{a.actor_id}"),
                        timeout_s=wizard_provider_cfg.timeout_s,
                        idle_timeout_s=wizard_provider_cfg.idle_timeout_s,
                        on_event=ui2.on_event,
                    )
                    ui2.finish()
                    if refined is None:
                        failures[a.actor_id] = meta2
                    else:
                        results[a.actor_id] = refined.docs

            for actor_id, refined_docs in results.items():
                docs_by_actor[actor_id].update(refined_docs)

            for actor_id, meta2 in failures.items():
                timed_out = bool(meta2.get("timed_out"))
                idle_timed_out = bool(meta2.get("idle_timed_out"))
                hint = ""
                if timed_out or idle_timed_out:
                    hint = f" (timed_out={timed_out}, idle_timed_out={idle_timed_out})"
                print(f"WARN: setup wizard packet failed for {actor_id}; using base templates.{hint}")
                print(f"WARN: see artifacts in {orch_dir / 'setup_artifacts' / f'wizard_{actor_id}'}")

    for agent in agents:
        packet_dir = packets_dir / agent.actor_id
        write_packet_documents(packet_dir, docs=docs_by_actor[agent.actor_id])

    pipeline = PipelineConfig(
        version=1,
        provider=pipeline_provider_cfg,
        actors=tuple(actor_cfgs),
        orchestration=orchestration,
        goal=goal,
        task=task,
        git_defaults=git_defaults,
        delivery=delivery,
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

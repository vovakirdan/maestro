from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.core.actor import Actor
from src.core.packet import Packet
from src.core.runtime import (
    ActorConfig,
    PipelineConfig,
    ProviderConfig,
    generate_run_id,
    load_pipeline,
    orchestrator_root,
)
from src.core.types import JsonDict, Status, StructuredReport
from src.orchestrator.validator import ReportValidator, retry_instructions
from src.providers.base import Provider
from src.providers.claude_cli import ClaudeCLIProvider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider
from src.providers.gemini_cli import GeminiCLIProvider


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_RE_SLUG_SAFE = re.compile(r"[^a-z0-9._/-]+")


def _sanitize_slug(value: str) -> str:
    v = value.strip().lower().replace(" ", "-")
    v = _RE_SLUG_SAFE.sub("-", v)
    v = re.sub(r"-{2,}", "-", v).strip("-")
    return v[:64] if v else "task"


_RE_BRANCH_SAFE = re.compile(r"[^A-Za-z0-9._/-]+")


def _sanitize_branch_name(value: str) -> str:
    """
    Best-effort sanitization for git branch names.
    """
    v = value.strip().replace(" ", "-")
    v = _RE_BRANCH_SAFE.sub("-", v)
    v = re.sub(r"-{2,}", "-", v)
    v = v.strip("-/")  # avoid empty segments at ends
    v = re.sub(r"/{2,}", "/", v)
    return v or "orch"


def _format_template(template: str, *, values: JsonDict) -> str:
    """
    Safe-ish string formatting for user/wizard-provided templates.
    """
    try:
        return template.format(**values)
    except Exception:
        return template


class TimelineWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: JsonDict) -> None:
        event = dict(event)
        event.setdefault("ts_utc", _now_iso_utc())
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: Status
    run_dir: Path
    final_report: StructuredReport | None
    final_text: str


PlanMode = Literal["none", "auto", "user"]


@dataclass(frozen=True)
class PlanRequest:
    mode: PlanMode
    text: str = ""


GitMode = Literal["off", "check", "branch"]


@dataclass(frozen=True)
class GitPolicy:
    mode: GitMode = "off"
    branch_prefix: str = "orch/"
    auto_commit: bool = False
    branch_template: str | None = None
    branch_slug: str | None = None
    commit_message_template: str | None = None


@dataclass
class OrchestratorEngine:
    orchestrator_dir: Path
    pipeline: PipelineConfig

    @staticmethod
    def load(*, workspace_dir: Path) -> OrchestratorEngine:
        orch_dir = orchestrator_root(workspace_dir.resolve())
        pipeline_path = orch_dir / "pipeline.json"
        pipeline = load_pipeline(pipeline_path)
        return OrchestratorEngine(orchestrator_dir=orch_dir, pipeline=pipeline)

    def _make_provider_from_cfg(self, cfg: ActorConfig | ProviderConfig) -> Provider:
        """
        Construct a provider instance from either a ProviderConfig or an ActorConfig
        (ActorConfig may carry a provider override).
        """
        if isinstance(cfg, ActorConfig):
            provider_cfg = cfg.provider or self.pipeline.provider
        else:
            provider_cfg = cfg

        cfg = provider_cfg
        if cfg.type == "deterministic":
            return DeterministicProvider()
        if cfg.type == "codex_cli":
            assert cfg.command is not None
            return CodexCLIProvider(command=cfg.command, cwd=self.orchestrator_dir.parent)
        if cfg.type == "gemini_cli":
            assert cfg.command is not None
            return GeminiCLIProvider(command=cfg.command, cwd=self.orchestrator_dir.parent)
        if cfg.type == "claude_cli":
            assert cfg.command is not None
            return ClaudeCLIProvider(command=cfg.command, cwd=self.orchestrator_dir.parent)
        raise ValueError(f"Unsupported provider type: {cfg.type!r}")

    def run(
        self,
        *,
        progress: Callable[[str], None] | None = None,
        on_event: Callable[[JsonDict], None] | None = None,
        plan: PlanRequest | None = None,
        git: GitPolicy | None = None,
        timeout_s_override: float | None = None,
        idle_timeout_s_override: float | None = None,
    ) -> RunOutcome:
        def log(msg: str) -> None:
            if progress:
                progress(msg)

        validator = ReportValidator()

        timeout_override = (
            float(timeout_s_override) if timeout_s_override is not None else None
        )
        idle_timeout_override = (
            float(idle_timeout_s_override) if idle_timeout_s_override is not None else None
        )
        if timeout_override is not None and timeout_override < 0:
            raise ValueError(f"timeout_s must be >= 0, got: {timeout_override}")
        if idle_timeout_override is not None and idle_timeout_override < 0:
            raise ValueError(f"idle_timeout_s must be >= 0, got: {idle_timeout_override}")

        run_id = generate_run_id()
        run_dir = self.orchestrator_dir / "runs" / run_id
        steps_dir = run_dir / "steps"
        timeline_path = run_dir / "timeline.jsonl"
        state_path = run_dir / "state.json"
        timeline = TimelineWriter(timeline_path)

        template_packets_dir = self.orchestrator_dir / "packets"
        run_packets_dir = run_dir / "packets"

        steps_dir.mkdir(parents=True, exist_ok=False)
        shutil.copytree(template_packets_dir, run_packets_dir)

        state: JsonDict = {
            "run_id": run_id,
            "started_at_utc": _now_iso_utc(),
            "status": "RUNNING",
            "steps": [],
            "provider_timeouts": {
                "timeout_s_override": timeout_override,
                "idle_timeout_s_override": idle_timeout_override,
                "pipeline_default": {
                    "timeout_s": self.pipeline.provider.timeout_s,
                    "idle_timeout_s": self.pipeline.provider.idle_timeout_s,
                },
            },
        }
        _write_json(state_path, state)
        timeline.append({"type": "run_started", "run_id": run_id})

        workspace_root = self.orchestrator_dir.parent

        def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", *args],
                cwd=str(workspace_root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

        def _git_repo_root() -> Path | None:
            p = _run_git(["rev-parse", "--show-toplevel"])
            if p.returncode != 0:
                return None
            raw = (p.stdout or "").strip()
            if not raw:
                return None
            try:
                return Path(raw).resolve()
            except Exception:
                return None

        def _git_current_branch() -> str | None:
            p = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
            if p.returncode != 0:
                return None
            b = (p.stdout or "").strip()
            return b or None

        def _git_is_clean() -> tuple[bool, str]:
            p = _run_git(["status", "--porcelain"])
            if p.returncode != 0:
                return False, (p.stderr or "").strip()
            out = (p.stdout or "").strip()
            return (out == ""), out

        def _git_filter_status_lines(
            status_porcelain: str, *, allow_prefixes: tuple[str, ...]
        ) -> tuple[tuple[str, ...], tuple[str, ...]]:
            kept: list[str] = []
            ignored: list[str] = []

            for raw in status_porcelain.splitlines():
                line = raw.rstrip("\n")
                if not line.strip():
                    continue

                # Porcelain v1 usually looks like: "XY <path>" or "?? <path>".
                path = ""
                if len(line) >= 4 and line[2] == " ":
                    path = line[3:]

                def is_allowed(p: str) -> bool:
                    p = p.strip()
                    if not p:
                        return False
                    if " -> " in p:
                        old, new = p.split(" -> ", 1)
                        return is_allowed(old) and is_allowed(new)
                    return any(p.startswith(prefix) for prefix in allow_prefixes)

                if path and is_allowed(path):
                    ignored.append(line)
                else:
                    kept.append(line)

            return tuple(kept), tuple(ignored)

        # Optional git safety policy (check cleanliness / create a dedicated branch for the run).
        if git is not None and git.mode != "off":
            repo_root = _git_repo_root()
            if repo_root is None:
                raise ValueError("Git policy requested, but workspace is not a git repository")
            if repo_root != workspace_root.resolve():
                raise ValueError(
                    "Git policy requested, but workspace is not the git repo root "
                    f"(repo_root={repo_root}, workspace={workspace_root.resolve()})"
                )

            clean, details = _git_is_clean()
            # The orchestrator control dir is typically untracked and should not block branch safety.
            # We still enforce cleanliness for the rest of the workspace.
            kept, ignored = _git_filter_status_lines(details, allow_prefixes=("orchestrator/",))
            if not clean and kept:
                raise ValueError(
                    "Workspace has uncommitted changes; commit/stash first or disable git policy. "
                    "status_porcelain:\n"
                    + "\n".join(kept)
                )
            if not clean and ignored:
                log(
                    "WARN: git policy ignoring uncommitted changes under orchestrator/ "
                    f"({len(ignored)} line(s))"
                )

            original_branch = _git_current_branch() or "unknown"
            state["git"] = {
                "mode": git.mode,
                "auto_commit": bool(getattr(git, "auto_commit", False)),
                "branch_prefix": git.branch_prefix,
                "branch_template": git.branch_template,
                "branch_slug": git.branch_slug,
                "commit_message_template": git.commit_message_template,
                "repo_root": repo_root.as_posix(),
                "original_branch": original_branch,
                "clean": not kept,
                "ignored_status_lines": list(ignored),
            }
            _write_json(state_path, state)
            timeline.append(
                {
                    "type": "git_checked",
                    "run_id": run_id,
                    "mode": git.mode,
                    "auto_commit": bool(getattr(git, "auto_commit", False)),
                    "branch_prefix": git.branch_prefix,
                    "branch_template": git.branch_template,
                    "branch_slug": git.branch_slug,
                    "commit_message_template": git.commit_message_template,
                    "repo_root": repo_root.as_posix(),
                    "original_branch": original_branch,
                }
            )

            if git.mode == "branch":
                prefix = (git.branch_prefix or "orch/").strip()
                if not prefix:
                    prefix = "orch/"
                if " " in prefix or "\t" in prefix or "\n" in prefix:
                    raise ValueError(f"Invalid git.branch_prefix (contains whitespace): {prefix!r}")
                slug = git.branch_slug or _sanitize_slug(self.pipeline.goal or run_id)
                date_utc = time.strftime("%Y%m%d", time.gmtime())
                values: JsonDict = {
                    "prefix": prefix,
                    "run_id": run_id,
                    "slug": slug,
                    "date_utc": date_utc,
                }
                if git.branch_template is not None and git.branch_template.strip():
                    branch_name = _sanitize_branch_name(
                        _format_template(git.branch_template.strip(), values=values)
                    )
                else:
                    branch_name = _sanitize_branch_name(prefix + run_id)
                p = _run_git(["checkout", "-b", branch_name])
                if p.returncode != 0:
                    raise ValueError(
                        "Failed to create git branch for run: "
                        f"{branch_name!r} stderr={(p.stderr or '').strip()!r}"
                    )
                active_branch = _git_current_branch() or branch_name
                state["git"]["run_branch"] = active_branch
                _write_json(state_path, state)
                timeline.append(
                    {
                        "type": "git_branch_created",
                        "run_id": run_id,
                        "branch": active_branch,
                        "original_branch": original_branch,
                    }
                )
                log(f"git: checked out {active_branch} (original={original_branch})")

        def _git_commit_step(
            *, step_name: str, actor_id: str, step_dir: Path, label: str
        ) -> str | None:
            """
            Optionally commit workspace changes after a step.

            This is only enabled when git policy is active in "branch" mode and auto_commit=True.
            Changes under orchestrator/ are always excluded from the commit.
            """
            if git is None or git.mode != "branch" or not git.auto_commit:
                return None

            clean, details = _git_is_clean()
            _ = clean
            kept, ignored = _git_filter_status_lines(details, allow_prefixes=("orchestrator/",))
            if not kept:
                timeline.append(
                    {
                        "type": "git_commit_skipped",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "reason": "no_changes",
                        "ignored_status_lines": list(ignored),
                    }
                )
                return None

            values: JsonDict = {
                "run_id": run_id,
                "step": step_name,
                "actor_id": actor_id,
                "label": label,
            }
            if git.commit_message_template is not None and git.commit_message_template.strip():
                msg = _format_template(git.commit_message_template.strip(), values=values).strip()
            else:
                msg = f"orch:{run_id} {step_name} {actor_id} {label}".strip()
            p = _run_git(["add", "-A", "--", ".", ":(exclude)orchestrator"])
            if p.returncode != 0:
                raise ValueError(f"git add failed: {(p.stderr or p.stdout or '').strip()!r}")

            staged = (_run_git(["diff", "--cached", "--name-only"]).stdout or "").strip()
            if not staged:
                timeline.append(
                    {
                        "type": "git_commit_skipped",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "reason": "nothing_staged",
                        "message": msg,
                    }
                )
                return None

            p = _run_git(["commit", "-m", msg])
            if p.returncode != 0:
                out = (p.stderr or p.stdout or "").strip()
                if "nothing to commit" in out.lower():
                    timeline.append(
                        {
                            "type": "git_commit_skipped",
                            "run_id": run_id,
                            "step": step_name,
                            "actor_id": actor_id,
                            "reason": "nothing_to_commit",
                            "message": msg,
                        }
                    )
                    return None
                raise ValueError(f"git commit failed: {out!r}")

            commit = (_run_git(["rev-parse", "HEAD"]).stdout or "").strip()
            if not commit:
                raise ValueError("git rev-parse HEAD returned empty after commit")

            changed = (
                _run_git(["diff-tree", "--no-commit-id", "--name-only", "-r", commit]).stdout
                or ""
            ).strip()
            info: JsonDict = {
                "commit": commit,
                "message": msg,
                "changed_paths": [p for p in changed.splitlines() if p.strip()],
            }
            _write_json(step_dir / "commit.json", info)

            # Update state step entry with commit id (best-effort).
            for entry in reversed(state.get("steps", [])):
                if isinstance(entry, dict) and entry.get("step") == step_name:
                    entry["commit"] = commit
                    break
            _write_json(state_path, state)

            timeline.append(
                {
                    "type": "git_commit_created",
                    "run_id": run_id,
                    "step": step_name,
                    "actor_id": actor_id,
                    "commit": commit,
                    "message": msg,
                    "changed_paths": info["changed_paths"],
                }
            )
            log(f"git: committed {commit[:12]} for {step_name} ({actor_id})")
            return commit

        def _provider_cfg_for_actor(actor_cfg: ActorConfig) -> ProviderConfig:
            return actor_cfg.provider or self.pipeline.provider

        def _effective_timeouts(cfg: ProviderConfig) -> tuple[float, float]:
            timeout_s = timeout_override if timeout_override is not None else cfg.timeout_s
            idle_timeout_s = (
                idle_timeout_override if idle_timeout_override is not None else cfg.idle_timeout_s
            )
            timeout_s = float(timeout_s)
            idle_timeout_s = float(idle_timeout_s)
            if timeout_s < 0:
                raise ValueError(f"timeout_s must be >= 0, got: {timeout_s}")
            if idle_timeout_s < 0:
                raise ValueError(f"idle_timeout_s must be >= 0, got: {idle_timeout_s}")
            return timeout_s, idle_timeout_s

        def _plan_provider_cfg() -> ProviderConfig:
            # Prefer the first non-deterministic provider in the pipeline, so auto-plan works
            # even when the pipeline default is deterministic but an actor override is not.
            for a in self.pipeline.actors:
                cfg = _provider_cfg_for_actor(a)
                if cfg.type != "deterministic":
                    return cfg
            return self.pipeline.provider

        plan_mode: PlanMode = "none"
        plan_text: str = ""
        if plan is not None:
            plan_mode = plan.mode
            if plan_mode == "user":
                plan_text = plan.text.strip()

        def _with_plan(inputs_md: str) -> str:
            if not plan_text.strip():
                return inputs_md if inputs_md.endswith("\n") else (inputs_md + "\n")
            if "## PLAN" in inputs_md.upper():
                return inputs_md if inputs_md.endswith("\n") else (inputs_md + "\n")
            out = inputs_md.rstrip() + "\n\n## PLAN\n\n" + plan_text.rstrip() + "\n"
            return out if out.endswith("\n") else (out + "\n")

        # Optional: generate a plan via provider before running steps.
        if plan is not None and plan.mode == "auto":
            plan_cfg = _plan_provider_cfg()
            if plan_cfg.type == "deterministic":
                log(
                    "WARN: plan mode 'auto' requested, but no non-deterministic provider is configured; skipping plan."
                )
                timeline.append(
                    {
                        "type": "plan_skipped",
                        "run_id": run_id,
                        "reason": "provider_is_deterministic",
                    }
                )
            else:
                provider = self._make_provider_from_cfg(plan_cfg)
                plan_timeout_s, plan_idle_timeout_s = _effective_timeouts(plan_cfg)
                goal = (self.pipeline.goal or "").strip()
                task_kind = "feature"
                task_details = ""
                if self.pipeline.task is not None:
                    task_kind = self.pipeline.task.kind
                    task_details = self.pipeline.task.details_md.strip()
                plan_prompt = "".join(
                    [
                        "You are producing an execution plan for a software task in a local workspace.\n\n",
                        "Constraints:\n",
                        "- Do not mention agents, orchestration, or pipeline order.\n",
                        "- Keep it language-agnostic.\n",
                        "- Be practical and actionable.\n",
                        "- Use plain markdown (no code fences).\n\n",
                        f"Workspace root: {self.orchestrator_dir.parent.as_posix()}\n\n",
                        f"Task type: {task_kind}\n",
                        "Task type notes:\n",
                        "- feature: avoid regressions; preserve existing behavior.\n",
                        "- bug: keep scope minimal; fix root cause; add regression coverage.\n",
                        "- bootstrap: define scope and acceptance criteria; create a runnable baseline.\n\n",
                        f"Goal:\n{goal or '(no goal provided)'}\n\n",
                        (f"Task details:\n{task_details}\n\n" if task_details else ""),
                        "Output:\n",
                        "- A short plan with sections: Investigation, Implementation, Validation, Risks.\n",
                        "- Keep it to a reasonable length.\n",
                    ]
                )

                plan_dir = run_dir / "plan" / "attempt_1"
                plan_dir.mkdir(parents=True, exist_ok=True)
                (plan_dir / "prompt.txt").write_text(plan_prompt, encoding="utf-8")
                timeline.append({"type": "plan_started", "run_id": run_id})

                def plan_event_cb(ev: JsonDict) -> None:
                    if on_event is None:
                        return
                    try:
                        on_event(
                            {
                                "type": "provider_event",
                                "run_id": run_id,
                                "step": "00_plan",
                                "actor_id": "plan",
                                "attempt": 1,
                                "event": ev,
                            }
                        )
                    except Exception:
                        pass

                try:
                    res = provider.run(
                        plan_prompt,
                        artifacts_dir=plan_dir,
                        timeout_s=plan_timeout_s,
                        idle_timeout_s=plan_idle_timeout_s,
                        on_event=plan_event_cb,
                    )
                    plan_text = res.final_text.strip()
                    (plan_dir / "plan.md").write_text(plan_text + "\n", encoding="utf-8")
                    timeline.append(
                        {
                            "type": "plan_finished",
                            "run_id": run_id,
                            "ok": bool(plan_text),
                            "provider_metadata": res.metadata,
                        }
                    )
                    if not plan_text:
                        log("WARN: auto plan generation returned empty output; continuing without plan.")
                except Exception as e:
                    timeline.append(
                        {
                            "type": "plan_failed",
                            "run_id": run_id,
                            "error": str(e),
                        }
                    )
                    log(f"WARN: auto plan generation failed ({e}); continuing without plan.")
                    plan_text = ""

        if plan_mode != "none":
            state["plan"] = {"mode": plan_mode, "present": bool(plan_text)}
            _write_json(state_path, state)
            if plan_text.strip():
                if plan_mode == "user":
                    pdir = run_dir / "plan"
                    pdir.mkdir(parents=True, exist_ok=True)
                    (pdir / "user_plan.md").write_text(plan_text.rstrip() + "\n", encoding="utf-8")
                # Inject the plan into all run-local packet inputs once up-front.
                for actor_cfg in self.pipeline.actors:
                    pdir = run_packets_dir / actor_cfg.packet_dir
                    ipath = pdir / "INPUTS.md"
                    try:
                        existing = ipath.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        existing = "# INPUTS\n\n"
                    ipath.write_text(_with_plan(existing), encoding="utf-8")

        def write_inputs(packet_dir: Path, content: str) -> None:
            path = packet_dir / "INPUTS.md"
            path.write_text(_with_plan(content), encoding="utf-8")

        def run_invocation(
            invocation_i: int, actor_cfg: ActorConfig
        ) -> tuple[str, str, StructuredReport | None, str, tuple[str, ...]]:
            actor_id = actor_cfg.actor_id
            step_name = f"{invocation_i:02d}_{actor_id}"
            step_dir = steps_dir / step_name
            step_dir.mkdir(parents=True, exist_ok=False)

            provider_cfg = _provider_cfg_for_actor(actor_cfg)
            provider = self._make_provider_from_cfg(provider_cfg)
            step_timeout_s, step_idle_timeout_s = _effective_timeouts(provider_cfg)

            log(f"[{invocation_i}] {actor_id}: step started")
            timeline.append(
                {
                    "type": "step_started",
                    "run_id": run_id,
                    "step": step_name,
                    "actor_id": actor_id,
                    "provider_type": provider_cfg.type,
                }
            )

            packet_dir = run_packets_dir / actor_cfg.packet_dir
            packet = Packet.load(packet_dir)
            report_format_text = packet.documents["REPORT_FORMAT"].text

            actor = Actor(
                actor_id=actor_id,
                packet=packet,
                provider=provider,
                include_paths_in_prompt=actor_cfg.include_paths_in_prompt,
            )

            attempt_errors: tuple[str, ...] = tuple()
            validated_report: StructuredReport | None = None
            validated_text: str = ""

            for attempt in (1, 2):
                attempt_dir = step_dir / f"attempt_{attempt}"
                log(f"  {actor_id}: attempt {attempt}/2")
                extra = None
                if attempt == 2:
                    extra = retry_instructions(attempt_errors)

                def provider_event_cb(
                    ev: JsonDict,
                    *,
                    _attempt: int = attempt,
                    _step: str = step_name,
                    _actor: str = actor_id,
                ) -> None:
                    if on_event is None:
                        return
                    try:
                        on_event(
                            {
                                "type": "provider_event",
                                "run_id": run_id,
                                "step": _step,
                                "actor_id": _actor,
                                "attempt": _attempt,
                                "event": ev,
                            }
                        )
                    except Exception:
                        pass

                res = actor.run(
                    artifacts_dir=attempt_dir,
                    timeout_s=step_timeout_s,
                    idle_timeout_s=step_idle_timeout_s,
                    extra_instructions=extra,
                    on_event=provider_event_cb,
                )

                validated_text = res.final_text
                v = validator.validate(
                    report_format_text=report_format_text, output_text=validated_text
                )
                _write_json(
                    attempt_dir / "validation.json",
                    {
                        "ok": v.ok,
                        "errors": list(v.errors),
                        "provider_metadata": res.metadata,
                    },
                )
                timeline.append(
                    {
                        "type": "attempt_finished",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "attempt": attempt,
                        "validation_ok": v.ok,
                        "provider_metadata": res.metadata,
                    }
                )

                if v.ok and v.report is not None:
                    log(f"  {actor_id}: attempt {attempt}/2 validation OK")
                    validated_report = v.report
                    break
                log(f"  {actor_id}: attempt {attempt}/2 validation FAILED")
                attempt_errors = v.errors

            # Canonical step outputs.
            (step_dir / "final.txt").write_text(validated_text, encoding="utf-8")
            if validated_report is not None:
                _write_json(
                    step_dir / "report.json",
                    {
                        "status": validated_report.status.value,
                        "output": validated_report.output,
                        "next_inputs": validated_report.next_inputs,
                        "artifacts": list(validated_report.artifacts),
                    },
                )

            # Append to NOTES.md (append-only within the run's packet copy).
            notes_path = packet_dir / "NOTES.md"
            with notes_path.open("a", encoding="utf-8") as f:
                f.write(f"\n## {run_id} {step_name}\n")
                if validated_report is None:
                    f.write("STATUS: FAILED (invalid report format)\n")
                    for e in attempt_errors:
                        f.write(f"- {e}\n")
                else:
                    f.write(f"STATUS: {validated_report.status.value}\n")

            if validated_report is None:
                state["steps"].append(
                    {
                        "step": step_name,
                        "actor_id": actor_id,
                        "status": "FAILED",
                        "reason": "invalid_report_format",
                        "errors": list(attempt_errors),
                    }
                )
                _write_json(state_path, state)
                timeline.append(
                    {
                        "type": "step_failed",
                        "run_id": run_id,
                        "step": step_name,
                        "actor_id": actor_id,
                        "reason": "invalid_report_format",
                        "errors": list(attempt_errors),
                    }
                )
                return step_name, actor_id, None, validated_text, attempt_errors

            # Step succeeded with a validated report.
            step_status = validated_report.status
            state["steps"].append(
                {
                    "step": step_name,
                    "actor_id": actor_id,
                    "status": step_status.value,
                }
            )
            _write_json(state_path, state)
            timeline.append(
                {
                    "type": "step_finished",
                    "run_id": run_id,
                    "step": step_name,
                    "actor_id": actor_id,
                    "status": step_status.value,
                }
            )
            return step_name, actor_id, validated_report, validated_text, attempt_errors

        final_report: StructuredReport | None = None
        final_text: str = ""
        overall_status = Status.OK

        orch = self.pipeline.orchestration
        if orch is None:
            # Default: run actors once, sequentially.
            for i, actor_cfg in enumerate(self.pipeline.actors, start=1):
                step_name, actor_id, report, text, _errors = run_invocation(i, actor_cfg)
                final_text = text
                if report is None:
                    overall_status = Status.FAILED
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "reason": "invalid_report_format",
                            "step": step_name,
                            "actor_id": actor_id,
                        }
                    )
                    break
                final_report = report
                if report.status == Status.OK:
                    _commit = _git_commit_step(
                        step_name=step_name,
                        actor_id=actor_id,
                        step_dir=(steps_dir / step_name),
                        label="ok",
                    )
                    if i < len(self.pipeline.actors):
                        next_cfg = self.pipeline.actors[i]
                        next_packet_dir = run_packets_dir / next_cfg.packet_dir
                        handoff = "# INPUTS\n\nUpstream report:\n\n" + report.output.rstrip() + "\n"
                        if report.next_inputs.strip():
                            handoff += "\nUpstream handoff:\n\n" + report.next_inputs.rstrip() + "\n"
                        if _commit:
                            handoff += "\nCommit:\n" f"- { _commit }\n"
                        write_inputs(next_packet_dir, handoff)
                    continue

                overall_status = report.status
                state["status"] = overall_status.value
                state["finished_at_utc"] = _now_iso_utc()
                _write_json(state_path, state)
                timeline.append(
                    {
                        "type": "run_stopped",
                        "run_id": run_id,
                        "status": overall_status.value,
                        "step": step_name,
                        "actor_id": actor_id,
                    }
                )
                break
        else:
            if orch.preset not in {"crt_v1", "cr_v1", "linear_v1"}:
                raise ValueError(f"Unsupported orchestration preset: {orch.preset!r}")

            stage_cfgs = list(self.pipeline.actors)
            if not stage_cfgs:
                raise ValueError("orchestration configured but pipeline has no actors")

            actor_ids = [a.actor_id for a in stage_cfgs]
            return_to_idx: int | None = None
            if orch.return_to is not None and orch.return_to in actor_ids:
                return_to_idx = actor_ids.index(orch.return_to)

            max_returns = orch.max_returns
            return_from = set(orch.return_from)
            returns_used = 0
            invocation_i = 0
            stage_i = 0
            last_commit: str | None = None

            while True:
                if stage_i < 0 or stage_i >= len(stage_cfgs):
                    overall_status = Status.FAILED
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "reason": "invalid_stage_index",
                            "stage_index": stage_i,
                        }
                    )
                    break

                invocation_i += 1
                actor_cfg = stage_cfgs[stage_i]
                step_name, actor_id, report, text, _errors = run_invocation(
                    invocation_i, actor_cfg
                )
                final_text = text

                if report is None:
                    overall_status = Status.FAILED
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "reason": "invalid_report_format",
                            "step": step_name,
                            "actor_id": actor_id,
                        }
                    )
                    break

                final_report = report

                if report.status == Status.OK:
                    commit = _git_commit_step(
                        step_name=step_name,
                        actor_id=actor_id,
                        step_dir=(steps_dir / step_name),
                        label="ok",
                    )
                    if commit:
                        last_commit = commit

                    if stage_i == (len(stage_cfgs) - 1):
                        overall_status = Status.OK
                        break

                    next_cfg = stage_cfgs[stage_i + 1]
                    next_packet_dir = run_packets_dir / next_cfg.packet_dir
                    parts: list[str] = ["# INPUTS", ""]
                    parts.append("Upstream report:")
                    parts.append("")
                    parts.append(report.output.rstrip())
                    parts.append("")
                    if report.next_inputs.strip():
                        parts.append("Upstream handoff:")
                        parts.append("")
                        parts.append(report.next_inputs.rstrip())
                        parts.append("")
                    if last_commit:
                        parts.append("Commit:")
                        parts.append(f"- {last_commit}")
                        parts.append("")
                        parts.append("Suggested commands:")
                        parts.append(f"- git show --stat {last_commit}")
                        parts.append(f"- git show {last_commit}")
                        parts.append("")
                    parts.append("Workspace:")
                    parts.append("- Inspect the workspace for the actual changes.")
                    write_inputs(next_packet_dir, "\n".join(parts).rstrip() + "\n")

                    stage_i += 1
                    continue

                if report.status == Status.NEEDS_INPUT:
                    overall_status = Status.NEEDS_INPUT
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "step": step_name,
                            "actor_id": actor_id,
                        }
                    )
                    break

                # FAILED
                can_return = (
                    return_to_idx is not None
                    and actor_id in return_from
                    and return_to_idx != stage_i
                    and max_returns >= 0
                )
                if not can_return:
                    overall_status = Status.FAILED
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "step": step_name,
                            "actor_id": actor_id,
                        }
                    )
                    break

                returns_used += 1
                if returns_used > max_returns:
                    overall_status = Status.NEEDS_INPUT
                    msg = (
                        "Escalation: exceeded max returns in workflow.\n"
                        f"returns_used={returns_used} max_returns={max_returns}\n"
                    )
                    final_report = StructuredReport(
                        status=Status.NEEDS_INPUT,
                        output=msg,
                        next_inputs=report.next_inputs or report.output,
                        artifacts=(),
                    )
                    final_text = (
                        json.dumps(
                            {
                                "status": final_report.status.value,
                                "output": final_report.output,
                                "next_inputs": final_report.next_inputs,
                                "artifacts": list(final_report.artifacts),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    state["status"] = overall_status.value
                    state["finished_at_utc"] = _now_iso_utc()
                    _write_json(state_path, state)
                    timeline.append(
                        {
                            "type": "run_stopped",
                            "run_id": run_id,
                            "status": overall_status.value,
                            "reason": "max_returns_exceeded",
                            "returns_used": returns_used,
                            "max_returns": max_returns,
                            "step": step_name,
                            "actor_id": actor_id,
                        }
                    )
                    break

                # Return feedback to return_to actor.
                assert return_to_idx is not None
                return_cfg = stage_cfgs[return_to_idx]
                return_packet_dir = run_packets_dir / return_cfg.packet_dir
                fb = ["# INPUTS", ""]
                if last_commit:
                    fb.append("Commit under review:")
                    fb.append(f"- {last_commit}")
                    fb.append("")
                fb.append("Feedback:")
                fb.append("")
                fb.append(report.output.rstrip())
                fb.append("")
                if report.next_inputs.strip():
                    fb.append("Requested changes:")
                    fb.append("")
                    fb.append(report.next_inputs.rstrip())
                    fb.append("")
                fb.append(f"Return count: {returns_used}/{max_returns}")
                write_inputs(return_packet_dir, "\n".join(fb).rstrip() + "\n")
                stage_i = return_to_idx
                continue

        if overall_status == Status.OK:
            state["status"] = overall_status.value
            state["finished_at_utc"] = _now_iso_utc()
            _write_json(state_path, state)
            timeline.append(
                {"type": "run_finished", "run_id": run_id, "status": overall_status.value}
            )

        (run_dir / "final.txt").write_text(final_text, encoding="utf-8")
        if final_report is not None:
            _write_json(
                run_dir / "final_report.json",
                {
                    "status": final_report.status.value,
                    "output": final_report.output,
                    "next_inputs": final_report.next_inputs,
                    "artifacts": list(final_report.artifacts),
                },
            )

        # Optional: emit merge request instructions at the end of the run.
        delivery = self.pipeline.delivery
        if delivery is not None and delivery.mr_mode == "instructions":
            git_info = state.get("git") if isinstance(state.get("git"), dict) else None
            run_branch = None
            if isinstance(git_info, dict):
                rb = git_info.get("run_branch")
                if isinstance(rb, str) and rb.strip():
                    run_branch = rb.strip()
            if run_branch is None:
                log("WARN: delivery.mr_mode=instructions requested, but no run_branch is recorded.")
                timeline.append(
                    {
                        "type": "mr_instructions_skipped",
                        "run_id": run_id,
                        "reason": "missing_run_branch",
                    }
                )
            else:
                slug = None
                if isinstance(git_info, dict):
                    s = git_info.get("branch_slug")
                    if isinstance(s, str) and s.strip():
                        slug = s.strip()
                if slug is None:
                    slug = _sanitize_slug(self.pipeline.goal or run_id)
                date_utc = time.strftime("%Y-%m-%d", time.gmtime())
                values: JsonDict = {
                    "run_id": run_id,
                    "goal": (self.pipeline.goal or "").strip(),
                    "slug": slug,
                    "branch": run_branch,
                    "remote": delivery.remote,
                    "target_branch": delivery.target_branch,
                    "date_utc": date_utc,
                }

                title_tpl = delivery.title_template or "{slug}: {goal}"
                title = _format_template(title_tpl, values=values).strip()
                if not title:
                    title = f"{slug}: {(self.pipeline.goal or run_id).strip()}"

                body_tpl = delivery.body_template or (
                    "Goal:\n{goal}\n\nRun:\n- run_id: {run_id}\n- branch: {branch}\n"
                )
                body = _format_template(body_tpl, values=values).rstrip() + "\n"

                mr_dir = run_dir / "delivery"
                mr_dir.mkdir(parents=True, exist_ok=True)
                body_path = mr_dir / "mr_body.md"
                body_path.write_text(body, encoding="utf-8")

                instr_path = mr_dir / "mr_instructions.md"
                lines: list[str] = []
                lines.append("# Merge Request Instructions")
                lines.append("")
                lines.append(f"- Remote: {delivery.remote}")
                lines.append(f"- Target branch: {delivery.target_branch}")
                lines.append(f"- Source branch: {run_branch}")
                lines.append("")
                lines.append("## Push")
                lines.append("")
                lines.append(f"git push -u {delivery.remote} {run_branch}")
                lines.append("")
                lines.append("## Create MR")
                lines.append("")
                lines.append("Title:")
                lines.append(title)
                lines.append("")
                lines.append("Body file:")
                lines.append(str(body_path))
                lines.append("")

                if shutil.which("gh") is not None:
                    lines.append("GitHub CLI (gh):")
                    lines.append(
                        f'gh pr create --base {delivery.target_branch} --head {run_branch} --title "{title}" --body-file "{body_path}"'
                    )
                    lines.append("")
                if shutil.which("glab") is not None:
                    lines.append("GitLab CLI (glab):")
                    lines.append(
                        f'glab mr create --target-branch {delivery.target_branch} --source-branch {run_branch} --title "{title}" --description-file "{body_path}"'
                    )
                    lines.append("")

                lines.append("Or create it in the web UI by comparing the branches above.")
                instr_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

                timeline.append(
                    {
                        "type": "mr_instructions_written",
                        "run_id": run_id,
                        "path": instr_path.as_posix(),
                        "remote": delivery.remote,
                        "target_branch": delivery.target_branch,
                        "source_branch": run_branch,
                    }
                )
                log(f"delivery: wrote MR instructions to {instr_path}")

        return RunOutcome(
            run_id=run_id,
            status=overall_status,
            run_dir=run_dir,
            final_report=final_report,
            final_text=final_text,
        )

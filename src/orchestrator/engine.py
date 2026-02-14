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
    ReviewReactionPolicy,
    generate_run_id,
    load_pipeline,
    orchestrator_root,
)
from src.core.types import JsonDict, Status, StructuredReport
from src.orchestrator.validator import ReportValidator, retry_instructions
from src.orchestrator.wizard import run_escalation_wizard
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


_RE_SEVERITY_TAG = re.compile(r"\\[(BLOCKER|MAJOR|MINOR|NIT)\\]")
_SEV_RANK: dict[str, int] = {"NIT": 0, "MINOR": 1, "MAJOR": 2, "BLOCKER": 3}


def _severity_counts(text: str) -> dict[str, int]:
    counts = {"BLOCKER": 0, "MAJOR": 0, "MINOR": 0, "NIT": 0}
    for m in _RE_SEVERITY_TAG.finditer(text):
        tag = m.group(1)
        if tag in counts:
            counts[tag] += 1
    return counts


def _merge_counts(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in ("BLOCKER", "MAJOR", "MINOR", "NIT")}


def _max_severity(counts: dict[str, int]) -> str | None:
    best: str | None = None
    best_rank = -1
    for tag in ("NIT", "MINOR", "MAJOR", "BLOCKER"):
        if int(counts.get(tag, 0)) <= 0:
            continue
        r = _SEV_RANK[tag]
        if r > best_rank:
            best_rank = r
            best = tag
    return best


def _evaluate_review_policy(
    policy: ReviewReactionPolicy, *, report: StructuredReport
) -> tuple[str, JsonDict]:
    """
    Returns (action, meta):
    - action: "default" | "return" | "escalate"
    """
    out_counts = _severity_counts(report.output or "")
    next_counts = _severity_counts(report.next_inputs or "")
    counts = _merge_counts(out_counts, next_counts)
    total = sum(counts.values())
    max_sev = _max_severity(counts)

    triggered = False
    if policy.trigger == "status_failed":
        triggered = report.status == Status.FAILED
    elif policy.trigger == "any_tag":
        triggered = total > 0
        if triggered and policy.min_severity is not None and max_sev is not None:
            if _SEV_RANK[max_sev] < _SEV_RANK[policy.min_severity]:
                triggered = False

    overflow = False
    if triggered:
        if policy.max_severity is not None and max_sev is not None:
            if _SEV_RANK[max_sev] > _SEV_RANK[policy.max_severity]:
                overflow = True
        if policy.max_blockers is not None and counts["BLOCKER"] > policy.max_blockers:
            overflow = True
        if policy.max_majors is not None and counts["MAJOR"] > policy.max_majors:
            overflow = True
        if policy.max_total is not None and total > policy.max_total:
            overflow = True

    action = "default"
    if triggered:
        action = policy.on_overflow if overflow else policy.on_trigger

    meta: JsonDict = {
        "policy_actor_id": policy.actor_id,
        "policy_trigger": policy.trigger,
        "policy_min_severity": policy.min_severity,
        "policy_max_severity": policy.max_severity,
        "policy_max_blockers": policy.max_blockers,
        "policy_max_majors": policy.max_majors,
        "policy_max_total": policy.max_total,
        "policy_on_trigger": policy.on_trigger,
        "policy_on_overflow": policy.on_overflow,
        "triggered": triggered,
        "overflow": overflow,
        "counts": counts,
        "total": total,
        "max_severity": max_sev,
        "report_status": report.status.value,
    }
    return action, meta


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
    workspace_dir: Path
    orchestrator_dir: Path
    pipeline: PipelineConfig
    task_id: str | None = None

    @staticmethod
    def load(*, workspace_dir: Path, task_id: str | None = None) -> OrchestratorEngine:
        workspace_dir = workspace_dir.resolve()
        orch_root = orchestrator_root(workspace_dir)
        tasks_root = orch_root / "tasks"

        def load_task_dir(tid: str) -> OrchestratorEngine:
            tdir = tasks_root / tid
            pipeline_path = tdir / "pipeline.json"
            if not pipeline_path.exists():
                raise ValueError(f"Task not found or missing pipeline.json: {tdir}")
            pipeline = load_pipeline(pipeline_path)
            return OrchestratorEngine(
                workspace_dir=workspace_dir,
                orchestrator_dir=tdir,
                pipeline=pipeline,
                task_id=tid,
            )

        if task_id is not None and task_id.strip():
            return load_task_dir(task_id.strip())

        current_path = orch_root / "CURRENT_TASK"
        if current_path.exists():
            tid = current_path.read_text(encoding="utf-8").strip()
            if tid:
                return load_task_dir(tid)

        if tasks_root.exists():
            task_dirs = sorted([p for p in tasks_root.iterdir() if p.is_dir()])
            candidates: list[str] = []
            for p in task_dirs:
                if (p / "pipeline.json").exists():
                    candidates.append(p.name)
            if len(candidates) == 1:
                return load_task_dir(candidates[0])

        # Legacy layout: <workspace>/orchestrator/pipeline.json
        legacy_pipeline_path = orch_root / "pipeline.json"
        if legacy_pipeline_path.exists():
            pipeline = load_pipeline(legacy_pipeline_path)
            return OrchestratorEngine(
                workspace_dir=workspace_dir,
                orchestrator_dir=orch_root,
                pipeline=pipeline,
                task_id=None,
            )

        raise ValueError(
            "No pipeline found. Expected one of:\n"
            "- <workspace>/orchestrator/pipeline.json (legacy)\n"
            "- <workspace>/orchestrator/CURRENT_TASK pointing to tasks/<task_id>/pipeline.json\n"
            "- <workspace>/orchestrator/tasks/<task_id>/pipeline.json (use --task)\n"
        )

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
            return CodexCLIProvider(command=cfg.command, cwd=self.workspace_dir)
        if cfg.type == "gemini_cli":
            assert cfg.command is not None
            return GeminiCLIProvider(command=cfg.command, cwd=self.workspace_dir)
        if cfg.type == "claude_cli":
            assert cfg.command is not None
            return ClaudeCLIProvider(command=cfg.command, cwd=self.workspace_dir)
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

        workspace_root = self.workspace_dir

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
            # Prefer wizard_provider when available, otherwise pick the first non-deterministic
            # provider in the pipeline so auto-plan works even when the pipeline default is
            # deterministic but an actor override is not.
            wiz = self.pipeline.wizard_provider
            if wiz is not None and wiz.type != "deterministic":
                return wiz
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
                        f"Workspace root: {self.workspace_dir.as_posix()}\n\n",
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

        qa_notes_file = self.orchestrator_dir / "QA_NOTES.md"

        def _read_qa_notes_for_prompt() -> tuple[str, str] | None:
            try:
                txt = qa_notes_file.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None
            if not txt.strip():
                return None
            body = txt.rstrip()
            max_chars = 12000
            if len(body) > max_chars:
                body = body[:max_chars].rstrip() + "\n\n[truncated; see full file]\n"
            return qa_notes_file.as_posix(), body + "\n"

        def _append_qa_notes(lines: list[str]) -> None:
            info = _read_qa_notes_for_prompt()
            if info is None:
                return
            path, body = info
            lines.append("QA notes:")
            lines.append(f"- path: {path}")
            lines.append("")
            lines.append(body.rstrip())
            lines.append("")

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
            last_parsed_report: StructuredReport | None = None
            last_failure_kind: str = "format"

            for attempt in (1, 2):
                attempt_dir = step_dir / f"attempt_{attempt}"
                log(f"  {actor_id}: attempt {attempt}/2")
                extra = None
                if attempt == 2:
                    if last_failure_kind == "qa_notes_postcondition":
                        extra = (
                            "Your previous response was valid ORCH_JSON_V1, but you did not satisfy a required postcondition.\n"
                            f"You MUST write a non-empty QA_NOTES.md at:\n{qa_notes_file.as_posix()}\n\n"
                            "QA_NOTES.md must include:\n"
                            "- change summary (what changed, where)\n"
                            "- what to verify (manual scenarios + automated checks)\n"
                            "- drift/regression areas\n"
                            "- known limitations / risky assumptions\n\n"
                            "Then return ORCH_JSON_V1 JSON (exactly one object; no extra text).\n"
                        )
                    else:
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

                post_ok = True
                post_errors: list[str] = []
                if v.ok and v.report is not None:
                    last_parsed_report = v.report
                    if actor_id == "qa_notes" and v.report.status == Status.OK:
                        try:
                            txt = qa_notes_file.read_text(encoding="utf-8")
                        except FileNotFoundError:
                            txt = ""
                        if not txt.strip():
                            post_ok = False
                            post_errors.append(
                                f"QA_NOTES.md is missing or empty at {qa_notes_file.as_posix()}"
                            )

                combined_ok = bool(v.ok and post_ok)
                combined_errors = list(v.errors) + post_errors
                _write_json(
                    attempt_dir / "validation.json",
                    {
                        "ok": combined_ok,
                        "errors": combined_errors,
                        "postcondition_ok": post_ok,
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
                        "validation_ok": combined_ok,
                        "postcondition_ok": post_ok,
                        "provider_metadata": res.metadata,
                    }
                )

                if combined_ok and v.report is not None:
                    log(f"  {actor_id}: attempt {attempt}/2 validation OK")
                    validated_report = v.report
                    break
                if not v.ok:
                    log(f"  {actor_id}: attempt {attempt}/2 validation FAILED")
                    last_failure_kind = "format"
                else:
                    log(f"  {actor_id}: attempt {attempt}/2 postcondition FAILED")
                    last_failure_kind = "qa_notes_postcondition"
                attempt_errors = tuple(combined_errors)

            if validated_report is None and actor_id == "qa_notes" and last_parsed_report is not None:
                # If the agent reported OK but did not produce QA_NOTES.md, stop with NEEDS_INPUT.
                try:
                    txt = qa_notes_file.read_text(encoding="utf-8")
                except FileNotFoundError:
                    txt = ""
                if last_parsed_report.status == Status.OK and not txt.strip():
                    msg = (
                        "Escalation: qa_notes step returned OK, but QA_NOTES.md is missing or empty.\n"
                        f"path={qa_notes_file.as_posix()}\n"
                    )
                    validated_report = StructuredReport(
                        status=Status.NEEDS_INPUT,
                        output=msg,
                        next_inputs=(
                            "Please allow the agent to write the QA notes file, or provide the QA notes content.\n"
                            "Minimum required content:\n"
                            "- change summary (what changed, where)\n"
                            "- what to verify (manual scenarios + automated checks)\n"
                            "- drift/regression areas\n"
                            "- known limitations / risky assumptions\n"
                        ),
                        artifacts=(),
                    )
                    validated_text = (
                        json.dumps(
                            {
                                "status": validated_report.status.value,
                                "output": validated_report.output,
                                "next_inputs": validated_report.next_inputs,
                                "artifacts": list(validated_report.artifacts),
                            },
                            indent=2,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    timeline.append(
                        {
                            "type": "qa_notes_missing",
                            "run_id": run_id,
                            "step": step_name,
                            "actor_id": actor_id,
                            "path": qa_notes_file.as_posix(),
                        }
                    )

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
        stop_reason: str | None = None
        stop_step: str | None = None
        stop_actor_id: str | None = None
        stop_report: StructuredReport | None = None
        stop_returns_used: int | None = None
        stop_max_returns: int | None = None
        stop_last_commit: str | None = None

        def mark_stop(
            *,
            reason: str,
            step: str,
            actor_id: str,
            report: StructuredReport | None,
            returns_used: int | None = None,
            max_returns: int | None = None,
            last_commit: str | None = None,
        ) -> None:
            nonlocal stop_reason, stop_step, stop_actor_id, stop_report
            nonlocal stop_returns_used, stop_max_returns, stop_last_commit
            stop_reason = reason
            stop_step = step
            stop_actor_id = actor_id
            stop_report = report
            stop_returns_used = returns_used
            stop_max_returns = max_returns
            stop_last_commit = last_commit

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
                        lines: list[str] = ["# INPUTS", ""]
                        lines.append("Upstream report:")
                        lines.append("")
                        lines.append(report.output.rstrip())
                        lines.append("")
                        if report.next_inputs.strip():
                            lines.append("Upstream handoff:")
                            lines.append("")
                            lines.append(report.next_inputs.rstrip())
                            lines.append("")
                        if _commit:
                            lines.append("Commit:")
                            lines.append(f"- {_commit}")
                            lines.append("")
                        _append_qa_notes(lines)
                        write_inputs(next_packet_dir, "\n".join(lines).rstrip() + "\n")
                    continue

                overall_status = report.status
                if overall_status == Status.NEEDS_INPUT:
                    mark_stop(
                        reason="agent_needs_input",
                        step=step_name,
                        actor_id=actor_id,
                        report=report,
                    )
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
            review_policies: dict[str, ReviewReactionPolicy] = {
                p.actor_id: p for p in orch.review_policies
            }
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

                # Optional policy: react to reviewer findings (tag-based) regardless of status.
                policy = review_policies.get(actor_id)
                if policy is not None:
                    action, meta = _evaluate_review_policy(policy, report=report)
                    if meta.get("triggered"):
                        timeline.append(
                            {
                                "type": "review_policy_evaluated",
                                "run_id": run_id,
                                "step": step_name,
                                "actor_id": actor_id,
                                "action": action,
                                "meta": meta,
                            }
                        )

                    if action == "escalate":
                        overall_status = Status.NEEDS_INPUT
                        counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
                        msg = (
                            "Escalation: review reaction policy requested escalation.\n"
                            f"actor_id={actor_id}\n"
                            f"trigger={meta.get('policy_trigger')}\n"
                            f"counts={counts}\n"
                            f"max_severity={meta.get('max_severity')}\n"
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
                        mark_stop(
                            reason="review_policy_escalation",
                            step=step_name,
                            actor_id=actor_id,
                            report=final_report,
                            returns_used=returns_used,
                            max_returns=max_returns,
                            last_commit=last_commit,
                        )
                        state["status"] = overall_status.value
                        state["finished_at_utc"] = _now_iso_utc()
                        _write_json(state_path, state)
                        timeline.append(
                            {
                                "type": "run_stopped",
                                "run_id": run_id,
                                "status": overall_status.value,
                                "reason": "review_policy_escalation",
                                "step": step_name,
                                "actor_id": actor_id,
                                "meta": meta,
                            }
                        )
                        break

                    if action == "return":
                        can_return = (
                            return_to_idx is not None
                            and actor_id in return_from
                            and return_to_idx != stage_i
                            and max_returns >= 0
                        )
                        if not can_return:
                            overall_status = Status.NEEDS_INPUT
                            msg = (
                                "Escalation: review reaction policy requested a return, "
                                "but orchestration return routing is not configured.\n"
                                f"actor_id={actor_id} return_to={orch.return_to or '(none)'}\n"
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
                            mark_stop(
                                reason="review_policy_return_not_possible",
                                step=step_name,
                                actor_id=actor_id,
                                report=final_report,
                                returns_used=returns_used,
                                max_returns=max_returns,
                                last_commit=last_commit,
                            )
                            state["status"] = overall_status.value
                            state["finished_at_utc"] = _now_iso_utc()
                            _write_json(state_path, state)
                            timeline.append(
                                {
                                    "type": "run_stopped",
                                    "run_id": run_id,
                                    "status": overall_status.value,
                                    "reason": "review_policy_return_not_possible",
                                    "step": step_name,
                                    "actor_id": actor_id,
                                    "meta": meta,
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
                            mark_stop(
                                reason="max_returns_exceeded",
                                step=step_name,
                                actor_id=actor_id,
                                report=final_report,
                                returns_used=returns_used,
                                max_returns=max_returns,
                                last_commit=last_commit,
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
                        counts = meta.get("counts") if isinstance(meta.get("counts"), dict) else {}
                        total_findings = meta.get("total")
                        if isinstance(total_findings, int) and total_findings > 0:
                            fb.append("Findings summary:")
                            fb.append(
                                f"- BLOCKER={counts.get('BLOCKER', 0)} "
                                f"MAJOR={counts.get('MAJOR', 0)} "
                                f"MINOR={counts.get('MINOR', 0)} "
                                f"NIT={counts.get('NIT', 0)}"
                            )
                            fb.append("")
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
                        timeline.append(
                            {
                                "type": "review_policy_return",
                                "run_id": run_id,
                                "from_step": step_name,
                                "from_actor_id": actor_id,
                                "to_actor_id": stage_cfgs[return_to_idx].actor_id,
                                "returns_used": returns_used,
                                "max_returns": max_returns,
                                "meta": meta,
                            }
                        )
                        stage_i = return_to_idx
                        continue

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
                    _append_qa_notes(parts)
                    parts.append("Workspace:")
                    parts.append("- Inspect the workspace for the actual changes.")
                    write_inputs(next_packet_dir, "\n".join(parts).rstrip() + "\n")

                    stage_i += 1
                    continue

                if report.status == Status.NEEDS_INPUT:
                    overall_status = Status.NEEDS_INPUT
                    mark_stop(
                        reason="agent_needs_input",
                        step=step_name,
                        actor_id=actor_id,
                        report=report,
                        returns_used=returns_used,
                        max_returns=max_returns,
                        last_commit=last_commit,
                    )
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
                    mark_stop(
                        reason="max_returns_exceeded",
                        step=step_name,
                        actor_id=actor_id,
                        report=final_report,
                        returns_used=returns_used,
                        max_returns=max_returns,
                        last_commit=last_commit,
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

        if overall_status == Status.NEEDS_INPUT and final_report is not None:
            # Convenience artifact for humans / tooling to resume with additional inputs.
            ni_path = run_dir / "NEEDS_INPUT.md"
            ni_parts: list[str] = []
            ni_parts.append("# NEEDS_INPUT")
            ni_parts.append("")
            ni_parts.append(f"run_id: {run_id}")
            if stop_step:
                ni_parts.append(f"step: {stop_step}")
            if stop_actor_id:
                ni_parts.append(f"actor_id: {stop_actor_id}")
            if stop_reason:
                ni_parts.append(f"reason: {stop_reason}")
            ni_parts.append("")
            if final_report.next_inputs.strip():
                ni_parts.append(final_report.next_inputs.rstrip())
            else:
                ni_parts.append("(no next_inputs provided)")
            ni_parts.append("")
            ni_parts.append("To continue:")
            ni_parts.append(
                "- Update the task packets under this task's packets/ directory (or re-run setup), then run again."
            )
            ni_path.write_text("\n".join(ni_parts).rstrip() + "\n", encoding="utf-8")

        # Optional: run an escalation wizard before stopping, to propose resolutions/subtasks.
        if (
            overall_status == Status.NEEDS_INPUT
            and stop_step is not None
            and stop_actor_id is not None
            and stop_report is not None
        ):
            packet_dir_by_actor = {a.actor_id: a.packet_dir for a in self.pipeline.actors}
            inputs_md = ""
            packet_dir_name = packet_dir_by_actor.get(stop_actor_id)
            if packet_dir_name is not None:
                try:
                    inputs_md = (run_packets_dir / packet_dir_name / "INPUTS.md").read_text(
                        encoding="utf-8"
                    )
                except Exception:
                    inputs_md = ""

            def escalation_provider_cfg() -> ProviderConfig | None:
                wiz = self.pipeline.wizard_provider
                if wiz is not None and wiz.type != "deterministic":
                    return wiz
                for a in self.pipeline.actors:
                    cfg = _provider_cfg_for_actor(a)
                    if cfg.type != "deterministic":
                        return cfg
                if self.pipeline.provider.type != "deterministic":
                    return self.pipeline.provider
                return None

            esc_cfg = escalation_provider_cfg()
            if esc_cfg is None:
                timeline.append(
                    {
                        "type": "escalation_wizard_skipped",
                        "run_id": run_id,
                        "reason": "no_non_deterministic_provider",
                        "step": stop_step,
                        "actor_id": stop_actor_id,
                    }
                )
                log("escalation: skipped (no non-deterministic provider configured).")
            else:
                esc_provider = self._make_provider_from_cfg(esc_cfg)
                esc_timeout_s, esc_idle_timeout_s = _effective_timeouts(esc_cfg)
                esc_attempt_dir = run_dir / "escalation" / "attempt_1"
                esc_attempt_dir.mkdir(parents=True, exist_ok=True)

                timeline.append(
                    {
                        "type": "escalation_wizard_started",
                        "run_id": run_id,
                        "step": stop_step,
                        "actor_id": stop_actor_id,
                        "reason": stop_reason or "needs_input",
                        "provider_type": esc_cfg.type,
                    }
                )

                def esc_event_cb(ev: JsonDict) -> None:
                    if on_event is None:
                        return
                    try:
                        on_event(
                            {
                                "type": "provider_event",
                                "run_id": run_id,
                                "step": "zz_escalation",
                                "actor_id": "wizard",
                                "attempt": 1,
                                "event": ev,
                            }
                        )
                    except Exception:
                        pass

                ok = False
                out_path = run_dir / "escalation" / "escalation.md"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    res, meta = run_escalation_wizard(
                        esc_provider,
                        workspace_dir=self.workspace_dir,
                        goal=(self.pipeline.goal or "").strip(),
                        task=self.pipeline.task,
                        run_id=run_id,
                        step=stop_step,
                        actor_id=stop_actor_id,
                        reason=stop_reason or "needs_input",
                        report=stop_report,
                        inputs_md=inputs_md,
                        last_commit=stop_last_commit,
                        returns_used=stop_returns_used,
                        max_returns=stop_max_returns,
                        artifacts_dir=esc_attempt_dir,
                        timeout_s=esc_timeout_s,
                        idle_timeout_s=esc_idle_timeout_s,
                        on_event=esc_event_cb,
                    )
                    ok = res is not None and bool(res.escalation_md.strip())
                    if res is None or not res.escalation_md.strip():
                        fallback = []
                        fallback.append("# Escalation")
                        fallback.append("")
                        fallback.append("The run stopped with NEEDS_INPUT, but the escalation wizard returned no result.")
                        fallback.append("")
                        fallback.append(f"reason: {stop_reason or 'needs_input'}")
                        fallback.append(f"step: {stop_step}")
                        fallback.append(f"actor_id: {stop_actor_id}")
                        fallback.append("")
                        fallback.append("Agent next_inputs:")
                        fallback.append("")
                        fallback.append((stop_report.next_inputs or "").rstrip() or "(empty)")
                        fallback.append("")
                        out_path.write_text("\n".join(fallback).rstrip() + "\n", encoding="utf-8")
                    else:
                        out_path.write_text(res.escalation_md.rstrip() + "\n", encoding="utf-8")

                    _write_json(
                        run_dir / "escalation" / "metadata.json",
                        {
                            "ok": ok,
                            "provider_type": esc_cfg.type,
                            "provider_metadata": meta,
                            "reason": stop_reason or "needs_input",
                            "step": stop_step,
                            "actor_id": stop_actor_id,
                        },
                    )
                    timeline.append(
                        {
                            "type": "escalation_wizard_finished",
                            "run_id": run_id,
                            "ok": ok,
                            "path": out_path.as_posix(),
                            "provider_metadata": meta,
                        }
                    )
                    state["escalation"] = {"present": True, "path": out_path.as_posix()}
                    _write_json(state_path, state)
                    log(f"escalation: wrote {out_path}")
                except Exception as e:
                    timeline.append(
                        {
                            "type": "escalation_wizard_failed",
                            "run_id": run_id,
                            "error": str(e),
                        }
                    )
                    log(f"WARN: escalation wizard failed ({e})")

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

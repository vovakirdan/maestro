from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.core.actor import Actor
from src.core.packet import Packet
from src.core.runtime import ActorConfig, PipelineConfig, generate_run_id, load_pipeline, orchestrator_root
from src.core.types import JsonDict, Status, StructuredReport
from src.orchestrator.validator import ReportValidator, retry_instructions
from src.providers.base import Provider
from src.providers.codex_cli import CodexCLIProvider
from src.providers.deterministic import DeterministicProvider


def _now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, obj: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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

    def _make_provider(self) -> Provider:
        cfg = self.pipeline.provider
        if cfg.type == "deterministic":
            return DeterministicProvider()
        if cfg.type == "codex_cli":
            assert cfg.command is not None
            return CodexCLIProvider(command=cfg.command, cwd=self.orchestrator_dir.parent)
        raise ValueError(f"Unsupported provider type: {cfg.type!r}")

    def run(
        self,
        *,
        progress: Callable[[str], None] | None = None,
        on_event: Callable[[JsonDict], None] | None = None,
        plan: PlanRequest | None = None,
        git: GitPolicy | None = None,
    ) -> RunOutcome:
        def log(msg: str) -> None:
            if progress:
                progress(msg)

        provider = self._make_provider()
        validator = ReportValidator()

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
            if not clean:
                raise ValueError(
                    "Workspace has uncommitted changes; commit/stash first or disable git policy. "
                    f"status_porcelain:\n{details}"
                )

            original_branch = _git_current_branch() or "unknown"
            state["git"] = {
                "mode": git.mode,
                "repo_root": repo_root.as_posix(),
                "original_branch": original_branch,
                "clean": True,
            }
            _write_json(state_path, state)
            timeline.append(
                {
                    "type": "git_checked",
                    "run_id": run_id,
                    "mode": git.mode,
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
                branch_name = prefix + run_id
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
            if self.pipeline.provider.type == "deterministic":
                log("WARN: plan mode 'auto' requested, but provider is deterministic; skipping plan.")
                timeline.append(
                    {
                        "type": "plan_skipped",
                        "run_id": run_id,
                        "reason": "provider_is_deterministic",
                    }
                )
            else:
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
                        timeout_s=self.pipeline.provider.timeout_s,
                        idle_timeout_s=self.pipeline.provider.idle_timeout_s,
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

            log(f"[{invocation_i}] {actor_id}: step started")
            timeline.append(
                {"type": "step_started", "run_id": run_id, "step": step_name, "actor_id": actor_id}
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
                    timeout_s=self.pipeline.provider.timeout_s,
                    idle_timeout_s=self.pipeline.provider.idle_timeout_s,
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
                    if i < len(self.pipeline.actors):
                        next_cfg = self.pipeline.actors[i]
                        next_packet_dir = run_packets_dir / next_cfg.packet_dir
                        handoff = "# INPUTS\n\nUpstream report:\n\n" + report.output.rstrip() + "\n"
                        if report.next_inputs.strip():
                            handoff += "\nUpstream handoff:\n\n" + report.next_inputs.rstrip() + "\n"
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
            if orch.preset != "crt_v1":
                raise ValueError(f"Unsupported orchestration preset: {orch.preset!r}")
            if len(self.pipeline.actors) != 3:
                raise ValueError("crt_v1 preset requires exactly 3 actors (coder, reviewer, tester)")

            coder_cfg, reviewer_cfg, tester_cfg = self.pipeline.actors
            max_returns = orch.max_returns
            returns_used = 0
            invocation_i = 0

            last_impl: StructuredReport | None = None

            next_role = "coder"
            while True:
                invocation_i += 1

                if next_role == "coder":
                    step_name, actor_id, report, text, _errors = run_invocation(
                        invocation_i, coder_cfg
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
                    if report.status != Status.OK:
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
                    last_impl = report

                    reviewer_packet_dir = run_packets_dir / reviewer_cfg.packet_dir
                    handoff = "# INPUTS\n\nImplementation report:\n\n" + report.output.rstrip() + "\n"
                    if report.next_inputs.strip():
                        handoff += "\nValidation notes:\n\n" + report.next_inputs.rstrip() + "\n"
                    handoff += "\nWorkspace:\n- Inspect the workspace for the actual changes.\n"
                    write_inputs(reviewer_packet_dir, handoff)
                    next_role = "reviewer"
                    continue

                if next_role == "reviewer":
                    step_name, actor_id, report, text, _errors = run_invocation(
                        invocation_i, reviewer_cfg
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

                    if report.status == Status.OK:
                        tester_packet_dir = run_packets_dir / tester_cfg.packet_dir
                        parts: list[str] = ["# INPUTS", ""]
                        if last_impl is not None:
                            parts.append("Implementation report:")
                            parts.append("")
                            parts.append(last_impl.output.rstrip())
                            parts.append("")
                            if last_impl.next_inputs.strip():
                                parts.append("Implementation handoff:")
                                parts.append("")
                                parts.append(last_impl.next_inputs.rstrip())
                                parts.append("")
                        parts.append("Review report:")
                        parts.append("")
                        parts.append(report.output.rstrip())
                        parts.append("")
                        if report.next_inputs.strip():
                            parts.append("Suggested focus:")
                            parts.append("")
                            parts.append(report.next_inputs.rstrip())
                            parts.append("")
                        parts.append("Workspace:")
                        parts.append("- Inspect the workspace for the actual changes.")
                        write_inputs(tester_packet_dir, "\n".join(parts).rstrip() + "\n")
                        next_role = "tester"
                        continue

                    # FAILED: return to coder with feedback, subject to max_returns.
                    returns_used += 1
                    if returns_used > max_returns:
                        overall_status = Status.NEEDS_INPUT
                        msg = (
                            "Escalation: exceeded max returns in preset crt_v1.\n"
                            f"returns_used={returns_used} max_returns={max_returns}\n"
                        )
                        final_report = StructuredReport(
                            status=Status.NEEDS_INPUT,
                            output=msg,
                            next_inputs=report.next_inputs or report.output,
                            artifacts=(),
                        )
                        final_text = json.dumps(
                            {
                                "status": final_report.status.value,
                                "output": final_report.output,
                                "next_inputs": final_report.next_inputs,
                                "artifacts": list(final_report.artifacts),
                            },
                            indent=2,
                            sort_keys=True,
                        ) + "\n"
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

                    coder_packet_dir = run_packets_dir / coder_cfg.packet_dir
                    fb = ["# INPUTS", ""]
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
                    write_inputs(coder_packet_dir, "\n".join(fb).rstrip() + "\n")
                    next_role = "coder"
                    continue

                # tester
                step_name, actor_id, report, text, _errors = run_invocation(invocation_i, tester_cfg)
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
                    overall_status = Status.OK
                    break
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

                # FAILED: return to coder with feedback, subject to max_returns.
                returns_used += 1
                if returns_used > max_returns:
                    overall_status = Status.NEEDS_INPUT
                    msg = (
                        "Escalation: exceeded max returns in preset crt_v1.\n"
                        f"returns_used={returns_used} max_returns={max_returns}\n"
                    )
                    final_report = StructuredReport(
                        status=Status.NEEDS_INPUT,
                        output=msg,
                        next_inputs=report.next_inputs or report.output,
                        artifacts=(),
                    )
                    final_text = json.dumps(
                        {
                            "status": final_report.status.value,
                            "output": final_report.output,
                            "next_inputs": final_report.next_inputs,
                            "artifacts": list(final_report.artifacts),
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n"
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

                coder_packet_dir = run_packets_dir / coder_cfg.packet_dir
                fb = ["# INPUTS", ""]
                fb.append("Feedback:")
                fb.append("")
                fb.append(report.output.rstrip())
                fb.append("")
                if report.next_inputs.strip():
                    fb.append("Repro / failing scenarios:")
                    fb.append("")
                    fb.append(report.next_inputs.rstrip())
                    fb.append("")
                fb.append(f"Return count: {returns_used}/{max_returns}")
                write_inputs(coder_packet_dir, "\n".join(fb).rstrip() + "\n")
                next_role = "coder"
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

        return RunOutcome(
            run_id=run_id,
            status=overall_status,
            run_dir=run_dir,
            final_report=final_report,
            final_text=final_text,
        )

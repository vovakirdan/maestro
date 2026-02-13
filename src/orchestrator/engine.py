from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

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

        def write_inputs(packet_dir: Path, content: str) -> None:
            path = packet_dir / "INPUTS.md"
            text = content
            if not text.endswith("\n"):
                text += "\n"
            path.write_text(text, encoding="utf-8")

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
